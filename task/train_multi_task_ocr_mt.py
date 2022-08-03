# coding: utf-8
import os
import sys
import time
import random
import string
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from utils import AttnLabelConverter, Averager
from dataset import hierarchical_dataset_3, AlignCollate_3, Batch_Balanced_Dataset_3, TextualPairDataset
from model import make_std_mask, Visual_Encoder, Textual_Encoder, Transformer_Encoder, Transformer_Decoder
from validate import validation_multi_task_ocr_mt

import logging
logging.basicConfig(level = logging.INFO, format = '%(message)s')
logger = logging.getLogger(__name__)
print = logger.info

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def multi_task_ocr_mt_train(opt):
    print('Load task multi_task_ocr_mt successfully.')
    
    # Split dataset if there are multiple datasets are used.
    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    print('-' * 80)

    # Load Dataset
    print('Loading dataset ...')
    train_dataset = Batch_Balanced_Dataset_3(opt)
    
    print('Length of train_dataset: {}'.format(len(train_dataset)))
    print('-' * 80)

    ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
    ## This part is used to load external textual parallel data
    ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
    # Load textual parallel data
    if opt.external_mt == 'yes':
        print('-' * 80)
        print('Loading textual parallel data ...')
        text_train_data = TextualPairDataset(opt.src_train_text, opt.tgt_train_text, opt)
        text_train_dataset = torch.utils.data.DataLoader(
            text_train_data, batch_size=opt.batch_size,
            shuffle=True,
            num_workers=int(opt.workers))
        text_train_loader = iter(text_train_dataset)
        external_text_epoch_num = 0
        print('Length of text_train_dataset: {}'.format(len(text_train_data)))
    print('Finished Loading Training Set.')

    AlignCollate_valid = AlignCollate_3(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_dataset, valid_dataset_log = hierarchical_dataset_3(root=opt.valid_data, opt=opt)
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    print(valid_dataset_log)
    print('Length of valid_dataset: {}'.format(len(valid_dataset)))
    print('-' * 80)
    
    print('Finished Loading Training and validation Data!')
    
    """ model configuration """
    print('-' * 80)
    print('Now in model configuration')
    src_converter = AttnLabelConverter(opt.src_character)
    tgt_converter = AttnLabelConverter(opt.tgt_character)
    opt.num_class = len(tgt_converter.character)
    opt.src_num_class = len(src_converter.character)
    opt.tgt_num_class = len(tgt_converter.character)

    if opt.rgb:
        opt.input_channel = 3
    
    # Construct Model information 
    ################################################################################
    # Define model modules

    ##### ##### ##### ##### ##### ##### #####
    # End-to-end TIT Part
    visual_encoder = Visual_Encoder(opt)
    encoder = Transformer_Encoder(opt)
    tgt_decoder = Transformer_Decoder(opt, opt_dim = opt.tgt_num_class)
    # External OCR Decoder
    src_decoder = Transformer_Decoder(opt, opt_dim = opt.src_num_class)
    # External MT Encoder
    textual_encoder = Textual_Encoder(opt)

    model_list = [visual_encoder, textual_encoder, encoder, src_decoder, tgt_decoder]
    model_name_list = ['visual_encoder', 'textual_encoder', 'encoder', 'src_decoder', 'tgt_decoder']

    # weight initialization
    for sub_model in model_name_list:
        for name, param in eval(sub_model).named_parameters():
            if 'localization_fc2' in name:
                continue
            try:
                if 'Transformer_encoder_layer' in name or 'Transformer_decoder_layer' in name \
                    or 'TransformerDecoder' in name or 'SequenceModeling' in name:
                    if param.dim() > 1:
                        init.xavier_uniform_(param)
                        continue
            except:
                pass
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if 'weight' in name:
                    param.data.fill_(1)
                continue

    # data parallel for multi-GPU
    visual_encoder = torch.nn.DataParallel(visual_encoder).to(device)
    textual_encoder = torch.nn.DataParallel(textual_encoder).to(device)
    encoder = torch.nn.DataParallel(encoder).to(device)
    src_decoder = torch.nn.DataParallel(src_decoder).to(device)
    tgt_decoder = torch.nn.DataParallel(tgt_decoder).to(device)
    
    visual_encoder.train()
    textual_encoder.train()
    encoder.train()
    src_decoder.train()
    tgt_decoder.train()
    
    # Set Up Loss
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    
    # loss averager
    loss_avg = Averager()
    vtmt_loss_avg = Averager()
    ocr_loss_avg = Averager()
    mt_loss_avg = Averager()
    
    filtered_parameters = []
    params_num = []
    for sub_model in model_name_list:
        for p in filter(lambda p: p.requires_grad, eval(sub_model).parameters()):
            filtered_parameters.append(p)
            params_num.append(np.prod(p.size()))
    print('Trainable params num : {}'.format(sum(params_num)))

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)

    """ start training """
    print('-' * 80)
    print('Start training ...')
    start_time = time.time()
    best_accuracy = -1
    best_bleu = -1
    best_valid_loss = 1000000
    iteration = -1
    previous_best_accuracy_iter = 0
    previous_best_bleu_iter = 0
    previous_best_valid_iter = 0
    old_time = time.time()

    while(True):
        iteration += 1
        image_tensors, src_labels, tgt_labels = train_dataset.get_batch()

        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        ## Textual Loading Part
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        if opt.external_mt == 'yes':
            try:
                text_src_labels, text_tgt_labels = text_train_loader.next()
            except:
                print('Textual Parallel Data: Start a new epoch!')
                text_train_loader = iter(text_train_dataset)
                text_src_labels, text_tgt_labels = text_train_loader.next()
                external_text_epoch_num += 1
        
        image = image_tensors.to(device)
        
        # Textual data transformation: For triple img-src-tgt
        src_text, src_length = src_converter.encode(src_labels, opt.src_level,  batch_max_length=opt.src_batch_max_length)
        tgt_text, tgt_length = tgt_converter.encode(tgt_labels, opt.tgt_level,  batch_max_length=opt.tgt_batch_max_length)
        
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        ## External Textual Loading Part
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        # Textual data transformation: For textual parallel
        if opt.external_mt == 'yes':
            textual_src_text, textual_src_length = src_converter.encode(text_src_labels, opt.src_level,  batch_max_length=opt.src_batch_max_length)
            textual_tgt_text, textual_tgt_length = tgt_converter.encode(text_tgt_labels, opt.tgt_level,  batch_max_length=opt.tgt_batch_max_length)

        batch_size = image.size(0)
        src_mask = make_std_mask(src_text[:, :-1], pad = 2)[0]
        tgt_mask = make_std_mask(tgt_text[:, :-1], pad = 2)[0]
        opt.src_mask = src_mask
        opt.tgt_mask = tgt_mask

        if opt.num_gpu > 1:
            print('Now processing tgt_mask to meet multi-gpu training ...')
            x_src_mask, y_src_mask = src_mask.size()
            new_src_mask = src_mask.repeat(opt.batch_size, 1)
            new_src_mask = new_src_mask.reshape(opt.batch_size, x_src_mask, y_src_mask)
            src_mask = new_src_mask
            
            x_tgt_mask, y_tgt_mask = tgt_mask.size()
            new_tgt_mask = tgt_mask.repeat(opt.batch_size, 1)
            new_tgt_mask = new_tgt_mask.reshape(opt.batch_size, x_tgt_mask, y_tgt_mask)
            tgt_mask = new_tgt_mask
                
        visual_feature = visual_encoder(input = image)
        textual_feature = textual_encoder(text = src_text[:, :-1])
        visual_contextual_feature = encoder(visual_feature)
        textual_contextual_feature = encoder(textual_feature)

        # print('run vtmt_preds ...')
        vtmt_preds = tgt_decoder(contextual_feature = visual_contextual_feature, text = tgt_text[:, :-1], tgt_mask = tgt_mask)
        # print('run ocr_preds ...')
        ocr_preds = src_decoder(contextual_feature = visual_contextual_feature, text = src_text[:, :-1], tgt_mask = src_mask)
        # print('run mt_preds ...')
        mt_preds = tgt_decoder(contextual_feature = textual_contextual_feature, text = tgt_text[:, :-1], tgt_mask = tgt_mask)

        # ###### Textual Parallel side data forward ...
        if opt.external_mt == 'yes':
            pure_textual_feature = textual_encoder(text = textual_src_text[:, :-1])
            pure_textual_contextual_feature = encoder(pure_textual_feature)
            pure_mt_preds = tgt_decoder(contextual_feature = pure_textual_contextual_feature, text = textual_tgt_text[:, :-1], tgt_mask = tgt_mask)
        
        # Save ground truth results both for teacher-forcing
        src_target = src_text[:, 1:]
        tgt_target = tgt_text[:, 1:]  # without [GO] Symbol
        if opt.external_mt == 'yes':
            textual_tgt_target = textual_tgt_text[:, 1:]
        
        # cost calculation for triple
        vtmt_cost = criterion(vtmt_preds.contiguous().view(-1, vtmt_preds.shape[-1]), tgt_target.contiguous().view(-1))
        ocr_cost = criterion(ocr_preds.contiguous().view(-1, ocr_preds.shape[-1]), src_target.contiguous().view(-1))
        mt_cost = criterion(mt_preds.contiguous().view(-1, mt_preds.shape[-1]), tgt_target.contiguous().view(-1))
        if opt.external_mt == 'yes':
            pure_mt_cost = criterion(pure_mt_preds.contiguous().view(-1, pure_mt_preds.shape[-1]), textual_tgt_target.contiguous().view(-1))

        weighted_vtmt_cost = opt.TIT_Weight * vtmt_cost
        weighted_ocr_cost = opt.OCR_Weight * ocr_cost
        if opt.external_mt == 'yes':
            weighted_mt_cost = opt.MT_Weight * (mt_cost + pure_mt_cost) / 2
        else:
            weighted_mt_cost = opt.MT_Weight * mt_cost

        cost = weighted_vtmt_cost + weighted_mt_cost + weighted_ocr_cost

        loss_avg.add(weighted_vtmt_cost)
        loss_avg.add(weighted_mt_cost)
        loss_avg.add(weighted_ocr_cost)
        
        vtmt_loss_avg.add(weighted_vtmt_cost)
        mt_loss_avg.add(weighted_mt_cost)
        ocr_loss_avg.add(weighted_ocr_cost)

        # print loss at each step ...
        duration_time = time.time() - old_time
        print_str=f'step = {iteration+1}, loss = {loss_avg.val():0.5f}, vtmt_loss = {vtmt_loss_avg.val():0.5f}, mt_loss = {mt_loss_avg.val():0.5f}, ocr_loss: {ocr_loss_avg.val():0.5f} duration = {duration_time:0.2f}s'
        old_time = time.time()
        print(print_str)
        print('-' * 100)
        
        visual_encoder.zero_grad()
        textual_encoder.zero_grad()
        encoder.zero_grad()
        src_decoder.zero_grad()
        tgt_decoder.zero_grad()

        cost.backward()
        torch.nn.utils.clip_grad_norm_(visual_encoder.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        torch.nn.utils.clip_grad_norm_(textual_encoder.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        torch.nn.utils.clip_grad_norm_(src_decoder.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        torch.nn.utils.clip_grad_norm_(tgt_decoder.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)

        optimizer.step()
        
        # # When Debug, just continue to see loss information.
        # print('Now is debuggin ...')
        # continue
        
        # validation part
        if (iteration + 1) % opt.valInterval == 0 or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0' 
            print('-' * 80)
            print('Now in validation on iteration {} ...'.format(iteration + 1))
            elapsed_time = time.time() - start_time
                
            visual_encoder.eval()
            textual_encoder.eval()
            encoder.eval()
            tgt_decoder.eval()
            src_decoder.eval()

            with torch.no_grad():
                valid_loss, current_accuracy, current_bleu, vtmt_preds_str, mt_preds_str, src_labels, tgt_labels, infer_time, length_of_data = validation_multi_task_ocr_mt(
                    model_list, criterion, valid_loader, src_converter, tgt_converter, opt)
            
            visual_encoder.train()
            textual_encoder.train()
            encoder.train()
            tgt_decoder.train()
            src_decoder.train()

            loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
            loss_avg.reset()

            current_model_log = f'{"Current_valid_loss":17s}: {valid_loss:0.5f}, {"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_bleu":17s}: {current_bleu:0.3f}'

            # keep best accuracy model (on valid dataset)
            if valid_loss <= best_valid_loss:
                print('Saving best_valid_loss model ...')
                best_valid_loss = valid_loss
                torch.save(visual_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'visual_encoder' + '.pth')
                torch.save(textual_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'textual_encoder' + '.pth')
                torch.save(encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'encoder' + '.pth')
                torch.save(src_decoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'src_decoder' + '.pth')
                torch.save(tgt_decoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'tgt_decoder' + '.pth')
                
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'visual_encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_final_' + 'visual_encoder' + '.pth')
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'textual_encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_final_' + 'textual_encoder' + '.pth')
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_final_' + 'encoder' + '.pth')
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'src_decoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_final_' + 'src_decoder' + '.pth')
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'tgt_decoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_final_' + 'tgt_decoder' + '.pth')

                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{previous_best_valid_iter}_' + 'visual_encoder' + '.pth')
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{previous_best_valid_iter}_' + 'textual_encoder' + '.pth')
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{previous_best_valid_iter}_' + 'encoder' + '.pth')
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{previous_best_valid_iter}_' + 'src_decoder' + '.pth')
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{previous_best_valid_iter}_' + 'tgt_decoder' + '.pth')

                previous_best_valid_iter = iteration + 1
            
            # keep best accuracy model (on valid dataset)  
            if current_accuracy >= best_accuracy:
                print('Saving best_accuracy model ...')
                best_accuracy = current_accuracy
                torch.save(visual_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'visual_encoder' + '.pth')
                torch.save(textual_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'textual_encoder' + '.pth')
                torch.save(encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'encoder' + '.pth')
                torch.save(src_decoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'src_decoder' + '.pth')
                torch.save(tgt_decoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'tgt_decoder' + '.pth')

                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'visual_encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_final_' + 'visual_encoder' + '.pth')
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'textual_encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_final_' + 'textual_encoder' + '.pth')
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_final_' + 'encoder' + '.pth')
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'src_decoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_final_' + 'src_decoder' + '.pth')
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'tgt_decoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_final_' + 'tgt_decoder' + '.pth')

                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{previous_best_accuracy_iter}_' + 'visual_encoder' + '.pth')
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{previous_best_accuracy_iter}_' + 'textual_encoder' + '.pth')
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{previous_best_accuracy_iter}_' + 'encoder' + '.pth')
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{previous_best_accuracy_iter}_' + 'src_decoder' + '.pth')
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{previous_best_accuracy_iter}_' + 'tgt_decoder' + '.pth')
                
                previous_best_accuracy_iter = iteration + 1
            
            # keep best bleu model (on valid dataset)  
            print('Current bleu: {}'.format(current_bleu))
            print('Current best_bleu: {}'.format(best_bleu))
            if current_bleu >= best_bleu:
                print('Saving best_bleu model ...')
                best_bleu = current_bleu
                torch.save(visual_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'visual_encoder' + '.pth')
                torch.save(textual_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'textual_encoder' + '.pth')
                torch.save(encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'encoder' + '.pth')
                torch.save(src_decoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'src_decoder' + '.pth')
                torch.save(tgt_decoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'tgt_decoder' + '.pth')
                
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'visual_encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_final_' + 'visual_encoder' + '.pth')
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'textual_encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_final_' + 'textual_encoder' + '.pth')
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_final_' + 'encoder' + '.pth')
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'src_decoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_final_' + 'src_decoder' + '.pth')
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'tgt_decoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_final_' + 'tgt_decoder' + '.pth')

                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{previous_best_bleu_iter}_' + 'visual_encoder' + '.pth')
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{previous_best_bleu_iter}_' + 'textual_encoder' + '.pth')
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{previous_best_bleu_iter}_' + 'encoder' + '.pth')
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{previous_best_bleu_iter}_' + 'src_decoder' + '.pth')
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{previous_best_bleu_iter}_' + 'tgt_decoder' + '.pth')
                
                previous_best_bleu_iter = iteration + 1
            
            best_model_log = f'{"Best_valid":17s}: {best_valid_loss:0.5f}, {"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_blue":17s}: {best_bleu:0.2f}'

            loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
            print(loss_model_log)

            # show part of predicted results
            dashed_line = '-' * 80
            print('Part of VTMT predicted and ground-truth results :')
            head = f'{"Ground Truth":25s} | {"Prediction":25s} | Match-up T/F'
            predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
            for gt, pred in zip(tgt_labels[:5], vtmt_preds_str[:5]):
                gt = gt[:gt.find('[s]')]
                pred = pred[:pred.find('[s]')]

                predicted_result_log += f'{gt:25s} | {pred:25s} | {str(pred == gt)}\n'
            predicted_result_log += f'{dashed_line}'
            print(predicted_result_log)

            dashed_line = '-' * 80
            print('Part of MT predicted and ground-truth results :')
            head = f'{"Ground Truth":25s} | {"Prediction":25s} | Match-up T/F'
            predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
            for gt, pred in zip(tgt_labels[:5], mt_preds_str[:5]):
                gt = gt[:gt.find('[s]')]
                pred = pred[:pred.find('[s]')]

                predicted_result_log += f'{gt:25s} | {pred:25s} | {str(pred == gt)}\n'
            predicted_result_log += f'{dashed_line}'
            print(predicted_result_log)
        
        ######################################################################
        # Monitor training procedure
        if (iteration + 1) % opt.valInterval == 0 or iteration == 0:
            print('See decoding results from training forward ...')
            _, tgt_preds_index = vtmt_preds.max(2)
            tgt_length_for_pred = torch.IntTensor([opt.tgt_batch_max_length] * batch_size).to(device)
            tgt_length_for_loss = torch.IntTensor([opt.tgt_batch_max_length] * batch_size).to(device)
            tgt_preds_str = tgt_converter.decode(tgt_preds_index, tgt_length_for_pred, opt.tgt_level)
            tgt_labels = tgt_converter.decode(tgt_text[:, 1:], tgt_length_for_loss, opt.tgt_level)

            print('-' * 80)
            print('Monitor Training Procedure. Prediction is given under teacher-forcing')
            dashed_line = '-' * 80
            head = f'{"Ground Truth":25s} | {"Prediction":25s} | Match-up T/F'
            predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
            for gt, pred in zip(tgt_labels[:5], tgt_preds_str[:5]):
                gt = gt[:gt.find('[s]')]
                pred = pred[:pred.find('[s]')]

                predicted_result_log += f'{gt:25s} | {pred:25s} | {str(pred == gt)}\n'
            predicted_result_log += f'{dashed_line}'
            print(predicted_result_log)

            ###### For MT monitoring
            print('See decoding results from training forward ...')
            _, tgt_preds_index = mt_preds.max(2)
            tgt_length_for_pred = torch.IntTensor([opt.tgt_batch_max_length] * batch_size).to(device)
            tgt_length_for_loss = torch.IntTensor([opt.tgt_batch_max_length] * batch_size).to(device)
            tgt_preds_str = tgt_converter.decode(tgt_preds_index, tgt_length_for_pred, opt.tgt_level)
            tgt_labels = tgt_converter.decode(tgt_text[:, 1:], tgt_length_for_loss, opt.tgt_level)

            print('-' * 80)
            print('Monitor Training Procedure. Prediction is given under teacher-forcing')
            dashed_line = '-' * 80
            head = f'{"Ground Truth":25s} | {"Prediction":25s} | Match-up T/F'
            predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
            for gt, pred in zip(tgt_labels[:5], tgt_preds_str[:5]):
                gt = gt[:gt.find('[s]')]
                pred = pred[:pred.find('[s]')]

                predicted_result_log += f'{gt:25s} | {pred:25s} | {str(pred == gt)}\n'
            predicted_result_log += f'{dashed_line}'
            print(predicted_result_log)

            ###### For OCRs monitoring
            print('See decoding results from training forward ...')
            _, src_preds_index = ocr_preds.max(2)
            src_length_for_pred = torch.IntTensor([opt.src_batch_max_length] * batch_size).to(device)
            src_length_for_loss = torch.IntTensor([opt.src_batch_max_length] * batch_size).to(device)
            src_preds_str = src_converter.decode(src_preds_index, src_length_for_pred, opt.src_level)
            src_labels = src_converter.decode(src_text[:, 1:], src_length_for_loss, opt.src_level)

            print('-' * 80)
            print('Monitor Training Procedure of OCR. Prediction is given under teacher-forcing')
            dashed_line = '-' * 80
            head = f'{"Ground Truth":25s} | {"Prediction":25s} | Match-up T/F'
            predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
            for gt, pred in zip(src_labels[:5], src_preds_str[:5]):
                gt = gt[:gt.find('[s]')]
                pred = pred[:pred.find('[s]')]

                predicted_result_log += f'{gt:25s} | {pred:25s} | {str(pred == gt)}\n'
            predicted_result_log += f'{dashed_line}'
            print(predicted_result_log)

        ######################################################################
        # save model per opt.saveInterval, originally 1e+5 iter
        if (iteration + 1) % opt.saveInterval == 0 or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0' 
            print('-' * 80)
            print('Saving model on set step of {} ...'.format(iteration + 1))

            torch.save(visual_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/iter_step_{iteration+1}_' + 'visual_encoder' + '.pth')
            torch.save(textual_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/iter_step_{iteration+1}_' + 'textual_encoder' + '.pth')
            torch.save(encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/iter_step_{iteration+1}_' + 'encoder' + '.pth')
            torch.save(src_decoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/iter_step_{iteration+1}_' + 'src_decoder' + '.pth')
            torch.save(tgt_decoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/iter_step_{iteration+1}_' + 'tgt_decoder' + '.pth')
        
        # Final Step and offer information
        if (iteration + 1) == opt.num_iter:
            print('end the training at step {}!'.format(iteration + 1))

            print('Remove iter_step_1_* model savings, which is just a model saving to see whether it could run normally.')
            os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/iter_step_1_' + 'visual_encoder' + '.pth')
            os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/iter_step_1_' + 'textual_encoder' + '.pth')
            os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/iter_step_1_' + 'encoder' + '.pth')
            os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/iter_step_1_' + 'src_decoder' + '.pth')
            os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/iter_step_1_' + 'tgt_decoder' + '.pth')
            
            sys.exit()
