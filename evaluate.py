import os
import string
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import time
from utils import AttnLabelConverter
from dataset import AlignCollate, RawDataset
from model import make_std_mask, Visual_Encoder, Textual_Encoder, Transformer_Encoder, Transformer_Decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import logging
logging.basicConfig(level = logging.INFO, format = '%(message)s')
logger = logging.getLogger(__name__)
print = logger.info

def multi_task_ocr_mt_task_eval(opt):
    time_before_load_model = time.time()
    """ model configuration """
    src_converter = AttnLabelConverter(opt.src_character)
    tgt_converter = AttnLabelConverter(opt.tgt_character)
    opt.src_num_class = len(src_converter.character)
    opt.tgt_num_class = len(tgt_converter.character)

    if opt.rgb:
        opt.input_channel = 3
    visual_encoder = Visual_Encoder(opt)
    textual_encoder = Textual_Encoder(opt)
    encoder = Transformer_Encoder(opt)
    tgt_decoder = Transformer_Decoder(opt, opt_dim = opt.tgt_num_class)
    
    visual_encoder = torch.nn.DataParallel(visual_encoder).to(device)
    textual_encoder = torch.nn.DataParallel(textual_encoder).to(device)
    encoder = torch.nn.DataParallel(encoder).to(device)
    tgt_decoder = torch.nn.DataParallel(tgt_decoder).to(device)

    # Load Model
    print('loading pretrained model from %s' % opt.saved_model)
    visual_encoder.load_state_dict(torch.load(opt.saved_model + '_' + opt.saved_iter + '_' + 'visual_encoder' +'.pth', map_location=device))
    textual_encoder.load_state_dict(torch.load(opt.saved_model + '_' + opt.saved_iter + '_' + 'textual_encoder' +'.pth', map_location=device))
    encoder.load_state_dict(torch.load(opt.saved_model + '_' + opt.saved_iter + '_' + 'encoder' +'.pth', map_location=device))
    tgt_decoder.load_state_dict(torch.load(opt.saved_model + '_' + opt.saved_iter + '_' + 'tgt_decoder' +'.pth', map_location=device))

    AlignCollate_eval = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    eval_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    eval_loader = torch.utils.data.DataLoader(
        eval_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_eval, pin_memory=True)

    # Prediction
    visual_encoder.eval()
    textual_encoder.eval()
    encoder.eval()
    tgt_decoder.eval()
    sample_stat = 0
    with torch.no_grad():
        idx = 1
        time_before_decoding = time.time()
        for image_tensors, image_path_list in eval_loader:
            print('Now decoding batch {}'.format(idx))
            idx += 1
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            
            tgt_length_for_pred = torch.IntTensor([opt.tgt_batch_max_length] * batch_size).to(device)
            tgt_text_for_pred = torch.LongTensor(batch_size, opt.tgt_batch_max_length + 1).fill_(0).to(device)
            valid_tgt_mask = make_std_mask(tgt_text_for_pred[:, :], pad = 2)[0]

            tgt_decoder_input = tgt_text_for_pred
            vtmt_decoder_input = tgt_text_for_pred
            
            for i in range(opt.tgt_batch_max_length + 1):
                visual_feature = visual_encoder(input = image, is_train=False)
                visual_contextual_feature = encoder(visual_feature, is_train=False)
                vtmt_preds = tgt_decoder(contextual_feature = visual_contextual_feature, text = vtmt_decoder_input.long(), tgt_mask = valid_tgt_mask, is_train=False)
                
                _, vtmt_preds_index = vtmt_preds.max(2)
                
                if i+1 < opt.tgt_batch_max_length + 1:
                    vtmt_decoder_input[:, i+1] = vtmt_preds_index[:, i]
                
                _, vtmt_preds_index = vtmt_preds.max(2)
                vtmt_preds_str = tgt_converter.decode(vtmt_preds_index, tgt_length_for_pred, opt.tgt_level)
                
            
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}'
            
            print(f'{dashed_line}\n{head}\n{dashed_line}')

            # Writting Target Results
            vtmt_preds_prob = F.softmax(vtmt_preds, dim=2)
            vtmt_preds_max_prob, _ = vtmt_preds_prob.max(dim=2)
            log = open(f'{opt.tgt_output}', 'a', encoding='utf-8')
            for img_name, pred, _ in zip(image_path_list, vtmt_preds_str, vtmt_preds_max_prob):
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])

                print(f'{img_name:25s}\t{pred:25s}')
                log.write(f'{pred}\n')
                sample_stat += 1
            log.close()
        
        time_after_decoding = time.time()
        duration_load_model_decoding = time_after_decoding - time_before_load_model
        duration_decoding = time_after_decoding - time_before_decoding

        num_samples = sample_stat
        print('Overall decoding sentences: {}'.format(num_samples))
        print('Time from load model to decoding: {}.'.format(duration_load_model_decoding))
        print('Time of decoding: {}.'.format(duration_decoding))
        print('Speed from load model to decoding: {} sentences/second.'.format(num_samples / duration_load_model_decoding))
        print('Speed of decoding: {} sentences/second.'.format(num_samples / duration_decoding))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default=None, help="Task parameter: None for normal training; others for special tasks")
    
    parser.add_argument('--image_folder', default=None, help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--src_batch_max_length', type=int, default=25, help='maximum-label-length of src')
    parser.add_argument('--tgt_batch_max_length', type=int, default=25, help='maximum-label-length of tgt')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    parser.add_argument('--saved_iter', required=True, help="iter step when saving model")
    
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    
    parser.add_argument('--Transformation', type=str, required=False, default='TPS', help='Transformation stage.')
    parser.add_argument('--FeatureExtraction', type=str, required=False, default='ResNoLSTM', help='FeatureExtraction stage.')
    parser.add_argument('--SequenceModeling', type=str, required=False, default='TransformerEncoder', help='SequenceModeling stage.')
    parser.add_argument('--Prediction', type=str, required=False, default='TransformerDecoder', help='Prediction stage.')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--tgt_output', help='output the target result log', default='log_test_result.txt')
    parser.add_argument('--data_format', type=str, default="pic", help='pic or lmdb')

    parser.add_argument('--src_vocab', required=True, help='path to source vocab')
    parser.add_argument('--tgt_vocab', required=True, help='path to target vocab')
    
    parser.add_argument('--src_level', type=str, default="char", help='char or word of src')
    parser.add_argument('--tgt_level', type=str, default="char", help='char or word of tgt')

    opt = parser.parse_args()

    def pic_file_texts(file):
        id=[]
        for line in open(file, 'r', encoding='utf-8'):
            line = line.replace("\n", "")
            id.append(line)
        return id

    if not opt.src_vocab:
    	print('not find src vocab!')
    src_dict_ = pic_file_texts(opt.src_vocab)
    opt.src_character = src_dict_

    if not opt.tgt_vocab:
    	print('not find tgt vocab!')
    tgt_dict_ = pic_file_texts(opt.tgt_vocab)
    opt.tgt_character = tgt_dict_

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    if opt.task == 'multi_task_ocr_mt':
        multi_task_ocr_mt_task_eval(opt)
    
    else:
        print('Input task {} is not defined. Please check your task name.'.format(opt.task))
        exit()