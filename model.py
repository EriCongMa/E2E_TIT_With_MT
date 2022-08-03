import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import ResNet_FeatureExtractor
from modules.sequence_modeling import (
    BidirectionalLSTM, PositionalEncoding, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
)

import logging
logging.basicConfig(level = logging.INFO, format = '%(message)s')
logger = logging.getLogger(__name__)
print = logger.info

class Visual_Encoder(nn.Module):
    def __init__(self, opt):
        super(Visual_Encoder, self).__init__()
        self.opt = opt
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')
            self.Transformation = None

        self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        self.make_feature_dim = nn.Linear(int(opt.imgW/4+1), opt.src_batch_max_length+1)   # make the sequential length as source language
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1
        if opt.FeatureExtraction == 'ResNoLSTM':
            self.Res_LSTM = False
            self.cv_bi_lstm = torch.nn.Linear(self.FeatureExtraction_output, opt.hidden_size)
        if not opt.FeatureExtraction == 'ResNoLSTM':
            self.Res_LSTM = True
            self.cv_bi_lstm = nn.Sequential(
                    BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                    BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.cv_bi_lstm_output = opt.hidden_size    # Used to initialize later layer

    def forward(self, input, text=None, tgt_mask=None, is_train=True):
        if not self.Transformation is None:
            input = self.Transformation(input)

        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)
        assert len(visual_feature.size()) == 3
        if not int(self.opt.imgW/4+1) == self.opt.src_batch_max_length+1:
            print('Now in make dim section.')
            visual_feature = self.make_feature_dim(visual_feature.permute(0, 2, 1)).permute(0, 2, 1)
        visual_feature = self.cv_bi_lstm(visual_feature)
        return visual_feature

class Textual_Encoder(nn.Module):
    def __init__(self, opt, lang_type='src'):
        super(Textual_Encoder, self).__init__()
        self.opt = opt
        if lang_type == 'src':
            self.FeatureExtraction = Embeddings(opt.hidden_size, opt.src_num_class)
        elif lang_type == 'tgt':
            self.FeatureExtraction = Embeddings(opt.hidden_size, opt.tgt_num_class)
        else:
            print('Please check the lang_type in initialization of textual encoder.')

    def forward(self, text, input=None, tgt_mask=None, is_train=True):
        textual_feature = self.FeatureExtraction(text)
        
        return textual_feature

class Transformer_Encoder(nn.Module):
    def __init__(self, opt, opt_from = None):
        super(Transformer_Encoder, self).__init__()
        self.opt = opt
        self.SequenceModeling_input = opt.hidden_size
        self.SequenceModeling_output = opt.hidden_size
        self.EncoderPositionalEmbedding = PositionalEncoding(d_model=self.SequenceModeling_output, dropout = 0, max_len = max(opt.src_batch_max_length, opt.tgt_batch_max_length) + 2)
        self.Transformer_encoder_layer = TransformerEncoderLayer(d_model=self.SequenceModeling_input, nhead=8)
        self.SequenceModeling = TransformerEncoder(self.Transformer_encoder_layer, num_layers=6)

    def forward(self, visual_feature, input=None, text=None, tgt_mask=None, is_train=True):
        visual_feature = self.EncoderPositionalEmbedding(visual_feature)
        batch_mid_variable = visual_feature.permute(1, 0, 2)    # Make batch dimension in the middle
        contextual_feature = self.SequenceModeling(src=batch_mid_variable)
        contextual_feature = contextual_feature.permute(1, 0, 2)
        
        return contextual_feature

class Transformer_Decoder(nn.Module):
    def __init__(self, opt, opt_dim=None, opt_from = None):
        super(Transformer_Decoder, self).__init__()
        self.opt = opt
        self.Prediction_input = opt.hidden_size
        self.Prediction_output = opt.hidden_size
        if opt_dim is None:
            opt_dim = opt.tgt_num_class
        self.tgt_embedding = Embeddings(opt.hidden_size, opt_dim)
        self.DecoderPositionalEmbedding = PositionalEncoding(d_model=self.Prediction_output, dropout = 0, max_len = max(opt.src_batch_max_length, opt.tgt_batch_max_length) + 2)
        self.Transformer_decoder_layer = TransformerDecoderLayer(d_model=opt.hidden_size, nhead=8)
        self.Prediction_TransformerDecoder = TransformerDecoder(self.Transformer_decoder_layer, num_layers=6, opt = opt, output_dim=opt_dim)

    def forward(self, contextual_feature, text, tgt_mask, input=None, is_train=True):
        pred_feature = contextual_feature
        pred_feature = pred_feature.permute(1, 0, 2)    # Make batch dimension in the middle
        text_input = self.tgt_embedding(text)
        text_input = self.DecoderPositionalEmbedding(text_input)
        text_input = text_input.permute(1, 0, 2)
        pred_feature = self.Prediction_TransformerDecoder(tgt = text_input, memory = pred_feature, tgt_mask = tgt_mask, is_train = is_train)
        pred_feature = pred_feature.permute(1, 0, 2)    # Make batch dimension in the top
        prediction = pred_feature
        
        return prediction

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(tgt, pad):
        # Generate mask without considering pad information
        tgt_mask = subsequent_mask(tgt.size(-1))
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        return Variable(tgt_mask.cuda(), requires_grad=False)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

