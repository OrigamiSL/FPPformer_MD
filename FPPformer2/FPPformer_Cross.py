# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
from FPPformer2.Modules import *
from FPPformer2.embed import DataEmbedding
from utils.RevIN import RevIN


class Encoder_process(nn.Module):
    def __init__(self, patch_size, encoder_num, encoders1, encoders2):
        super(Encoder_process, self).__init__()
        self.patch_size = patch_size
        self.encoder_num = encoder_num
        self.encoders1 = encoders1
        self.encoders2 = encoders2

    def forward(self, x_enc, var_mask):
        B, V, L, D = x_enc.shape
        x_patch_attn = x_enc.contiguous().view(B, V, -1, self.patch_size, D)

        encoder_out_list = []
        for i in range(self.encoder_num):
            if self.encoders1 is not None:
                _, x_patch_attn = self.encoders1[i](x_patch_attn, var_mask)
            x_out, x_patch_attn = self.encoders2[i](x_patch_attn, var_mask)
            encoder_out_list.append(x_out)
        return encoder_out_list


class Encoder_map(nn.Module):
    def __init__(self, input_len, pred_len, encoder_layer=3, layer_stack=2, patch_size=12, d_model=4, mra_level=3, dropout=0.05):
        super(Encoder_map, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.patch_size = patch_size
        self.encoder_num = encoder_layer
        self.d_model = d_model

        self.Embed1 = DataEmbedding(mra_level + 1, d_model)
        if layer_stack == 2:
            self.encoders1 = [Encoder_Cross(self.patch_size * 2 ** i, d_model, dropout, split=False)
                              for i in range(encoder_layer)]
            self.encoders1 = nn.ModuleList(self.encoders1)
        else:
            self.encoders1 = None
        self.encoders2 = [Encoder_Cross(self.patch_size * 2 ** i, d_model, dropout, split=True)
                          for i in range(encoder_layer)]
        self.encoders2 = nn.ModuleList(self.encoders2)
        self.encoder_process = Encoder_process(patch_size, self.encoder_num, self.encoders1, self.encoders2)

    def forward(self, x, mra_x, var_mask):
        current_x = torch.cat([x.unsqueeze(-1), mra_x], dim=-1)
        x_enc = self.Embed1(current_x).transpose(1, 2)

        encoder_out_list = self.encoder_process(x_enc, var_mask)
        return encoder_out_list


class Decoder_process(nn.Module):
    def __init__(self, input_len, pred_len, encoder_layer=3, layer_stack=2, patch_size=12, d_model=4, dropout=0.05):
        super(Decoder_process, self).__init__()
        self.pred_len = pred_len
        self.patch_size = patch_size
        self.encoder_num = encoder_layer
        self.d_model = d_model
        self.b_patch_size = self.patch_size * 2 ** (self.encoder_num - 1)

        self.Embed2 = DataEmbedding(1, d_model, start_pos=input_len)
        if layer_stack == 2:
            self.decoders1 = [Decoder(self.patch_size * 2 ** (encoder_layer - 1 - i), d_model, dropout, split=False)
                              for i in range(encoder_layer)]
            self.decoders1 = nn.ModuleList(self.decoders1)
        else:
            self.decoders1 = None
        self.decoders2 = [Decoder(self.patch_size * 2 ** (encoder_layer - 1 - i), d_model, dropout, split=True)
                          for i in range(encoder_layer)]
        self.decoders2 = nn.ModuleList(self.decoders2)
        self.projection = nn.Linear(d_model, 1)

    def forward(self, x_enc, y):
        B, L, V = y.shape
        x_dec = y.unsqueeze(-1)
        x_dec = self.Embed2(x_dec).transpose(1, 2). \
            expand(B, V, L, self.d_model)  # [B V L_pred D]

        x_dec = x_dec.contiguous().view(B, V, -1, self.b_patch_size, self.d_model)

        for i in range(self.encoder_num):
            if self.decoders1 is not None:
                x_dec = self.decoders1[i](x_dec, x_enc[-1 - i])
            x_dec = self.decoders2[i](x_dec, x_enc[-1 - i])

        x_map = self.projection(x_dec.contiguous().view(B, V, -1, self.d_model)).squeeze(-1)
        x_out = x_map.transpose(1, 2)

        return x_out[:, :self.pred_len, :]


class FPPformer_Cross(nn.Module):
    def __init__(self, input_len, pred_len, encoder_layer, layer_stack, patch_size, d_model,
                 mra_level, dropout, decoder_IN):
        super(FPPformer_Cross, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.patch_size = patch_size
        self.encoder_num = encoder_layer
        self.d_model = d_model
        self.layer_stack = layer_stack
        self.DIN = decoder_IN

        self.b_patch_size = self.patch_size * 2 ** (self.encoder_num - 1)
        # tackle the problem when the prediction sequence length is not the multiple integer of
        # the patch size at certain stage
        self.total_len = math.ceil(self.pred_len / self.b_patch_size) * self.b_patch_size

        self.revin = RevIN()
        self.mra_in = nn.LayerNorm(input_len, elementwise_affine=False)
        self.Encoder_process = (
            Encoder_map(input_len, pred_len, encoder_layer, layer_stack, patch_size, d_model, mra_level, dropout))
        self.Decoder_process = (
            Decoder_process(input_len, pred_len, encoder_layer, layer_stack, patch_size, d_model, dropout))

    def forward(self, x, mra_x, var_sp_matrix):
        B, _, V = x.shape
        self.revin(x, 'stats')
        x_enc = self.revin(x, 'norm')  # [B L V+top10%]
        mra_x = self.mra_in(mra_x.transpose(1, 3)).transpose(1, 3)
        enc_list = self.Encoder_process(x_enc, mra_x, var_sp_matrix)
        y = torch.zeros([B, self.pred_len, V]).to(x.device)
        # Only when the prediction sequence length is not the multiple integer of the patch size at certain stage
        if self.total_len > self.pred_len:
            zero_pad = torch.zeros([B, self.total_len - self.pred_len, V]).to(y.device)
            y = torch.cat([y, zero_pad], dim=1)

        if self.DIN:
            y_enc = self.revin(y, 'norm')  # [B L V]
        else:
            y_enc = y
        x_out = self.Decoder_process(enc_list, y_enc)
        x_out = self.revin(x_out, 'denorm')

        return x_out
