from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from .PatchTST_backbone import PatchTST_backbone
from .PatchTST_layers import series_decomp

class PatchTSTModel(nn.Module):
    def __init__(self, 
                 configs,
                 max_seq_len=1024, 
                 d_k=None, 
                 d_v=None, 
                 norm='BatchNorm', 
                 attn_dropout=0., 
                 act="gelu", 
                 key_padding_mask='auto',
                 padding_var=None, 
                 attn_mask=None, 
                 res_attention=True, 
                 pre_norm=False, 
                 store_attn=False, 
                 pe='zeros', 
                 learn_pe=True, 
                 pretrain_head=False, 
                 head_type = 'flatten', 
                 verbose=False, 
                 **kwargs):
        
        super().__init__()
        
        c_in = configs.enc_in
        c_out = getattr(configs, 'c_out', c_in)
        self.c_in = c_in
        self.c_out = c_out
        self.proj_head = nn.Identity() if c_out == c_in else nn.Linear(c_in, c_out)

        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        
        
        # model
        self.decomposition = decomposition
        self.decomp_module = series_decomp(kernel_size)
        self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                subtract_last=subtract_last, verbose=verbose, **kwargs)
        self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                subtract_last=subtract_last, verbose=verbose, **kwargs)

    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        res_init, trend_init = self.decomp_module(x)
        res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
        res = self.model_res(res_init)
        trend = self.model_trend(trend_init)
        x = res + trend
        x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        x = self.proj_head(x)
        return x