import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from net_utils import run_lstm, col_name_encode


class AggPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth):
        super(AggPredictor, self).__init__()

        self.agg_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.agg_col_name_enc = nn.LSTM(input_size=N_word+N_h,
                hidden_size=N_h/2, num_layers=N_depth,
                batch_first=True, dropout=0.3, bidirectional=True)
        self.agg_att = nn.Linear(N_h, N_h)
        self.sel_att = nn.Linear(N_h, N_h)
        self.agg_out_se = nn.Linear(N_word, N_h)
        self.agg_out_agg = nn.Linear(N_word, N_h)
        self.agg_out_K = nn.Linear(N_h, N_h)
        self.agg_out_f = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))
        self.softmax = nn.Softmax() #dim=1

    def forward(self, x_emb_var, x_len, agg_emb_var, col_inp_var=None,
            col_len=None):
        B = len(x_emb_var)
        max_x_len = max(x_len)

        h_enc, _ = run_lstm(self.agg_lstm, x_emb_var, x_len)

        agg_enc = self.agg_out_agg(agg_emb_var)
        #agg_enc: (B, 6, hid_dim)
        #self.sel_att(h_enc) -> (B, max_x_len, hid_dim) .transpose(1, 2) -> (B, hid_dim, max_x_len)
        #att_val_agg: (B, 6, max_x_len)
        att_val_agg = torch.bmm(agg_enc, self.sel_att(h_enc).transpose(1, 2))

        for idx, num in enumerate(x_len):
            if num < max_x_len:
                att_val_agg[idx, :, num:] = -100

        #att_agg: (B, 6, max_x_len)
        att_agg = self.softmax(att_val_agg.view((-1, max_x_len))).view(B, -1, max_x_len)
        #h_enc.unsqueeze(1) -> (B, 1, max_x_len, hid_dim)
        #att_agg.unsqueeze(3) -> (B, 6, max_x_len, 1)
        #K_agg_expand -> (B, 6, hid_dim)
        K_agg_expand = (h_enc.unsqueeze(1) * att_agg.unsqueeze(3)).sum(2)
        #agg_score = self.agg_out(K_agg)
        agg_score = self.agg_out_f(self.agg_out_se(agg_emb_var) + self.agg_out_K(K_agg_expand)).squeeze()

        return agg_score
