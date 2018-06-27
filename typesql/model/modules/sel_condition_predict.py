import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from net_utils import run_lstm, col_name_encode

class SelCondPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, gpu, db_content):
        super(SelCondPredictor, self).__init__()
        self.N_h = N_h
        self.gpu = gpu

        if db_content == 0:
            in_size = N_word+N_word/2
        else:
            in_size = N_word+N_word
        self.selcond_lstm = nn.LSTM(input_size=in_size, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.ty_num_out = nn.Linear(N_h, N_h)
        self.cond_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 5))
        self.selcond_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.num_type_att = nn.Linear(N_h, N_h)

        self.sel_att = nn.Linear(N_h, N_h)
        self.sel_out_K = nn.Linear(N_h, N_h)
        self.sel_out_col = nn.Linear(N_h, N_h)
        self.sel_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

        self.cond_col_att = nn.Linear(N_h, N_h)
        self.cond_col_out_K = nn.Linear(N_h, N_h)
        self.cond_col_out_col = nn.Linear(N_h, N_h)
        self.cond_col_out_sel = nn.Linear(N_h, N_h)
        self.col_att = nn.Linear(N_h, N_h)
        self.cond_col_out = nn.Sequential(nn.ReLU(), nn.Linear(N_h, 1))

        self.softmax = nn.Softmax() #dim=1


    def forward(self, x_emb_var, x_len, col_inp_var, col_len, x_type_emb_var, gt_sel):
        max_x_len = max(x_len)
        max_col_len = max(col_len)
        B = len(x_len)

        #Predict the selection condition
        x_emb_concat = torch.cat((x_emb_var, x_type_emb_var), 2)
        e_col, _ = run_lstm(self.selcond_name_enc, col_inp_var, col_len)
        h_enc, _ = run_lstm(self.selcond_lstm, x_emb_concat, x_len)

        #att_val: (B, max_col_len, max_x_len)
        sel_att_val = torch.bmm(e_col, self.sel_att(h_enc).transpose(1, 2))
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                sel_att_val[idx, :, num:] = -100
        sel_att = self.softmax(sel_att_val.view((-1, max_x_len))).view(B, -1, max_x_len)
        #K_sel_expand -> (B, max_number of col names in batch tables, hid_dim)
        K_sel_expand = (h_enc.unsqueeze(1) * sel_att.unsqueeze(3)).sum(2)
        sel_score = self.sel_out(self.sel_out_K(K_sel_expand) + \
                self.sel_out_col(e_col)).squeeze()

        for idx, num in enumerate(col_len):
            if num < max_col_len:
                sel_score[idx, num:] = -100

        # Predict the number of conditions
        #att_num_type_val:(B, max_col_len, max_x_len)
        att_num_type_val = torch.bmm(e_col, self.num_type_att(h_enc).transpose(1, 2))

        for idx, num in enumerate(col_len):
            if num < max_col_len:
                att_num_type_val[idx, num:, :] = -100
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                att_num_type_val[idx, :, num:] = -100

        #att_num_type: (B, max_col_len, max_x_len)
        att_num_type = self.softmax(att_num_type_val.view((-1, max_x_len))).view(B, -1, max_x_len)
        #h_enc.unsqueeze(1): (B, 1, max_x_len, hid_dim)
        #att_num_type.unsqueeze(3): (B, max_col_len, max_x_len, 1)
        #K_num_type (B, max_col_len, hid_dim)
        K_num_type = (h_enc.unsqueeze(1) * att_num_type.unsqueeze(3)).sum(2).sum(1)
        #K_cond_num: (B, hid_dim)
        #K_num_type (B, hid_dim)
        cond_num_score = self.cond_num_out(self.ty_num_out(K_num_type))

        #Predict the columns of conditions
        if gt_sel is None:
            gt_sel = np.argmax(sel_score.data.cpu().numpy(), axis=1)
        #gt_sel (B)
        chosen_sel_idx = torch.LongTensor(gt_sel)
        #aux_range (B) (0,1,...)
        aux_range = torch.LongTensor(range(len(gt_sel)))
        if x_emb_var.is_cuda:
            chosen_sel_idx = chosen_sel_idx.cuda()
            aux_range = aux_range.cuda()
        #chosen_e_col: (B, hid_dim)
        chosen_e_col = e_col[aux_range, chosen_sel_idx]
        #chosen_e_col.unsqueeze(2): (B, hid_dim, 1)
        #self.col_att(h_enc): (B, max_x_len, hid_dim)
        #att_sel_val: (B, max_x_len)
        att_sel_val = torch.bmm(self.col_att(h_enc), chosen_e_col.unsqueeze(2)).squeeze()

        col_att_val = torch.bmm(e_col, self.cond_col_att(h_enc).transpose(1, 2))
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                col_att_val[idx, :, num:] = -100
                att_sel_val[idx, num:] = -100
        sel_att = self.softmax(att_sel_val)
        K_sel_agg = (h_enc * sel_att.unsqueeze(2).expand_as(h_enc)).sum(1)
        col_att = self.softmax(col_att_val.view((-1, max_x_len))).view(B, -1, max_x_len)
        K_cond_col = (h_enc.unsqueeze(1) * col_att.unsqueeze(3)).sum(2)

        cond_col_score = self.cond_col_out(self.cond_col_out_K(K_cond_col)
                + self.cond_col_out_col(e_col)
                + self.cond_col_out_sel(K_sel_agg.unsqueeze(1).expand_as(K_cond_col))).squeeze()

        for b, num in enumerate(col_len):
            if num < max_col_len:
                cond_col_score[b, num:] = -100

        sel_cond_score = (cond_num_score, sel_score, cond_col_score)

        return sel_cond_score
