# -*- coding: utf-8 -*-
# file: RAM.py
# author: Drdajie
# email: drdajie@gmail.com

from Tool_and_Layor import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class RAM(nn.Module):
    def locationed_memory(self, memory, memory_len, left_len, aspect_len):
        batch_size = memory.shape[0]
        seq_len = memory.shape[1]
        memory_len = memory_len.cpu().numpy()
        left_len = left_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        u = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for idx in range(left_len[i]):
                weight[i].append(1 - (left_len[i] - idx) / memory_len[i])
                u[i].append(idx - left_len[i])
            for idx in range(left_len[i], left_len[i] + aspect_len[i]):
                weight[i].append(1)
                u[i].append(0)
            for idx in range(left_len[i] + aspect_len[i], memory_len[i]):
                weight[i].append(1 - (idx - left_len[i] - aspect_len[i] + 1) / memory_len[i])
                u[i].append(idx - left_len[i] - aspect_len[i] + 1)
            for idx in range(memory_len[i], seq_len):
                weight[i].append(1)
                u[i].append(0)
        u = torch.tensor(u, dtype=memory.dtype).to(self.opt.device).unsqueeze(2)
        weight = torch.tensor(weight).to(self.opt.device).unsqueeze(2)
        v = memory * weight
        u = u.double()
        memory = torch.cat([v, u], dim=2)
        return memory

    def __init__(self, embedding_matrix, opt):
        super(RAM, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.bi_lstm_context = Dynamic_LSTM(opt.embedding_dim, opt.hidden_dim, num_layers=1, batch_first=True,
                                           bidirectional=True)
        self.att_linear = nn.Linear(opt.hidden_dim * 2 + 1 + opt.embedding_dim * 2, 1)
        self.gru_cell = nn.GRUCell(opt.hidden_dim * 2 + 1, opt.embedding_dim)
        self.dense = nn.Linear(opt.embedding_dim, opt.num_class)

    def forward(self, inputs):
        text_raw_indices, aspect_indices, text_left_indices = inputs['text'], inputs['aspect'], inputs['left_context']
        left_len = torch.sum(text_left_indices != 0, dim=-1)
        memory_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        nonzeros_aspect = aspect_len.float()
        #1_Memory ??????
        #1,1_?????? Memory ??????
        memory = self.embed(text_raw_indices)
        memory, (_, _) = self.bi_lstm_context(memory, memory_len)
        #1.2_??? Memory ???????????????
        memory = self.locationed_memory(memory, memory_len, left_len, aspect_len)

        #2_Recurrent Attention ??????
        #2.1_?????? aspect ???????????????????????? -> ?????? et ????????????????????????
        aspect = self.embed(aspect_indices)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.unsqueeze(-1))
        #2.2_?????? e0??????????????? 0 ?????????
        et = torch.zeros_like(aspect).to(self.opt.device)
        #2.3_????????????
        batch_size = memory.size(0)
        seq_len = memory.size(1)
        memory = memory.float()
        for _ in range(self.opt.hops):
            #2.3.1_??????
            #(1)_?????? [m,et-1,v??] -> ??????????????????????????????
            a = torch.cat([memory,
                       torch.zeros(batch_size, seq_len, self.opt.embedding_dim,dtype=torch.float32).to(
                           self.opt.device) + et.unsqueeze(1),
                       torch.zeros(batch_size, seq_len, self.opt.embedding_dim,dtype=torch.float32).to(
                           self.opt.device) + aspect.unsqueeze(1)],
                      dim=-1) #??????????????????????????????
            a = torch.tensor(a,dtype=torch.float32).to(self.opt.device)
            #(2)_?????? -> ???????????????????????????????????????????????????
            g = self.att_linear(a)
            #2.3.2_????????????
            alpha = F.softmax(g, dim=1)
            #2.3.3_?????? attention layer ????????????
            memory = memory.float()
            i = torch.bmm(alpha.transpose(1, 2), memory).squeeze(1)
            #2.3.4_?????? GRU ?????? et
            et = self.gru_cell(i, et)
        out = self.dense(et)
        return out