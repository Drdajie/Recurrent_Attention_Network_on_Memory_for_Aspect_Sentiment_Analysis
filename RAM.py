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

        memory = self.embed(text_raw_indices)
        memory, (_, _) = self.bi_lstm_context(memory, memory_len)
        memory = self.locationed_memory(memory, memory_len, left_len, aspect_len)

        #论文中
        aspect = self.embed(aspect_indices)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.unsqueeze(-1))
        et = torch.zeros_like(aspect).to(self.opt.device)

        batch_size = memory.size(0)
        seq_len = memory.size(1)
        #et = et.double()
        #aspect = aspect.float()
        memory = memory.float()
        #print(memory.dtype,'\n',et.dtype,'\n',aspect.dtype)
        for _ in range(self.opt.hops):
            a = torch.cat([memory,
                       torch.zeros(batch_size, seq_len, self.opt.embedding_dim,dtype=torch.float32).to(
                           self.opt.device) + et.unsqueeze(1),
                       torch.zeros(batch_size, seq_len, self.opt.embedding_dim,dtype=torch.float32).to(
                           self.opt.device) + aspect.unsqueeze(1)],
                      dim=-1)
            a = torch.tensor(a,dtype=torch.float32).to(self.opt.device)
            g = self.att_linear(a)
            alpha = F.softmax(g, dim=1)
            memory = memory.float()
            i = torch.bmm(alpha.transpose(1, 2), memory).squeeze(1)
            #i = torch.tensor(i,dtype=torch.float32).to(self.opt.device)
            #et = torch.tensor(et,dtype=torch.float32).to(self.opt.device)
            et = self.gru_cell(i, et)
        out = self.dense(et)
        return out