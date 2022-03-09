from glove import embed_matrix
import os, sys
import json
import random
import numpy as np
import string
import re
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.loss import MSELoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class wicMeta(torch.nn.Module):

    def __init__(self):
        super(wicMeta, self).__init__()
        embed_matrix = torch.load('glove/embedding_matrix.pt')
        vocab_dim, embed_dim = embed_matrix.shape
        self.embedding_layer = nn.Embedding(vocab_dim, embed_dim)
        self.embedding_layer = nn.Embedding(26809, embed_dim)
        # self.embedding_layer = self.embedding_layer.from_pretrained(embed_matrix, freeze=True)

        self.loss = nn.CrossEntropyLoss()
        # self.loss = nn.BCELoss()


    def prediction(self, sentences1, sentences2, lengths, targets, masks1, masks2):
        with torch.no_grad():
            _, output = self(sentences1, sentences2, lengths, targets, masks1, masks2)
            output = output.argmax(-1)
            return output

    def eval(self, sentences1, sentences2, lengths, targets, masks1, masks2, labels):
        with torch.no_grad():
            logits, output = self(sentences1, sentences2, lengths, targets, masks1, masks2, training=False)
            loss = self.loss(logits, labels)
            output = output.argmax(-1)
            # output = output.ge(0.5).float()
            accuracy = torch.mean((output==labels).float())
            return loss.item(), accuracy.item()

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location='cpu'))


class ModelA1M1(wicMeta):

    def __init__(self,
                 hidden_dim,
                 output_dim,
                 lr = 1e-3,
                 bias=False):
        super(ModelA1M1, self).__init__()
        embed_dim = self.embedding_layer.embedding_dim
        self.fc1 = nn.Linear(embed_dim * 2, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2, bias=bias)
        self.fc_out = nn.Linear(hidden_dim//2, output_dim, bias=bias)
        # self.fc_out = nn.Linear(embed_dim * 2, output_dim, bias=bias)
        self.dropout = nn.Dropout(0.95)
        # self.loss = nn.BCELoss()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.0)

    def forward(self, sentences1, sentences2, lengths, targets, masks1, masks2, training=True):
        
        # Word embedding
        embedding1 = self.embedding_layer(sentences1)
        embedding2 = self.embedding_layer(sentences2)
        # Target masking
        mask1 = torch.unsqueeze(masks1, -1)
        mask2 = torch.unsqueeze(masks2, -1)
        mask1 = mask1.expand(-1, -1, embedding1.shape[-1])
        mask2 = mask2.expand(-1, -1, embedding2.shape[-1])
        
        masked1 = torch.mul(embedding1, mask1)
        masked2 = torch.mul(embedding2, mask2)

        # Aggregation
        aggr1 = torch.mean(masked1, dim=-2)
        aggr2 = torch.mean(masked2, dim=-2)
        # aggr1 = torch.mean(embedding1, dim=-2)
        # aggr2 = torch.mean(embedding2, dim=-2)
        # aggr1 = torch.max(masked1, dim=1).values
        # aggr2 = torch.max(masked2, dim=1).values
        # aggr1s = torch.sum(masked1, dim=-2)
        # aggr2s = torch.sum(masked2, dim=-2)


        # MLP
        x = torch.cat([aggr1, aggr2], dim=-1)
        # x = torch.cat([aggr1, aggr2, aggr1m, aggr2m, aggr1s, aggr2s], dim=-1)

        x = self.fc1(x)
        x = F.relu(x)

        if training: x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)

        if training: x = self.dropout(x)
        logits = self.fc_out(x)
        output = F.softmax(logits, dim=-1)
        return logits, output
    
    def update(self, sentences1, sentences2, lengths, targets, masks1, masks2, labels):
        logits, output = self(sentences1, sentences2, lengths, targets, masks1, masks2, training=True)

        self.optimizer.zero_grad()
        loss = self.loss(logits, labels)
        output = output.argmax(-1)
        # output = output.ge(0.5).float()

        accuracy = torch.mean((output==labels).float())

        # l1_reg = torch.sum(torch.tensor([torch.norm(param, 1)**2 for param in self.parameters()]))
        # l2_reg = torch.sum(torch.tensor([torch.norm(param, 2)**2 for param in self.parameters()]))
        # loss = loss + l1_reg + l2_reg

        loss.backward()
        self.optimizer.step()

        return loss.item(), accuracy.item()



class ModelA2M1(wicMeta):

    def __init__(self,
                 hidden_dim,
                 output_dim,
                 lr = 1e-3,
                 bias=False):
        super(ModelA2M1, self).__init__()

        dropout_val = 0.0
        embed_dim = self.embedding_layer.embedding_dim
        
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=0.0, bidirectional=True, bias=bias)

        self.dropout = nn.Dropout(0.95)
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(4 * hidden_dim, hidden_dim//2, bias=bias)
        self.fc2 = nn.Linear(hidden_dim//2, hidden_dim//4, bias=bias)
        self.fc_out = nn.Linear(hidden_dim//4, output_dim, bias=bias)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, sentences1, sentences2, lengths, targets, masks1, masks2, training=False):
        # sentences1 = sentences[:, 0, :]
        # sentences2 = sentences[:, 1, :]
        device = sentences1.device

        target1 = targets[:, 0]
        target2 = targets[:, 1]
        # Word embedding
        embedding1 = self.embedding_layer(sentences1)
        embedding2 = self.embedding_layer(sentences2)
        lengths1 = lengths[:, 0].cpu()
        lengths2 = lengths[:, 1].cpu()

        embedding1 = pack_padded_sequence(embedding1, lengths1, batch_first=True, enforce_sorted=False)
        embedding2 = pack_padded_sequence(embedding2, lengths2, batch_first=True, enforce_sorted=False)
        
        h1 = torch.randn(2, target1.shape[0], self.hidden_dim).to(device)
        h2 = torch.randn(2, target1.shape[0], self.hidden_dim).to(device)
        c1 = torch.randn(2, target1.shape[0], self.hidden_dim).to(device)
        c2 = torch.randn(2, target1.shape[0], self.hidden_dim).to(device)

        x1, (self.hidden1, _) = self.bilstm(embedding1, (h1, c1))
        x2, (self.hidden2, _) = self.bilstm(embedding2, (h2, c2))
        # x1 = x1.view(x1.shape[0], x1.shape[1], 2, self.hidden_dim)
        # x2 = x2.view(x2.shape[0], x2.shape[1], 2, self.hidden_dim)
        f1, b1 = self.hidden1
        f2, b2 = self.hidden2
        x1, _ = pad_packed_sequence(x1, batch_first=True)
        x2, _ = pad_packed_sequence(x2, batch_first=True)

        target_embed1 = x1[range(x1.shape[0]), target1]
        target_embed2 = x2[range(x1.shape[0]), target2]
        # hidden1 = torch.cat([f1, b1], dim=-1)
        # hidden2 = torch.cat([f2, b2], dim=-1)

        # x = torch.cat([f1, f2], dim=-1)
        # x = torch.cat([x1, x2], dim=-1)
        # x = torch.cat([hidden1, hidden2], dim=-1)
        x = torch.cat([target_embed1, target_embed2], dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        if training: x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        if training: x = self.dropout(x)
        logits = self.fc_out(x)
        output = F.softmax(logits, dim=-1)

        return logits, output

    def update(self, sentences1, sentences2, lengths, targets, masks1, masks2, labels):
        logits, output = self(sentences1, sentences2, lengths, targets, masks1, masks2, training=True)
        self.optimizer.zero_grad()
        loss = self.loss(logits, labels)
        output = output.argmax(-1)

        accuracy = torch.mean((output==labels).float())
        loss.backward()
        self.optimizer.step()

        return loss.item(), accuracy.item()