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
from vocabulary import Vocabulary

def unknown_token():
    with open('glove/glove.6B.300d.txt', 'r', encoding='utf8') as reader:
        raw_glove = reader.read().split('\n')[:-1]
        reader.close()
    print(len(raw_glove))
    unk_token = torch.zeros((len(raw_glove), 300))
    for e, raw in enumerate(raw_glove):
        tmp = raw.split()
        try:
            t = torch.tensor([float(i) for i in tmp[1:]])
            unk_token[e] = t
        except Exception as ex:
            print(e)
            print(t)
            raise ex
    
    unk_token = torch.mean(unk_token, 0)
    print(unk_token)
    print(unk_token.shape)
    torch.save(unk_token, 'glove/unk_token.pt')


def matches():

    with open('glove/glove.6B.300d.txt', 'r', encoding='utf8') as reader:
        raw_glove = reader.read().splitlines()
        reader.close()
    vocabulary = Vocabulary()
    words_list = vocabulary.words_list

    glove_keys = []
    glove_embeddings = {}
    for raw in raw_glove:
        tmp = raw.split()
        glove_keys.append(tmp[0])
        if tmp[0] in words_list:
            glove_embeddings[tmp[0]] = tmp[1:]
        assert len(tmp[1:]) == 300

    print("Number of words in my vocabulary : ", len(words_list))
    # print("Number of words in GloVe : ", len(glove_keys.key()))
    with open('glove/matches.json', 'w') as writer:
        json.dump(glove_embeddings, writer)
        writer.close()
    print("Number of words coinciding : ", len(glove_embeddings.keys()))

def embed_matrix():   
    vocab = Vocabulary() # Must be GloVe vocab
    with open('glove/matches.json', 'r') as reader:
        glove_embeddings = json.load(reader)
        reader.close()
    unk_token = torch.load('glove/unk_token.pt')
    embedding_matrix = torch.zeros((vocab.total_number, 300))
    print(embedding_matrix.shape)
    print(unk_token.shape)
    embedding_matrix[1] = unk_token
    tmp = []
    for k,v in glove_embeddings.items():
        assert k != "<PAD>" or k != "<UNK>"
        embedding_matrix[vocab.word2index[k]] = torch.tensor([float(i) for i in v])
        tmp.append(vocab.word2index[k])
    tmp = list(set(tmp))
    print(len(tmp))

    torch.save(embedding_matrix, 'glove/embedding_matrix.pt')


# matches()
# unknown_token()
# embed_matrix()