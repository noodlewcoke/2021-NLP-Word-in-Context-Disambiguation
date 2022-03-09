import os, sys
import json
import random
import numpy as np
import string
import re


class Vocabulary:

    def __init__(self, load=True):

        self.index2word = {0 : '<PAD>', 1 : '<UNK>'}
        self.word2index = {'<PAD>' : 0, '<UNK>' : 1}
        self.words_list = []
        self.total_number = 0
        self.stop_words = []
        self.mode = 'normal'
        if load:
            self.load()

    def make(self, document_path):
        with open(document_path, 'r') as reader:
            document = json.load(reader)
            reader.close()

        with open('data/stop_words.json') as reader:
            self.stop_words = json.load(reader)['stop_words']
            print('Number of stop words : ', len(self.stop_words))
            reader.close()

        for i in document.values():
            # Add sentence1 words
            self.words_list.extend(i['sentence1'])
            # Add sentence2 words
            self.words_list.extend(i['sentence2'])

        self.words_list = list(set(self.words_list))
        
        self.words_list = [word for word in self.words_list if word.isalpha()]
        self.words_list.sort()
        # self.words_list.insert(0, "<NUM>")
        self.words_list.insert(0, "<UNK>")
        self.words_list.insert(0, "<PAD>")
        self.index2word = {k:v for k, v in enumerate(self.words_list)}
        self.word2index = {v:k for k, v in enumerate(self.words_list)}
        self.total_number = len(self.words_list)
        print(f"Total number of words in vocabulary : {self.total_number}")

        vocab = {'word2index': self.word2index,
                 'index2word': self.index2word,
                 'words_list': self.words_list,
                 'stop_words': self.stop_words,
                 'total_number' : self.total_number,
                 'mode' : self.mode
                }
        
        with open('data/vocabulary.json', 'w') as writer:
            json.dump(vocab, writer)
            writer.close()
    ### TODO: Reomve all the numbers from the vocabulary. Either make 'em <UNK> or <NUM>
    def load(self):
        with open('data/vocabulary.json', 'r') as reader:
            vocab = json.load(reader)
            reader.close()
        self.word2index = vocab['word2index']
        self.index2word = vocab['index2word']
        self.words_list = vocab['words_list']
        self.stop_words = vocab['stop_words']
        self.total_number = vocab['total_number']
        self.mode = vocab['mode']
        print("Vocabulary loaded. Mode : ", self.mode)
        print(f"Total number of words in vocabulary : {self.total_number}")
        del vocab
    
    def convert_to_glove(self):
        with open('glove/matches.json', 'r') as reader:
            glove_matches = json.load(reader)
            reader.close()
        word2index = {'<PAD>' : 0, '<UNK>' : 1}
        index = 2
        keys = sorted([k for k in glove_matches.keys()])
        for k in keys:
            word2index[k] = index
            index += 1
        self.word2index = word2index
        self.index2word = {v:k for k,v in self.word2index.items()}
        self.words_list = sorted([k for k in self.word2index.keys()])
        self.total_number = len(self.words_list)
        self.mode = 'glove'
        vocab = {'word2index': self.word2index,
                 'index2word': self.index2word,
                 'words_list': self.words_list,
                 'stop_words': self.stop_words,
                 'total_number' : self.total_number,
                 'mode' : self.mode
                }
        
        with open('data/vocabulary.json', 'w') as writer:
            json.dump(vocab, writer)
            writer.close()

        print('Vocab reconstructed to GloVe. Total number : ', self.total_number)
if __name__ == "__main__":
    vocab = Vocabulary(False)
    vocab.make('preprocessed_train.json')
    # vocab.convert_to_glove()
    vocab = Vocabulary()