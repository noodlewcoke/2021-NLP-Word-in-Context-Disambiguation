import os, sys
import json
import random
import numpy as np
import string
import re
from pprint import pprint
'''
RAW DATA STRUCTURE
number of data: train -> 8000  dev -> 1000
id          --> "train.0"
lemma       --> "play"
pos         --> "NOUN"
sentence1   --> "In that context of coordination and integration, Bolivia holds a key play in any process of infrastructure development."
sentence2   --> "A musical play on the same subject was also staged in Kathmandu for three days."
start1      --> "69"
end1        --> "73"
start2      --> "10"
end2        --> "14"
label       --> "False"
'''

'''
PROCESSED DATA STRUCTURE

id          --> key in dict
lemma       --> "play"
pos         --> "NOUN"
sentence1   --> list of int OR list of str
sentence2   --> list of int OR list of str
target1     --> int OR str
target2     --> int OR str
index1      --> int
index2      --> int
label       --> bool
'''
def stop_word_list():
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    puncs = string.punctuation
    mapping = str.maketrans('', '', puncs)
    stop_wordsp = [word.encode('ascii', 'ignore').decode() for word in stop_words]
    stop_wordsp = [word.translate(mapping).lower() for word in stop_wordsp]
    stop_wordsp += stop_words
    stop_wordsp = list(set(stop_wordsp))
    with open('data/train.jsonl', 'r', encoding='utf-8') as reader:
        raw_data = reader.read().splitlines()
        reader.close()
    raw_data = [json.loads(data) for data in raw_data]
    target_list = []
    for raw in raw_data:
        target_list.append(raw['lemma'].lower())
    target_list = list(set(target_list))

    matches = [stop_word for stop_word in stop_wordsp if stop_word in target_list]
    # stop_wordsp = [stop_word for stop_word in stop_wordsp if not stop_word in target_list]

    with open('data/stop_words.json', 'w') as writer:
        tmp = {
            'stop_words' : stop_wordsp,
            'matches'    : matches
        }
        json.dump(tmp, writer)
        writer.close()
    return stop_wordsp, matches

targetlist = {'train':[], 'dev':[]}
def raw_to_list(raw_data, path, stop_words):
    global targetlist
    all_labels = {"true" : 0, "false" : 0}

    data = {}
    c = 0
    puncs = string.punctuation
    # puncs = puncs.replace("'", "")
    mapping = str.maketrans('', '', puncs)
    label_label = {"true" : 1, "false": 0}
    duplicates = {}
    sentence_lengths, target_indices = [], []
    for raw in raw_data:
        # strip ID

        try:
            id = raw["id"].split('.')
            id = int(id[1])

            # extract the target words
            start1 = int(raw['start1'])
            end1 = int(raw['end1'])
            start2 = int(raw['start2'])
            end2 = int(raw['end2'])
            # target1 = raw['sentence1'][start1:end1]
            # target2 = raw['sentence2'][start2:end2]
            lemma = raw["lemma"].lower()
            target1 = lemma
            target2 = lemma
            targetlist[dataset].append(target1)
            target_buffer = 'xteapotsxxxrevengex'
            assert target1.isalpha(), "Not a word {}".format(target1)
            assert target2.isalpha(), "Not a word {}".format(target2)
            sentence1 = raw['sentence1']
            sentence1 = sentence1[:start1] + ' ' + target_buffer + ' ' + sentence1[end1:]
            sentence2 = raw['sentence2']
            sentence2 = sentence2[:start2] + ' ' + target_buffer + ' ' + sentence2[end2:]

            # Split words and remove punctuations
            sentence1 = sentence1.split()
            sentence1 = [word.encode('ascii', 'ignore').decode() for word in sentence1]
            sentence1 = [word.translate(mapping).lower() for word in sentence1]

            sentence2 = sentence2.split()
            sentence2 = [word.encode('ascii', 'ignore').decode() for word in sentence2]
            sentence2 = [word.translate(mapping).lower() for word in sentence2]

            # Remove '' from sentences
            sentence1 = [word for word in sentence1 if word]
            sentence2 = [word for word in sentence2 if word]

            sentence1 = [word for word in sentence1 if word.isalpha()]
            sentence2 = [word for word in sentence2 if word.isalpha()]

            # Remove stop words from sentences
            sentence1 = [word for word in sentence1 if word not in stop_words]
            sentence2 = [word for word in sentence2 if word not in stop_words]

            sentence_lengths.extend([len(sentence1), len(sentence2)])

            # Process label
            label = raw["label"].lower()
            all_labels[label] += 1
            label = label_label[label]
            data[id] = {}
            data[id]["lemma"] = raw["lemma"]
            data[id]["pos"] = raw["pos"]
            data[id]["sentence1"] = sentence1
            data[id]["sentence2"] = sentence2
            data[id]["target1"] = target1
            data[id]["target2"] = target2

            ###### |  This doesnt work if there are multiple words in 
            ###### V  the sentence matching the target word
            dup1 = [i for i in sentence1 if i == target_buffer]
            if len(dup1) > 1:
                duplicates[str(id) + '1'] = dup1
            dup2 = [i for i in sentence2 if i == target_buffer]
            if len(dup2) > 1:
                duplicates[str(id) + '2'] = dup2
            target_index1 = sentence1.index(target_buffer)
            sentence1[target_index1] = target1
            data[id]["index1"] = target_index1
            target_index2 = sentence2.index(target_buffer)
            sentence2[target_index2] = target2
            target_indices.extend([target_index1, target_index2])
            data[id]["index2"] = target_index2
            data[id]["label"] = label
        except Exception as ex:
            print(id)
            print(sentence1, target1)
            print(sentence2, target2)
            print(ex)
    pprint(duplicates)

    print(f"sentence length max  : {np.max(sentence_lengths)}")      # 88
    print(f'sentence length min  : {np.min(sentence_lengths)}')      # 6
    print(f'sentence length mean : {np.mean(sentence_lengths)}')   # 22.0
    print(f'sentence length med  : {np.median(sentence_lengths)}')   # 22.0
    print(f'sentence length std  : {np.std(sentence_lengths)}')      # 9.05
    print(f'target indices max   : {np.max(target_indices)}')        # 64   
    print(f'Number of all labels', all_labels)
    with open(f'preprocessed_{path}.json', 'w') as f:
        json.dump(data, f)
        f.close()


stop_words, matches = stop_word_list()
data_path = "data/"
datasets = ['train', 'dev']
for dataset in datasets:
    with open(data_path + f"{dataset}.jsonl", 'r', encoding='utf-8') as f:
        raw_data =  f.read().splitlines()
        f.close()

    raw_data = [json.loads(data) for data in raw_data]

    print(raw_data[0].keys())
    print(len(raw_data))

    raw_to_list(raw_data, dataset, stop_words)

train_target = list(set(targetlist['train']))
dev_target = list(set(targetlist['dev']))

similar = [word for word in dev_target if word in train_target]
print('Train targets num : ', len(train_target))
print('Dev targets num : ', len(dev_target))
print('Number of similarities : ', len(similar))
print(similar)