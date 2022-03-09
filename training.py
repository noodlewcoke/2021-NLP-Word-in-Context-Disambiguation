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
from dataloader import DataLoader
from vocabulary import Vocabulary
from models import ModelA1M1, ModelA2M1
CUDA_LAUNCH_BLOCKING=1
EPOCHS = 200
EXPERIMENT = 'exp_a2_we'
SAVEPATH = 'experiments/' + EXPERIMENT + '/'


def a1m1(vocab_dim):
    trainloader = DataLoader('data/tokenized_train.json', 200, shuffle=True, random_unk=False)
    devloader = DataLoader('data/tokenized_dev.json', 200)
    print("INFO: Dataloader ready.")
    model = ModelA2M1(
                      hidden_dim=400,
                      output_dim=2,
                      lr=4e-4,
                      bias=True)
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print(f"INFO: Device used is {device}")
    model.to(device)
    losses, epoch_losses = [], []
    accuracies, epoch_accuracies = [], []
    dev_losses, dev_epoch_losses = [], []
    dev_accuracies, dev_epoch_accuracies = [], []
    for epoch in range(1, EPOCHS):
        epoch_loss = 0
        epoch_acc = 0

        if epoch == 120:
            for opt in model.optimizer.param_groups:
                opt['weight_decay'] = 0.001
                # opt['lr'] = 3e-4
        print(f'\nTRAINING EPOCH {epoch}')
        for episode, (sentences1, sentences2, lengths, targets, masks1, masks2, labels) in enumerate(trainloader):
            # labels = torch.unsqueeze(labels, dim=1)
            loss, accuracy = model.update(sentences1.to(device).long(),
                                          sentences2.to(device).long(), 
                                          lengths, 
                                          targets.to('cpu').long(), 
                                          masks1.to(device).float(), 
                                          masks2.to(device).float(), 
                                          labels.to(device).long())
            # loss, accuracy = model.update(sentences.to(device).long(), lengths, targets.to('cpu').long(), labels.to(device).long())
            epoch_loss += loss
            epoch_acc += accuracy
            losses.append(loss)
            accuracies.append(accuracy)
            
        epoch_loss /= trainloader.n_batch
        epoch_acc /= trainloader.n_batch
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_acc)
        print(f'Epoch : {epoch}\tLoss : {epoch_loss}\tAccuracy : {epoch_acc}')

        print(f'\nVALIDATION EPOCH {epoch}')
        dev_epoch_loss = 0
        dev_epoch_acc = 0
        for episode, (sentences1, sentences2, lengths, targets, masks1, masks2, labels) in enumerate(devloader):
            # labels = torch.unsqueeze(labels, dim=1)
            loss, accuracy = model.eval(sentences1.to(device).long(), 
                                        sentences2.to(device).long(), 
                                        lengths, 
                                        targets.to('cpu').long(), 
                                        masks1.to(device).float(), 
                                        masks2.to(device).float(), 
                                        labels.to(device).long())
            dev_epoch_loss += loss
            dev_epoch_acc += accuracy
            dev_losses.append(loss/(episode + 1))
            dev_accuracies.append(accuracy/(episode + 1))
            
        dev_epoch_loss /= devloader.n_batch
        dev_epoch_acc /= devloader.n_batch
        dev_epoch_losses.append(dev_epoch_loss)
        dev_epoch_accuracies.append(dev_epoch_acc)
        print(f'Epoch : {epoch}\tLoss : {dev_epoch_loss}\tAccuracy : {dev_epoch_acc}')
        
        model.save(SAVEPATH+f'models/model{epoch}')
        np.save(f'{SAVEPATH}epoch_losses', epoch_losses)
        np.save(f'{SAVEPATH}dev_epoch_losses', dev_epoch_losses)
        np.save(f'{SAVEPATH}epoch_accuracies', epoch_accuracies)
        np.save(f'{SAVEPATH}dev_epoch_accuracies', dev_epoch_accuracies)

    print("\nTraining finished.")
    print(f"Best model : {np.argmax(dev_epoch_accuracies)+1}")
    print(f"Best Accuracy : {np.max(dev_epoch_accuracies)}")
if __name__ == '__main__':
    vocab = Vocabulary()
    vocab_dim = vocab.total_number
    del vocab

    a1m1(vocab_dim)