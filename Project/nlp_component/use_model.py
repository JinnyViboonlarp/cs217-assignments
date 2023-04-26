# coding: utf-8

import torch
import sys

from model import LSTMTextClassifier
from torch.utils.data import DataLoader
import dill as pickle

#========== Do the necessary loading and setting things up ================

max_len = 16
label_map = {"O": 0, "NAME": 1, "STATE": 2, "UNIT": 3, "QUANTITY": 4,
                 "SIZE": 5, "TEMP": 6, "DF": 7, "PADDING": 8}
label_map_inv = {v: k for k, v in label_map.items()}

model = torch.load('model.pt')

with open('vocab.p', 'rb') as file:
    vocab = pickle.load(file)
    
with open('text_pipeline.p', 'rb') as file:
    text_pipeline = pickle.load(file)

#==========================================================================

class Entity:
    def __init__(self, start_char, end_char, text, label):
        self.start_char = start_char
        self.end_char = end_char
        self.text = text
        self.label = label

def tokenize(text):
    text = text.replace(',',' , ').replace('(',' ( ').replace(')',' ) ')
    return text.strip().split()[:max_len]
    #return text.replace(',',' , ').strip().split()[:max_len]

def get_loader(tokenized_text, batch_size=32, max_len=16, shuffle=False):

    def collate_batch(batch):

        pad_token_id = -1
        label_list, text_list, mask_list = [], [], []
        pre_augmented_text = text_pipeline(tokenized_text)
        for i,w in enumerate(pre_augmented_text):
            text = [w] + pre_augmented_text
            # If too short, pad to max_len.
            text = text + ([pad_token_id] * max_len)
            # If sentence too long, truncate to max_len.
            text = text[:max_len]
            # Create a mask tensor indicating where padding is present.
            pad_mask = [0 if i==pad_token_id else 1 for i in text]

            text_list.append(text)
            mask_list.append(pad_mask)

        text_list = torch.tensor(text_list)
        mask_list = torch.tensor(mask_list)
        return text_list, mask_list
    
    return DataLoader(tokenized_text, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_batch)

def predict(model, dataloader, label_map_inv):
    model.eval() # do not use dropout
    with torch.no_grad():
        # Iterate over the data in the data loader
        pred = []
        for i, (ids, mask) in enumerate(dataloader):
            pass
            # Perform forward pass to get outputs (like in train)
            output = model(ids, mask)
            # Get the predictions by taking the argmax
            int_pred = torch.argmax(output, dim=1).tolist()
            pred = pred + [label_map_inv[w] for w in int_pred]
            
    return pred
    # e.g. ['QUANTITY', 'SIZE', 'NAME', 'NAME', 'NAME', 'O', 'O', 'STATE']

def create_entity_list(tokenized_text, pred):

    def valid(word):
        if(word in [',','(',')']):
            return False
        return True

    new_text = " ".join(tokenized_text)
    entities = []
    start_char = 0
    i = 0
    while(i < len(pred)):
        word = tokenized_text[i]
        tag = pred[i]
        end_char = (start_char + len(word))
        if(tag != 'O' and tag!= 'PADDING' and valid(word)):
            while((i+1)<len(pred) and pred[i+1] ==  tag
                  and valid(tokenized_text[i+1])):
                # i.e. while the next tag is the same as the current tag
                word = (word + " " + tokenized_text[i+1])
                end_char = (end_char + 1 + len(tokenized_text[i+1]))
                i += 1
            entity = Entity(start_char, end_char, word, tag)
            #print(start_char, end_char, word, tag)
            entities.append(entity)
        start_char = (end_char + 1)
        i += 1
    return new_text, entities 

def ner(text):

    tokenized_text = tokenize(text)
    dataloader = get_loader(tokenized_text, batch_size=32, max_len=max_len, shuffle=False)
    pred = predict(model, dataloader, label_map_inv)
    new_text, entities = create_entity_list(tokenized_text, pred)
    return new_text, entities
    
if __name__ == "__main__":

    new_text, entities = ner("1/2 large sweet red onion, thinly sliced")
    print(new_text)
    
    
