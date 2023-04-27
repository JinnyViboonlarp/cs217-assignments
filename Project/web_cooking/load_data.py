# coding: utf-8

import io
import orjson
import torch
import dill as pickle
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from utils import get_device

import sys


def get_data(filename):
    
    data = []
    ingredients = []
    tags = []
    with open(filename) as file:
        for line in file:
            token = line.strip().split('\t')
            if(len(token)<2): #line break
                if(len(ingredients)>0):
                    data.append((ingredients, tags))
                ingredients = []
                tags = []
            else:
                ingredients.append(token[0])
                tags.append(token[1])

    return data


def get_vocab_and_pipeline(train_data):
    """Takes training data (of the form (label, text) pairs) and builds
    a torchtext vocab object with the vocabulary found in the training data.
    Also creates a text pipeline callable that can be used to tokenize text
    and map to one-hot/integer encoding of the text
    """
    # Get a tokenizer from torchtext
    #tokenizer = get_tokenizer("basic_english")
    
    # The items in our data iterable are (label, text) pairs.  To build a
    # vocabulary we just want the tokens in the text so we throw away the first
    # item while iterating.  We tokenize each text and yield those tokens.
    def yield_tokens(data_iter):
        for text, _ in data_iter:
            yield text
        """
        for _, text in data_iter:
            yield tokenizer(text)
        """

    # Build the actual vocabulary from the tokens
    vocab = build_vocab_from_iterator(
        yield_tokens(iter(train_data)), specials=["<pad>", "<unk>"]
    )

    # The default integer index should be <unk> (out-of-vocab ID)
    vocab.set_default_index(vocab["<unk>"])

    # The "text pipeline" performs this transform:
    # str (entire sentence)
    #   -> list[str] (tokens)
    #   -> list[int] (token IDs)
    #text_pipeline = lambda x: vocab(tokenizer(x))
    text_pipeline = lambda x: vocab(x)

    return vocab, text_pipeline


def get_loader(data, text_pipeline, label_pipeline, batch_size, max_len, shuffle=False):
    """Takes a dataset and pipelines returning a PyTorch dataloader ready for use in training."""

    # Use helper function to infer what device to use (use GPU if available)
    device = get_device()

    def collate_batch(batch):
        """Collate function should take a batch as a list of (label, text) pairs and apply
        the provided text and label pipelines to create label and input tensors. Input tensors
        should have a fixed length specified by max_len. Inputs shorter than max_len should be
        padded, while inputs longer than max_len shoudl be truncated. Additionally,
        create a mask tensor that denotes which portions of the input correspond to padding and
        should be ignored.
        """

        # A "collate_fn" takes in a batch of raw data (`batch`) and processes it into one or more Tensors.

        pad_token_id = -1
        label_list, text_list, mask_list = [], [], []
        # 1.) Iterate over all (label, text) tuples
        for (_text, _label) in batch:
            # 2.) Process label and text using pipelines
            label = label_pipeline(_label)
            #label = label[0] #added in to test the lstm
            orig_text = text_pipeline(_text)
            for i,w in enumerate(orig_text):
                text = [w] + orig_text
                # 5.) If too short, pad to max_len.
                text = text + ([pad_token_id] * max_len)
                # 4.) If sentence too long, truncate to max_len.
                text = text[:max_len]
                # 3.) Convert text tokens to Tensor.
                #text = torch.tensor(text)
                # 6.) Create a mask tensor indicating where padding is present.
                #pad_mask = (~(text == pad_token_id)) # boolean tensor where 1=genuine token and 0=padding
                pad_mask = [0 if i==pad_token_id else 1 for i in text]
                #pad_mask = [1 if i else 0 for i in pad_mask]
                # 7.) Convert the label list to tensor
                #label = torch.tensor(label)
        
                label_list.append(label[i])
                text_list.append(text)
                mask_list.append(pad_mask)

        label_list = torch.tensor(label_list)
        text_list = torch.tensor(text_list)
        mask_list = torch.tensor(mask_list)

        return label_list.to(device), text_list.to(device), mask_list.to(device)

    # At the end, we create a DataLoader object and return it
    return DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch
    )


def get_data_loaders(train_file, val_file, test_file, batch_size, max_len, label_map):

    # Take the file paths {train,val,test}_file and load the data contained in the files.
    train_data = get_data(train_file)
    val_data = get_data(val_file)
    test_data = get_data(test_file)
    # e.g. train_data[4] = ('Objectives', 'The geometric structure is Cl-M-Ph 3')
    # e.g. train_data[0] = (['4', 'cloves', 'garlic'], ['QUANTITY', 'UNIT', 'NAME'])

    # Callable for data loader that gets a label from `label_map` for each data point x
    label_pipeline = lambda x: [label_map.get(w, 0) for w in x]
    #label_pipeline = lambda x: label_map.get(x, 0)
    #print(label_pipeline(['QUANTITY', 'UNIT', 'NAME'])) # [4, 3, 1]
    #print(len(train_data))

    
    # Construct a vocabulary from the training data and create a callable
    # (text_pipeline) that tokenizes and then uses the vocab to get token IDs.
    vocab, text_pipeline = get_vocab_and_pipeline(train_data)
    #print(vocab['pecans']) # 2
    #print(len(vocab)) #779
    #print(text_pipeline(['1/2','teaspoon','citric','acid','-LRB-'])) #[8, 9, 1, 1, 6]
    #print(text_pipeline('The geometric structure is the Cl-M-Ph 3 is')) #[2, 635, 749, 12, 2, 1, 14, 12]
    #print(text_pipeline(['4', 'cloves', 'garlic'])) #[18, 63, 28]
    with open('vocab.p', "wb") as file:
        pickle.dump(vocab, file)
    with open('text_pipeline.p', "wb") as file:
        pickle.dump(text_pipeline, file)

    # Get {train,val,test} data loaders given:
    # * a Vocab object
    # * a "label pipeline" that can transform a list of labels to IDs
    # * a "text pipeline" that tokenizes text and converts the tokens to IDs
    train_dataloader = get_loader(
        train_data, text_pipeline, label_pipeline, batch_size, max_len, shuffle=True
    )
    val_dataloader = get_loader(
        val_data, text_pipeline, label_pipeline, batch_size, max_len, shuffle=False
    )
    test_dataloader = get_loader(
        test_data, text_pipeline, label_pipeline, batch_size, max_len, shuffle=False
    )
    
    return vocab, train_dataloader, val_dataloader, test_dataloader
