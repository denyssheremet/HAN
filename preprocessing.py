import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from my_utils import save_to_pickle, load_from_pickle
import string
from tqdm import tqdm
from collections import OrderedDict
import torchtext
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader



class MyTokenizer():
    def __init__(self):
        pass
    
    def __call__(self, text):
        # lowercase, remove punctuation, and split by space
        out = text.lower().replace("\n","").replace('\t','').translate(str.maketrans('', '', string.punctuation)).split(" ")
        res = []
        for i, word in enumerate(out):
            if (any(char.isnumeric() for char in word)):
                for letter in word:
                    res.append(letter)
            else:
                res.append(word)
        return res
    
    
def find_all_tokens(dataset, tokenizer):
    try: all_tokens = load_from_pickle("pickle/all_tokens.pickle")
    except:
        all_tokens = set()
        for d in [dataset['headline'], dataset['short_description']]:
            for sentence in tqdm(d):
                for token in tokenizer(sentence):
                    all_tokens.add(token)
        all_tokens = list(all_tokens)
        all_tokens.sort()
        save_to_pickle(all_tokens, "pickle/all_tokens.pickle")
    return all_tokens

def calc_token_counts(dataset, tokenizer):
    try: token_counts = load_from_pickle("pickle/token_counts.pickle")
    except:
        token_counts = OrderedDict()
        for d in [dataset['headline'], dataset['short_description']]:
            for sentence in tqdm(d):
                for token in tokenizer(sentence):
                    if token in token_counts.keys():
                        token_counts[token] += 1
                    else:
                        token_counts[token] = 1
        save_to_pickle(token_counts, "pickle/token_counts.pickle")
    return token_counts

def text_to_indices(text, tokenizer, vocab):
    return torch.tensor(vocab(tokenizer(text)))

def item_to_indices(item, tokenizer, vocab, category_map):
    return {
        'headline': text_to_indices(item['headline'], tokenizer, vocab), 
        'description': text_to_indices(item['short_description'], tokenizer, vocab),
        'category': category_map[item['category']],
    }

def map_item_to_indices(tokenizer, vocab, category_map):
    return lambda item: item_to_indices(item, tokenizer, vocab, category_map)

def split_dataset(dataset):
    # splits 60,20,20
    ds1 = dataset.train_test_split(0.2, seed=42)
    ds2 = ds1['train'].train_test_split(0.25, seed=42)
    ds_train = ds2['train']
    ds_val = ds2['test']
    ds_test = ds1['test']
    return ds_train, ds_val, ds_test


def preprocess_datasets(ds_train, ds_val, ds_test, tokenizer, vocab, category_map):
    try:
        ds_train_tok = load_from_pickle("pickle/ds_train_tok.pickle")
        ds_val_tok = load_from_pickle("pickle/ds_train_tok.pickle")
        ds_test_tok = load_from_pickle("pickle/ds_train_tok.pickle")
    except:
        f_map = map_item_to_indices(tokenizer, vocab, category_map)
        # cols = ['headline','description','category']
        rem_cols = ['link', 'short_description', 'authors', 'date']
        ds_train_tok = ds_train.map(f_map).remove_columns(rem_cols)
        ds_val_tok = ds_val.map(f_map).remove_columns(rem_cols)
        ds_test_tok = ds_test.map(f_map).remove_columns(rem_cols)
        
        save_to_pickle(ds_train_tok, "pickle/ds_train_tok.pickle") 
        save_to_pickle(ds_val_tok, "pickle/ds_val_tok.pickle") 
        save_to_pickle(ds_test_tok, "pickle/ds_test_tok.pickle")
    return ds_train_tok, ds_val_tok, ds_test_tok
        
def create_vocab(token_counts, min_freq):
    unk_token = '<unk>'
    vocab = torchtext.vocab.vocab(
        token_counts,
        min_freq=min_freq,
        specials=[unk_token], 
        special_first=True
    )
    vocab.set_default_index(vocab[unk_token])
    return vocab