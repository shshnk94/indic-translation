from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import pandas as pd

class PadSequence:
    
    def __init__(self, src_padding_value, tgt_padding_value):
        self.src_padding_value = src_padding_value
        self.tgt_padding_value = tgt_padding_value
    
    def __call__(self, batch):
        
        x = [s[0] for s in batch]
        x = pad_sequence(x, 
                         batch_first=True, 
                         padding_value=self.src_padding_value)

        y = [s[1] for s in batch]
        y = pad_sequence(y, 
                         batch_first=True, 
                         padding_value=self.tgt_padding_value)

        return x, y


class IndicDataset(Dataset):
  
    def __init__(self, 
                 src_tokenizer,
                 tgt_tokenizer,
                 filepath,
                 is_train=True):
        filepath += 'train' if is_train else 'valid'
        self.df = pd.read_csv(filepath, header=None)

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        y, x = self.df.loc[index]

        #tokenize into integer indices
        x = self.src_tokenizer.encode_ids(x)
        y = self.tgt_tokenizer.encode_ids(y)

        #add special tokens to target
        y = [self.tgt_tokenizer.BOS] + y + [self.tgt_tokenizer.EOS]

        return torch.LongTensor(x), torch.LongTensor(y)
