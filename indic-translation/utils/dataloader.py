from torch.utils.data import Dataset, DataLoader
import pandas as pd

class IndicDataset(Dataset):
  
  def __init__(self, filepath='data/', is_train=True):
    filepath += 'train' if is_train else 'valid'
    self.df = pd.read_csv(filepath, header=None)

  def __len__(self):
    return self.df.shape[0]

  def __getitem__(self, index):
    y, x = self.df.loc[index]

    #tokenize into integer indices
    x = src_tokenizer.encode_ids(x)
    y = tgt_tokenizer.encode_ids(y)

    #add special tokens to target
    y = [tgt_tokenizer.BOS] + y + [tgt_tokenizer.EOS]

    return torch.LongTensor(x), torch.LongTensor(y)
