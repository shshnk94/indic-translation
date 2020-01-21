from torch.nn.utils.rnn import pad_sequence

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(np.equal(pred_flat, labels_flat)) / len(labels_flat)

def pad_sequence(batch, src_padding_value, tgt_padding_value):

  x = [s[0] for s in batch]
  x = pad_sequence(x, batch_first=True, padding_value=src_padding_value)

  y = [s[1] for s in batch]
  y = pad_sequence(y, batch_first=True, padding_value=tgt_padding_value)

  return x, y
