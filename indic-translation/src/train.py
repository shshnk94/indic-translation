import sys
sys.path.append('..')

import argparse
from time import time

from data.preprocessing import Preprocess
from utils.modules import flat_accuracy, pad_sequence
from utils.dataloader import IndicDataset
from translation import TranslationModel
from transformers import BertConfig
import torch.nn as nn

from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter

from bpemb import BPEmb

parser = argparse.ArgumentParser(description='Indic Translation Training')

parser.add_argument('--data', help='path to the directory containing dataset')
parser.add_argument('--output', help='path to the output directory')
parser.add_argument('--lang', help='language to be translated into English')
parser.add_argument('--epochs', type=int, help='number of training epochs')
parser.add_argument('--lr', type=float, help='learning rate of the training step')
parser.add_argument('--batch_size', type=int, help='batch size of training data')
parser.add_argument('--eval_size', type=int, help='batch size of validation data')
parser.add_argument('--vocab_size', type=int, help='vocabulary size of the source and target tokenizer')
parser.add_argument('--embed_dim', type=int, help='embedding dimension for source and target')
parser.add_argument('--hidden_size', type=int, help='hidden layer size of the self/cross attention layer')
parser.add_argument('--intermediate_size', type=int, help='intermediate layer size of the fully-connected layer')
parser.add_argument('--num_attention_heads', type=int, help='number of attention heads per self/cross attention layer')
parser.add_argument('--num_hidden_layers', type=int, help='number of attention hidden layers')
parser.add_argument('--hidden_act', default='gelu', help='hidden layer activation function')
parser.add_argument('--dropout_prob', type=float, help='dropout probability')

args = parser.parse_args()

#Create different tokenizers for both source and target language.
tgt_tokenizer = BPEmb(lang='en', vs=args.vocab_size, dim=args.embed_dim, add_pad_emb=True)
src_tokenizer = BPEmb(lang=args.lang, vs=args.vocab_size, dim=args.embed_dim, add_pad_emb=True)

#Hyperparameters
seed_val = 42
epochs = args.epochs
lr = args.lr

#hidden_size and intermediate_size are both wrt all the attention heads. 
#Should be divisible by num_attention_heads
hidden_size = args.hidden_size
intermediate_size = args.intermediate_size
num_attention_heads = args.num_attention_heads
num_hidden_layers = args.num_hidden_layers

encoder_config = BertConfig(vocab_size=args.vocab_size+1, #To handle UNK
                            hidden_size=args.hidden_size,
                            num_hidden_layers=args.num_hidden_layers,
                            num_attention_heads=num_attention_heads,
                            intermediate_size=args.intermediate_size,
                            hidden_act=args.hidden_act,
                            hidden_dropout_prob=args.dropout_prob,
                            attention_probs_dropout_prob=args.dropout_prob,
                            max_position_embeddings=512,
                            type_vocab_size=2,
                            initializer_range=0.02,
                            layer_norm_eps=1e-12)

decoder_config = BertConfig(vocab_size=args.vocab_size+1,
                            hidden_size=args.hidden_size,
                            num_hidden_layers=args.num_hidden_layers,
                            num_attention_heads=num_attention_heads,
                            intermediate_size=args.intermediate_size,
                            hidden_act=args.hidden_act,
                            hidden_dropout_prob=args.dropout_prob,
                            attention_probs_dropout_prob=args.dropout_prob,
                            max_position_embeddings=512,
                            type_vocab_size=2,
                            initializer_range=0.02,
                            layer_norm_eps=1e-12,
                            is_decoder=True)

#Create encoder and decoder embedding layers.
encoder_embeddings = nn.Embedding(args.vocab_size+1, args.hidden_size, padding_idx=src_tokenizer.vs)
decoder_embeddings = nn.Embedding(args.vocab_size+1, args.hidden_size, padding_idx=tgt_tokenizer.vs)
model = TranslationModel(encoder_config, decoder_config, encoder_embeddings, decoder_embeddings)

preprocessing()
train_dataset = IndicDataset(args.data, True)
valid_dataset = IndicDataset(args.data, False)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_sequence)
eval_loader = DataLoader(valid_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=pad_sequence)

writer = SummaryWriter(args.output + 'logs/') 

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader), eta_min=lr)

np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

training_loss_values = []
validation_loss_values = []
validation_accuracy_values = []

for epoch in range(epochs):

    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
    start_time = time()

    total_loss = 0
    model.train()

    for batch_no, batch in enumerate(train_loader):

        source = batch[0].to(device)
        target = batch[1].to(device)

        model.zero_grad()        

        loss, logits = model(source, target)
        total_loss += loss.item()
        
        logits = logits.detach().cpu().numpy()
        label_ids = target.to('cpu').numpy()

        # Debugging begins here.
        if batch_no == 1:
          print("----------Translated training data-----------")
          translated = np.argmax(logits, axis=2)
          special_token = [0, 1, 2, tgt_tokenizer.vs]
          for index in range(target.shape[0]):
            truth = [int(token) for token in target[index] if token not in special_token]
            print("Ground truth: ", tgt_tokenizer.decode_ids(truth))
            translation = [int(token) for token in translated[index] if token not in special_token]
            print("Translation: ", tgt_tokenizer.decode_ids(translation))
        # Debugging ends here.

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    #Logging the loss and accuracy (below) in Tensorboard
    #avg_train_loss = total_loss / len(train_loader)            
    #writer.add_scalar('Train/Loss', avg_train_loss, epoch)
    #writer.flush()

    print("Average training loss: {0:.2f}".format(avg_train_loss))
    print("Running Validation...")

    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps = 0

    for batch_no, batch in enumerate(eval_loader):
        
        source = batch[0].to(device)
        target = batch[1].to(device)
        
        with torch.no_grad():        
            loss, logits = model(source, target)

        logits = logits.detach().cpu().numpy()
        label_ids = target.to('cpu').numpy()
        
        # Debugging begins here.
        if batch_no == 1:
          print("----------Translated validation data-----------")
          translated = np.argmax(logits, axis=2)
          special_token = [0, 1, 2, tgt_tokenizer.vs]
          for index in range(target.shape[0]):
            truth = [int(token) for token in target[index] if token not in special_token]
            print("Ground truth: ", tgt_tokenizer.decode_ids(truth))
            translation = [int(token) for token in translated[index] if token not in special_token]
            print("Translation: ", tgt_tokenizer.decode_ids(translation))
        # Debugging ends here.

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        eval_loss += loss

        nb_eval_steps += 1

    avg_valid_acc = eval_accuracy/nb_eval_steps
    validation_accuracy_values.append(avg_valid_acc)
    avg_valid_loss = eval_loss/nb_eval_steps
    validation_loss_values.append(avg_valid_loss)

    #writer.add_scalar('Valid/Loss', avg_valid_loss, epoch)
    #writer.add_scalar('Valid/Accuracy', avg_valid_acc, epoch)
    #writer.flush()

    print("Accuracy: {0:.2f}".format(avg_valid_acc))
    print("Average validation loss: {0:.2f}".format(avg_valid_loss))
    print("Time taken by epoch: {0:.2f}".format(time() - start_time))

# summarize training loss
plt.plot(training_loss_values)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

# summarize validation loss
plt.plot(validation_loss_values)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

# summarize validation accuracy
plt.plot(validation_accuracy_values)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

