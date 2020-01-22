import sys
sys.path.append('..')

import argparse
from time import time
import numpy as np
from data.preprocessing import preprocessing
from utils.modules import flat_accuracy, pad_sequence
from utils.dataloader import IndicDataset
from translation import TranslationModel
from transformers import BertConfig
import torch, torch.nn as nn
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from bpemb import BPEmb

seed_val = 42
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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


preprocessing()

writer = SummaryWriter(args.output + 'logs/') 

def train(epoch, model, train_loader, optimizer, scheduler):
    
    model.train()

    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
    start_time = time()

    total_loss = 0

    for batch_no, batch in enumerate(train_loader):

        source = batch[0].to(device)
        target = batch[1].to(device)

        model.zero_grad()        

        loss, logits = model(source, target)
        total_loss += loss.item()
        
        logits = logits.detach().cpu().numpy()
        label_ids = target.to('cpu').numpy()

        loss.backward()

        optimizer.step()
        scheduler.step()

    #Logging the loss and accuracy (below) in Tensorboard
    avg_train_loss = total_loss / len(train_loader)            
    #writer.add_scalar('Train/Loss', avg_train_loss, epoch)
    #writer.flush()

    print("Average training loss: {0:.2f}".format(avg_train_loss))

    return avg_train_loss

def validation(epoch, model, eval_loader):
    
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
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        eval_loss += loss

        nb_eval_steps += 1

    avg_valid_acc = eval_accuracy/nb_eval_steps
    avg_valid_loss = eval_loss/nb_eval_steps

    #writer.add_scalar('Valid/Loss', avg_valid_loss, epoch)
    #writer.add_scalar('Valid/Accuracy', avg_valid_acc, epoch)
    #writer.flush()

    print("Accuracy: {0:.2f}".format(avg_valid_acc))
    print("Average validation loss: {0:.2f}".format(avg_valid_loss))
    print("Time taken by epoch: {0:.2f}".format(time() - start_time))

    return avg_valid_loss, avg_valid_acc

if __name__ == "__main__":
    
    args = parser.parse_args()
    
    train_loader = DataLoader(IndicDataset(args.data, True), 
                              batch_size=args.batch_size, 
                              shuffle=False, 
                              collate_fn=pad_sequence)
    eval_loader = DataLoader(IndicDataset(args.data, False), 
                             batch_size=args.eval_size, 
                             shuffle=False, 
                             collate_fn=pad_sequence)

    #Create different tokenizers for both source and target language.
    tgt_tokenizer = BPEmb(lang='en', vs=args.vocab_size, dim=args.embed_dim, add_pad_emb=True)
    src_tokenizer = BPEmb(lang=args.lang, vs=args.vocab_size, dim=args.embed_dim, add_pad_emb=True)
    
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

    training_loss_values = []
    validation_loss_values = []
    validation_accuracy_values = []

    optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader), eta_min=lr)

    for epoch in args.epochs:
    
        training_loss = train(epoch)
        validation_loss, validation_acc = validation(epoch)

        training_loss_values.append(training_loss)
        validation_loss_values.append(validation_loss)
        validation_accuracy_values.append(validation_acc)
