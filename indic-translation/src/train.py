import sys
sys.path.append('..')

import argparse
from time import time
import numpy as np
from config import Config
from utils.modules import flat_accuracy
from utils.dataloader import IndicDataset, PadSequence
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

def train(config, model, train_loader, eval_loader, writer=None):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader), eta_min=config.lr)
    
    training_loss_values = []
    validation_loss_values = []
    validation_accuracy_values = []

    for epoch in range(config.epochs):

        model.train()

        print('======== Epoch {:} / {:} ========'.format(epoch + 1, config.epochs))
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
        training_loss_values.append(avg_train_loss)
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
        
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            eval_loss += loss

            nb_eval_steps += 1

        avg_valid_acc = eval_accuracy/nb_eval_steps
        avg_valid_loss = eval_loss/nb_eval_steps
        validation_loss_values.append(avg_valid_loss)
        validation_accuracy_values.append(avg_valid_acc)
        #writer.add_scalar('Valid/Loss', avg_valid_loss, epoch)
        #writer.add_scalar('Valid/Accuracy', avg_valid_acc, epoch)
        #writer.flush()

        print("Accuracy: {0:.2f}".format(avg_valid_acc))
        print("Average validation loss: {0:.2f}".format(avg_valid_loss))
        print("Time taken by epoch: {0:.2f}".format(time() - start_time))

    return training_loss_values, validation_loss_values, validation_accuracy_values

def build_model(config):
    
    #Create different tokenizers for both source and target language.
    tgt_tokenizer = BPEmb(lang='en', vs=config.vocab_size, dim=config.embed_dim, add_pad_emb=True)
    src_tokenizer = BPEmb(lang=config.lang, vs=config.vocab_size, dim=config.embed_dim, add_pad_emb=True)
    
	#hidden_size and intermediate_size are both wrt all the attention heads. 
    #Should be divisible by num_attention_heads
    encoder_config = BertConfig(vocab_size=config.vocab_size+1, #To handle UNK
                                hidden_size=config.hidden_size,
                                num_hidden_layers=config.num_hidden_layers,
                                num_attention_heads=config.num_attention_heads,
                                intermediate_size=config.intermediate_size,
                                hidden_act=config.hidden_act,
                                hidden_dropout_prob=config.dropout_prob,
                                attention_probs_dropout_prob=config.dropout_prob,
                                max_position_embeddings=512,
                                type_vocab_size=2,
                                initializer_range=0.02,
                                layer_norm_eps=1e-12)

    decoder_config = BertConfig(vocab_size=config.vocab_size+1,
                                hidden_size=config.hidden_size,
                                num_hidden_layers=config.num_hidden_layers,
                                num_attention_heads=config.num_attention_heads,
                                intermediate_size=config.intermediate_size,
                                hidden_act=config.hidden_act,
                                hidden_dropout_prob=config.dropout_prob,
                                attention_probs_dropout_prob=config.dropout_prob,
                                max_position_embeddings=512,
                                type_vocab_size=2,
                                initializer_range=0.02,
                                layer_norm_eps=1e-12,
                                is_decoder=True)

    #Create encoder and decoder embedding layers.
    encoder_embeddings = nn.Embedding(config.vocab_size+1, config.hidden_size, padding_idx=src_tokenizer.vs)
    decoder_embeddings = nn.Embedding(config.vocab_size+1, config.hidden_size, padding_idx=tgt_tokenizer.vs)
    model = TranslationModel(encoder_config, decoder_config, encoder_embeddings, decoder_embeddings)

    return model, src_tokenizer, tgt_tokenizer

def run_experiment(config):

    model, src_tokenizer, tgt_tokenizer = build_model(config)
   
    pad_sequence = PadSequence(src_tokenizer.vs, tgt_tokenizer.vs)
    train_loader = DataLoader(IndicDataset(src_tokenizer, tgt_tokenizer, config.data, True), 
                              batch_size=config.batch_size, 
                              shuffle=False, 
                              collate_fn=pad_sequence)
    eval_loader = DataLoader(IndicDataset(src_tokenizer, tgt_tokenizer, config.data, False), 
                             batch_size=config.eval_size, 
                             shuffle=False, 
                             collate_fn=pad_sequence)

    #writer = SummaryWriter(config.output + 'logs/') 
    writer = None

    (training_loss_values, 
     validation_loss_values, 
     validation_accuracy_values) = train(config, model, train_loader, eval_loader, writer)
    
    return training_loss_values, validation_loss_values, validation_accuracy_values

if __name__ == '__main__':
 
    hin_config = Config(epochs=40, 
                        batch_size=64, 
                        eval_size=16, 
                        vocab_size=25000, 
                        embed_dim=100, 
                        hidden_size=256, 
                        intermediate_size=512,
                        num_attention_heads=1, 
                        num_hidden_layers=1)

    run_experiment(hin_config)
