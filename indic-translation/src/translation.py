import torch.nn as nn
from transformers import BertModel, BertForMaskedLM

class TranslationModel(nn.Module):

    def __init__(self, encoder_config, decoder_config, encoder_embeddings, decoder_embeddings):

        super().__init__() 

        #Creating encoder and decoder with their respective embeddings.
        self.encoder = BertModel(encoder_config)
        self.encoder.set_input_embeddings(encoder_embeddings)

        self.decoder = BertForMaskedLM(decoder_config)
        self.decoder.set_input_embeddings(decoder_embeddings)

    def forward(self, encoder_input_ids, decoder_input_ids):

        encoder_hidden_states, _ = self.encoder(encoder_input_ids)
        loss, logits = self.decoder(decoder_input_ids,
                                    encoder_hidden_states=encoder_hidden_states, 
                                    lm_labels=decoder_input_ids)

        return loss, logits
