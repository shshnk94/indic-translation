import torch.nn as nn

class TranslationModel(nn.Module):

    def __init__(self, encoder, decoder):

        super().__init__() 

        #Creating encoder and decoder with their respective embeddings.
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_input_ids, decoder_input_ids):

        encoder_hidden_states = self.encoder(encoder_input_ids)[0]
        loss, logits = self.decoder(decoder_input_ids,
                                    encoder_hidden_states=encoder_hidden_states, 
                                    masked_lm_labels=decoder_input_ids)

        return loss, logits
