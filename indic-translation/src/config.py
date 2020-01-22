from dataclasses import dataclass

@dataclass
class Config:
	
    epochs: int
    batch_size: int
    eval_size: int
    vocab_size: int
    embed_dim: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    lr: float = 0.001
    hidden_act: str = 'gelu'
    dropout_prob: float = 0.1
    data: str = '../data/'
    output: str = '../output/'
    lang: str = 'hi'

hin_config = Config(epochs=40, 
                      batch_size=64, 
                      eval_size=16, 
                      vocab_size=25000, 
                      embed_dim=100, 
                      hidden_size=256, 
                      intermediate_size=512,
                      num_attention_heads=1, 
                      num_hidden_layers=1)
