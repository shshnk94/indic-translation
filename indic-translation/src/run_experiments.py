import argparse
import os
from dataclasses import replace

def run(args):
        
    import config as M
    C = getattr(M, args.lang + '_config')
    
    os.system("""python train.py --data {{C.data}} --output {{C.output}} --lang {{C.lang}} --batch_size {{C.batch_size}} --eval_size {{C.eval_size}} --vocab_size {{C.vocab_size}} --embed_dim {{C.embed_dim}} --hidden_size {{C.hidden_size}} --intermediate_size {{C.intermediate_size}} --num_attention_heads {{C.num_attention_heads}} --num_hidden_layers {{C.num_hidden_layers}} --dropout_prob {{C.dropout_prob}}""")

    #When any new language use dataclasses.replace(C, attr=value) to create the new config instance.

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Experiment engine')
    parser.add_argument('--lang', help="Print the rendered command that will be run")

    args = parser.parse_args()
    run(args)
