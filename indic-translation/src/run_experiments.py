import argparse
import json

def run(args):
    C = load_config(config_dir='.', config_name=args.config_name, update_args=args)

    #print (C.er.build)
    expts = C.get_experiments()
    dispatch_expts(expts, engine=args.engine, dry_run=args.d)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Experiment engine')
    parser.add_argument('--lang', help="Print the rendered command that will be run")

    args = parser.parse_args()
    run(args)
