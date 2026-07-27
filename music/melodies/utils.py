import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='gpt-4-0314')
    parser.add_argument("--exp_dir", type=str)
    
    parser.add_argument('--chain_of_thought', action='store_true', dest='chain_of_thought')
    parser.set_defaults(chain_of_thought=False)
    parser.add_argument('--is_control', action='store_true', dest='is_control')
    parser.set_defaults(is_control=False)
    parser.add_argument(
        '--world',
        choices=['all', 'default', 'counterfactual'],
        default='all',
        help='Evaluate C major, non-C-major keys, or both.',
    )
    return parser
