import argparse
import numpy as np
import torch

#-----------------------------------------------------------------------------#

from solver_test import Solver
from utils import str2bool

###############################################################################
    
def print_opts(opts):
    
    '''
    Print the values of all command-line arguments
    '''
    
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)

#-----------------------------------------------------------------------------#
    
def create_parser():
    
    '''
    Create a parser for command-line arguments
    '''
    
    parser = argparse.ArgumentParser()
    
    # specify the model to load
    # (refer to the name of the folder that contains the model you wanna load)
    parser.add_argument( '--dataset', default='oval_dsprites', type=str, 
        help='dataset name' )
    parser.add_argument( '--gamma', default=35.0, type=float, 
        help='gamma (impact of total correlation)' )
    parser.add_argument( '--etaS', default=1.0, type=float, 
      help='impact of L1-norm loss (spaseness) term for the relevance vector' )
    parser.add_argument( '--etaH', default=100.0, type=float, 
      help='impact of entropy loss term for the relevance vector' )
    parser.add_argument( '--z_dim', default=10, type=int, 
        help='dimension of the representation z' )
    parser.add_argument( '--run_id', default=0, type=int, 
        help='run id' )
    parser.add_argument( '--rseed', default=0, type=int, 
      help='random seed (default=0)' )
    parser.add_argument( '--lr_VAE', default=1e-4, type=float, 
      help='learning rate of the VAE' )
    parser.add_argument( '--lr_D', default=1e-4, type=float, 
      help='learning rate of the discriminator' )
    # and the iter# of the previously saved model
    parser.add_argument( '--ckpt_load_iter', default=300000, type=int, 
        help='iter# to load the previously saved model' )
        
    # what to do in this test
    parser.add_argument( '--num_recon', default=0, type=int, 
        help='# times to repeat reconstruction (0 for no evaluation)' )
    parser.add_argument( '--num_synth', default=0, type=int, 
        help='# times to repeat image synthesis (0 for no evaluation)' )
    parser.add_argument( '--num_trvsl', default=0, type=int, 
        help='# times to repeat latent traversal (0 for no evaluation)' )
    #
    parser.add_argument( '--losses', action='store_true', default=False, 
        help="whether or not to evaluate losses (incl. individual kl's)" )
    #
    parser.add_argument( '--num_eval_metric1', default=0, type=int, 
        help='# times to repeat evaluating metric 1 (0 for no evaluation)' )
    parser.add_argument( '--num_eval_metric2', default=0, type=int, 
        help='# times to repeat evaluating metric 2 (0 for no evaluation)' )
        
    # hyperparameters that need to be consistent with the above saved model
    parser.add_argument( '--dset_dir', default='data', type=str, 
        help='dataset directory' )
    parser.add_argument( '--image_size', default=64, type=int, 
        help='image size; now only (64 x 64) is supported' )
        
    # other parameters
    parser.add_argument( '--cuda', default=True, type=str2bool, 
        help='enable cuda' )
    parser.add_argument( '--num_workers', default=0, type=int, 
        help='dataloader num_workers' )    
    parser.add_argument( '--batch_size', default=64, type=int, 
        help='batch size' )

    return parser

#-----------------------------------------------------------------------------#

def main(args):
    
    # set the random seed manually for reproducibility
    SEED = args.rseed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    solver = Solver(args)
    
    solver.test()


###############################################################################
    
if __name__ == "__main__":
    
    parser = create_parser()
    args = parser.parse_args()
    print_opts(args)
    
    main(args)
