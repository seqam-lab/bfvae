import os, argparse
import numpy as np

import torch

#-----------------------------------------------------------------------------#

from utils import mkdirs, str2bool
from model import * 
from dataset import create_dataloader


###############################################################################

class Solver(object):
    
    ####
    def __init__(self, args):
        
        self.args = args
        
        self.name = ( '%s_eta_%s_gamma_%s_zDim_%s' + \
            '_lrVAE_%s_lrD_%s_rseed_%s' ) % \
            ( args.dataset, args.eta, args.gamma, args.z_dim,
              args.lr_VAE, args.lr_D, args.rseed )
        self.name = self.name + '_run_' + str(args.run_id)
        
        self.use_cuda = args.cuda and torch.cuda.is_available()
        
        # data info
        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        if args.dataset.endswith('dsprites'):
            self.nc = 1
        elif args.dataset == '3dfaces':
            self.nc = 1
        else:
            self.nc = 3
            
        # groundtruth factor labels (only available for "dsprites")
        if self.dataset=='dsprites':
            
            # latent factor = (color, shape, scale, orient, pos-x, pos-y)
            #   color = {1} (1)
            #   shape = {1=square, 2=oval, 3=heart} (3)
            #   scale = {0.5, 0.6, ..., 1.0} (6)
            #   orient = {2*pi*(k/39)}_{k=0}^39 (40)
            #   pos-x = {k/31}_{k=0}^31 (32)
            #   pos-y = {k/31}_{k=0}^31 (32)
            # (number of variations = 1*3*6*40*32*32 = 737280)
            
            latent_values = np.load( os.path.join( self.dset_dir, 
              'dsprites-dataset', 'latents_values.npy'), encoding='latin1' )
            self.latent_values = latent_values[:, [1,2,3,4,5]]
                # latent values (actual values);(737280 x 5)
            latent_classes = np.load( os.path.join( self.dset_dir, 
              'dsprites-dataset', 'latents_classes.npy'), encoding='latin1' )
            self.latent_classes = latent_classes[:, [1,2,3,4,5]]
                # classes ({0,1,...,K}-valued); (737280 x 5)
            self.latent_sizes = np.array([3, 6, 40, 32, 32])
            self.N = self.latent_values.shape[0]
                
        # groundtruth factor labels
        elif self.dataset=='oval_dsprites':
            
            latent_classes = np.load( os.path.join( self.dset_dir, 
              'dsprites-dataset', 'latents_classes.npy'), encoding='latin1' )
            idx = np.where(latent_classes[:,1]==1)[0]  # "oval" shape only
            self.latent_classes = latent_classes[idx,:]
            self.latent_classes = self.latent_classes[:,[2,3,4,5]]
                # classes ({0,1,...,K}-valued); (245760 x 4)
            latent_values = np.load( os.path.join( self.dset_dir, 
              'dsprites-dataset', 'latents_values.npy'), encoding='latin1' )
            self.latent_values = latent_values[idx,:]
            self.latent_values = self.latent_values[:,[2,3,4,5]]
                # latent values (actual values);(245760 x 4)
            
            self.latent_sizes = np.array([6, 40, 32, 32])
            self.N = self.latent_values.shape[0]
            
        # groundtruth factor labels
        elif self.dataset=='3dfaces':
            
            # latent factor = (id, azimuth, elevation, lighting)
            #   id = {0,1,...,49} (50)
            #   azimuth = {-1.0,-0.9,...,0.9,1.0} (21)
            #   elevation = {-1.0,0.8,...,0.8,1.0} (11)
            #   lighting = {-1.0,0.8,...,0.8,1.0} (11)
            # (number of variations = 50*21*11*11 = 127050)
            
            latent_classes, latent_values = np.load( os.path.join( 
                self.dset_dir, '3d_faces/rtqichen/gt_factor_labels.npy' ) )
            self.latent_values = latent_values
                # latent values (actual values);(127050 x 4)
            self.latent_classes = latent_classes
                # classes ({0,1,...,K}-valued); (127050 x 4)
            self.latent_sizes = np.array([50, 21, 11, 11])
            self.N = self.latent_values.shape[0]
            
        elif self.dataset=='edinburgh_teapots':
            
            # latent factor = (azimuth, elevation, R, G, B)
            #   azimuth = [0, 2*pi]
            #   elevation = [0, pi/2]
            #   R, G, B = [0,1]
            #
            #   "latent_values" = original (real) factor values
            #   "latent_classes" = equal binning into K=10 classes
            #
            # (refer to "data/edinburgh_teapots/my_make_split_data.py")
            
            K = 10
            val_ranges = [2*np.pi, np.pi/2, 1, 1, 1]
            bins = []
            for j in range(5):
                bins.append(np.linspace(0, val_ranges[j], K+1))
            
            latent_values = np.load( os.path.join( self.dset_dir, 
              'edinburgh_teapots', 'gtfs_tr.npz' ) )['data']
            latent_values = np.concatenate( ( latent_values,
                np.load( os.path.join( self.dset_dir, 
                        'edinburgh_teapots', 'gtfs_va.npz' ) )['data'] ),
                axis = 0 )
            latent_values = np.concatenate( ( latent_values,
                np.load( os.path.join( self.dset_dir, 
                        'edinburgh_teapots', 'gtfs_te.npz' ) )['data'] ),
                axis = 0 )
            self.latent_values = latent_values
            
            latent_classes = np.zeros(latent_values.shape)
            for j in range(5):
                latent_classes[:,j] = np.digitize(latent_values[:,j], bins[j])
            self.latent_classes = latent_classes-1  # {0,...,K-1}-valued
                
            self.latent_sizes = K*np.ones(5, 'int64')
            self.N = self.latent_values.shape[0]
            
        elif self.dataset=='celeba':
            
            # 40 attrs, all binary {-1,+1} 
            
            latent_values, _ = np.load( os.path.join( 
                self.dset_dir, 'celeba/np_attr_celeba.npy' ) )
            self.latent_values = latent_values
                # latent values (actual values);(202599 x 40)
            self.latent_sizes = np.array([2]*40)
            self.N = self.latent_values.shape[0]

        # networks and optimizers
        self.batch_size = args.batch_size
        self.z_dim = args.z_dim
        
        # checkpoints
        self.ckpt_dir = os.path.join("ckpts", self.name)
        
        # create dirs: "records", "ckpts", "outputs" (if not exist)
        mkdirs("records");  mkdirs("outputs")
        
        # records
        self.record_file = 'records/%s.txt' % ("save_z_" + self.name)

        # outputs (where latent vectors are saved)
        self.latent_file = 'outputs/%s.npz' % ("save_z_" + self.name)
        
        # load a previously saved model
        self.ckpt_load_iter = args.ckpt_load_iter
        print('Loading saved models (iter: %d)...' % self.ckpt_load_iter)
        self.load_checkpoint()
        print('...done')
            
        if self.use_cuda:
            print('Models moved to GPU...')
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            print('...done')
        
        self.set_mode(train=False)


    ####
    def run(self):
            
        # prepare dataloader (iterable)
        print('Start loading data...')
        self.data_loader = create_dataloader(self.args)
        print('...done')
        
        # iterator from dataloader
        iterator = iter(self.data_loader)
        iter_per_epoch = len(iterator)
        
        self.Z = []
        self.GT = []
        
        #----#
                
        prn_str = 'Start going through the entire data...'
        print(prn_str)
        self.dump_to_record(prn_str)
        
        for iteration in range(1, 100000000):

            # reset data iterators for each epoch
            if iteration>1 and (iteration-1) % iter_per_epoch == 0:
                break
            
            with torch.no_grad():
                
                # sample a mini-batch
                X, ids = next(iterator)  # (n x C x H x W)
                if self.use_cuda:
                    X = X.cuda()
                    ids = ids.cuda()
                    
                # enc(X)
                mu, std, logvar = self.encoder(X)
                
                self.Z.append(mu.cpu().detach().numpy())
                
                ids = ids.cpu().detach().numpy()
                self.GT.append(self.latent_values[ids,:])
                
            prn_str = 'batch iter = %d / %d done' % (iteration, iter_per_epoch)
            print(prn_str)
            self.dump_to_record(prn_str)
                
        self.Z = np.vstack(self.Z)
        self.GT = np.vstack(self.GT)
        np.savez(self.latent_file, name1=self.Z, name2=self.GT)
        
            
    ####
    def set_mode(self, train=True):
        
        if train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
            

    ####
    def load_checkpoint(self):
        
        encoder_path = os.path.join( self.ckpt_dir, 
          'iter_%s_encoder.pt' % self.ckpt_load_iter )
        decoder_path = os.path.join( self.ckpt_dir, 
          'iter_%s_decoder.pt' % self.ckpt_load_iter )
        
        if self.use_cuda:
            self.encoder = torch.load(encoder_path)
            self.decoder = torch.load(decoder_path)
        else:
            self.encoder = torch.load(encoder_path, map_location='cpu')
            self.decoder = torch.load(decoder_path, map_location='cpu')


    ####
    def dump_to_record(self, prn_str):
        
        record = open(self.record_file, 'a')
        record.write('%s\n' % (prn_str,))
        record.close()


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
    
    parser.add_argument( '--eta', default=1.0, type=float, 
      help='impact of regaularizer for the prior variances' )
    parser.add_argument( '--lr_VAE', default=1e-4, type=float, 
      help='learning rate of the VAE' )
    parser.add_argument( '--lr_D', default=1e-4, type=float, 
      help='learning rate of the discriminator' )
    parser.add_argument( '--gamma', default=6.4, type=float, 
      help='gamma (impact of total correlation)' )
    parser.add_argument( '--z_dim', default=10, type=int, 
        help='dimension of the representation z' )
    parser.add_argument( '--run_id', default=0, type=int, 
        help='run id' )
    parser.add_argument( '--rseed', default=0, type=int, 
      help='random seed (default=0)' )
    # and the iter# of the previously saved model
    parser.add_argument( '--ckpt_load_iter', default=300000, type=int, 
        help='iter# to load the previously saved model' )

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

    solver.run()


###############################################################################
    
if __name__ == "__main__":
    
    parser = create_parser()
    args = parser.parse_args()
    print_opts(args)
    
    main(args)
