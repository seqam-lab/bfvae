import os
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

#-----------------------------------------------------------------------------#

from utils import DataGather, mkdirs, grid2gif
from model import *
from dataset import create_dataloader

###############################################################################

class Solver(object):
    
    ####
    def __init__(self, args):
        
        self.args = args
        
        self.name = ( '%s_gamma_%s_etaS_%s_etaH_%s_zDim_%s' + \
            '_lrVAE_%s_lrD_%s_rseed_%s' ) % \
            ( args.dataset, args.gamma, args.etaS, args.etaH, args.z_dim,
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
            
        elif self.dataset=='celeba':
        
            self.N = 202599
            self.eval_metrics = False
            
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
            
        # networks and optimizers
        self.batch_size = args.batch_size
        self.z_dim = args.z_dim
        
        # checkpoints
        self.ckpt_dir = os.path.join("ckpts", self.name)
        
        # create dirs: "records", "ckpts", "outputs" (if not exist)
        mkdirs("records");  mkdirs("outputs")
            
        # records
        self.record_file = 'records/%s.txt' % ("refined_traverse_" + self.name)

        # outputs
        self.output_dir_trvsl = os.path.join( "outputs",  
                                              "refined_traverse_" + self.name )
        
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
    def test(self):

        ones = torch.ones( self.batch_size, dtype=torch.long )
        zeros = torch.zeros( self.batch_size, dtype=torch.long )
        if self.use_cuda:
            ones = ones.cuda()
            zeros = zeros.cuda()
                    
        # prepare dataloader (iterable)
        print('Start loading data...')
        self.data_loader = create_dataloader(self.args)
        print('...done')
        
        # latent traversal
        prn_str = 'Start doing refined latent traversal...'
        print(prn_str)
        self.dump_to_record(prn_str)
        self.save_refined_traverse( self.ckpt_load_iter )

    
    ####
    def save_refined_traverse( self, iters ):
        
        encoder = self.encoder
        decoder = self.decoder
        
        num_vars = 30  # number of variations in each dim
        inter = torch.arange(-16.0, 16.001, 0.2)  # used in "solver_test.py"
        inter = inter.detach().numpy()
        
        interpolation = [0]*self.z_dim
        for row in range(self.z_dim):
            interpolation[row] = torch.tensor(
                np.linspace(-3.0, 3.0, num_vars), dtype=torch.float32 )
        
        ####
        
        if self.dataset.lower() == 'oval_dsprites':
            
            idx = 87040  # image ID
            factor_ranges = {}
            factor_ranges[0] = inter[[1, 160]]
            factor_ranges[1] = inter[[1, 160]]
            factor_ranges[4] = inter[[1, 160]]
            factor_ranges[5] = inter[[43, 120]]
            factor_ranges[7] = inter[[4, 160]]
            
        elif self.dataset.lower() == '3dfaces':
            
            idx = 6245  # image ID
            factor_ranges = {}
            factor_ranges[0] = inter[[76, 87]]  # vary z_0 from inter[76] to inter[87]
            factor_ranges[3] = inter[[68, 90]]
            factor_ranges[5] = inter[[79, 82]]
            factor_ranges[7] = inter[[58, 80]]
            
        elif self.dataset.lower() == 'edinburgh_teapots':
            
            idx = 46203  # image ID
            factor_ranges = {}
            factor_ranges[2] = inter[[72, 82]]
            factor_ranges[3] = inter[[60, 100]]
            factor_ranges[4] = inter[[68, 96]]
            factor_ranges[6] = inter[[74, 98]]
            factor_ranges[7] = inter[[73, 91]]
            factor_ranges[8] = inter[[75, 94]]
            factor_ranges[9] = inter[[64, 81]]
            
        elif self.dataset.lower() == 'celeba':
            
            all_idx = [4195, 95070]  # image IDs
            
            all_factor_ranges = [0]*len(all_idx)
            
            cnt = -1
            
            # 4195
            cnt += 1
            all_factor_ranges[cnt] = {}
            all_factor_ranges[cnt][2-1] = inter[[98, 120]]
            all_factor_ranges[cnt][7-1] = inter[[56, 89]]
            all_factor_ranges[cnt][12-1] = inter[[72, 112]]
            all_factor_ranges[cnt][20-1] = inter[[40, 84]]
            
            # 95070
            cnt += 1
            all_factor_ranges[cnt] = {}
            all_factor_ranges[cnt][2-1] = inter[[72, 119]]
            all_factor_ranges[cnt][7-1] = inter[[46, 75]]
            all_factor_ranges[cnt][12-1] = inter[[73, 115]]
            all_factor_ranges[cnt][20-1] = inter[[17, 56]]
            
#            all_idx = [4195, 95070]  # image IDs
#            
#            all_factor_ranges = [0]*len(all_idx)
#            
#            cnt = -1
#            
#            # 4195
#            cnt += 1
#            all_factor_ranges[cnt] = {}
#            all_factor_ranges[cnt][4-1] = inter[[57, 80]]
#            all_factor_ranges[cnt][9-1] = inter[[60, 80]]
#            all_factor_ranges[cnt][12-1] = inter[[49, 87]]
#            all_factor_ranges[cnt][15-1] = inter[[48, 70]]
#            
#            # 95070
#            cnt += 1
#            all_factor_ranges[cnt] = {}
#            all_factor_ranges[cnt][4-1] = inter[[67, 86]]
#            all_factor_ranges[cnt][9-1] = inter[[83, 106]]
#            all_factor_ranges[cnt][12-1] = inter[[54, 95]]
#            all_factor_ranges[cnt][15-1] = inter[[59, 81]]
            
#            all_idx = [4195, 2428, 148838, 95070, 118857]  # image IDs
#            
#            all_factor_ranges = [0]*len(all_idx)
#            
#            cnt = -1
#            
#            # 4195
#            cnt += 1
#            all_factor_ranges[cnt] = {}
#            all_factor_ranges[cnt][9-1] = inter[[55, 80]]
#            all_factor_ranges[cnt][12-1] = inter[[49, 87]]
#            all_factor_ranges[cnt][15-1] = inter[[48, 70]]
#            all_factor_ranges[cnt][3-1] = inter[[44, 129]]
#            all_factor_ranges[cnt][11-1] = inter[[76, 107]]
#            all_factor_ranges[cnt][13-1] = inter[[77, 102]]
#            
#            # 2428
#            cnt += 1
#            all_factor_ranges[cnt] = {}
#            all_factor_ranges[cnt][9-1] = inter[[60, 87]]
#            all_factor_ranges[cnt][12-1] = inter[[46, 117]]
#            all_factor_ranges[cnt][15-1] = inter[[43, 88]]
#            all_factor_ranges[cnt][3-1] = inter[[50, 129]]
#            all_factor_ranges[cnt][11-1] = inter[[80, 110]]
#            all_factor_ranges[cnt][13-1] = inter[[45, 101]]
#            
#            # 148838
#            cnt += 1
#            all_factor_ranges[cnt] = {}
#            all_factor_ranges[cnt][9-1] = inter[[56, 105]]
#            all_factor_ranges[cnt][12-1] = inter[[46, 104]]
#            all_factor_ranges[cnt][15-1] = inter[[40, 80]]
#            all_factor_ranges[cnt][3-1] = inter[[43, 129]]
#            all_factor_ranges[cnt][11-1] = inter[[82, 111]]
#            all_factor_ranges[cnt][13-1] = inter[[61, 106]]
#            
#            # 95070
#            cnt += 1
#            all_factor_ranges[cnt] = {}
#            all_factor_ranges[cnt][9-1] = inter[[72, 116]]
#            all_factor_ranges[cnt][12-1] = inter[[52, 104]]
#            all_factor_ranges[cnt][15-1] = inter[[50, 91]]
#            all_factor_ranges[cnt][3-1] = inter[[23, 112]]
#            all_factor_ranges[cnt][11-1] = inter[[83, 111]]
#            all_factor_ranges[cnt][13-1] = inter[[79, 108]]
#            
#            # 118857
#            cnt += 1
#            all_factor_ranges[cnt] = {}
#            all_factor_ranges[cnt][9-1] = inter[[85, 109]]
#            all_factor_ranges[cnt][12-1] = inter[[52, 105]]
#            all_factor_ranges[cnt][15-1] = inter[[60, 92]]
#            all_factor_ranges[cnt][3-1] = inter[[45, 110]]
#            all_factor_ranges[cnt][11-1] = inter[[77, 106]]
#            all_factor_ranges[cnt][13-1] = inter[[60, 110]]
            
        else:
            
            raise NotImplementedError

        ####
        
        if self.dataset.lower() == 'celeba':
            
            num_vars = 11
            
            for i, idx in enumerate(all_idx):
                
                interpolation = {}
                for key in all_factor_ranges[i]:
                    interpolation[key] = torch.tensor( np.linspace( 
                           all_factor_ranges[i][key][0], 
                           all_factor_ranges[i][key][1], num_vars ), 
                       dtype=torch.float32 )
                
                img = self.data_loader.dataset.__getitem__(idx)[0]
                if self.use_cuda:
                    img = img.cuda()
                img = img.unsqueeze(0)
                z_ori, _, _ = encoder(img) 
                
                # do for each dim 
                for key in all_factor_ranges[i]:
                
                    # do traversal and collect generated images
                    gifs = []
                    z = z_ori.clone()
                    for val in interpolation[key]:
                        z[:,key] = val
                        sample = torch.sigmoid(decoder(z)).data
                        gifs.append(sample)
                        
                    # save the generated files, also the animated gifs  
                    out_dir = os.path.join(self.output_dir_trvsl, str(iters))
                    mkdirs(self.output_dir_trvsl)
                    mkdirs(out_dir)
                    gifs = torch.cat(gifs)
                    gifs = gifs.view( 
                        1, 1, num_vars, self.nc, 64, 64 ).transpose(1,2)
                    gifs = gifs.squeeze(2)

                    save_image( tensor=gifs[0].cpu(),
                        filename=os.path.join(out_dir, 
                            'all_%d_z%02d.jpg' % (idx,key+1)),
                        nrow=num_vars, pad_value=1 ) 

#                    for j in range(num_vars):
#                        I = torch.cat([img, gifs[0][j]], dim=0)
#                            # input image leftmost
#                        I2 = gifs[0][j]  # no leftmost input image
#                        save_image( tensor=I.cpu(),
#                            filename=os.path.join(out_dir, 
#                                '%d_z%02d_%03d.jpg' % (idx,key+1,j)),
#                            nrow=1+1, pad_value=1 )
#                        save_image( tensor=I2.cpu(),
#                            filename=os.path.join(out_dir, 
#                                'nox_%d_z%02d_%03d.jpg' % (idx,key+1,j)),
#                            nrow=1, pad_value=1 )                    
            
            return
        
        ####
        
        for key in factor_ranges:
            interpolation[key] = torch.tensor( np.linspace( 
                factor_ranges[key][0], factor_ranges[key][1], num_vars ), 
                    dtype=torch.float32 )
    
        img = self.data_loader.dataset.__getitem__(idx)[0]
        if self.use_cuda:
            img = img.cuda()
        img = img.unsqueeze(0)
        img_zmu, _, _ = encoder(img)        
            
        # do traversal and collect generated images 
        gifs = []
        z_ori = img_zmu
        for row in range(self.z_dim):
            z = z_ori.clone()
            for val in interpolation[row]:
                z[:,row] = val
                sample = torch.sigmoid(decoder(z)).data
                gifs.append(sample)

        # save the generated files, also the animated gifs            
        out_dir = os.path.join(self.output_dir_trvsl, str(iters))
        mkdirs(self.output_dir_trvsl)
        mkdirs(out_dir)
        gifs = torch.cat(gifs)
        gifs = gifs.view( 
            1, self.z_dim, num_vars, self.nc, 64, 64 ).transpose(1,2)
        for j in range(num_vars):
            I = torch.cat([img, gifs[0][j]], dim=0)  # input image leftmost
            I2 = gifs[0][j]  # no leftmost input image
            save_image( tensor=I.cpu(),
                filename=os.path.join(out_dir, '%d_%03d.jpg' % (idx,j)),
                nrow=1+self.z_dim, pad_value=1 )
            save_image( tensor=I2.cpu(),
                filename=os.path.join(out_dir, 'nox_%d_%03d.jpg' % (idx,j)),
                nrow=self.z_dim, pad_value=1 )
        
        # make animated gif
        grid2gif( out_dir, str(idx), 
            str(os.path.join(out_dir, str(idx)+'.gif')), delay=10 )
        grid2gif( out_dir, 'nox_'+str(idx), 
            str(os.path.join(out_dir, 'nox_'+str(idx)+'.gif')), delay=10 )

    
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
        
        
        