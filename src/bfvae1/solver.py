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
        
        self.name = ( '%s_gamma_%s_zDim_%s' + \
            '_lrVAE_%s_lrD_%s_rseed_%s' ) % \
            ( args.dataset, args.gamma, args.z_dim,
              args.lr_VAE, args.lr_D, args.rseed )
                # to be appended by run_id
        
        self.use_cuda = args.cuda and torch.cuda.is_available()
         
        self.max_iter = int(args.max_iter)
        
        # do it every specified iters
        self.print_iter = args.print_iter
        self.ckpt_save_iter = args.ckpt_save_iter
        self.output_save_iter = args.output_save_iter
        
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
            
            if args.eval_metrics:
                self.eval_metrics = True
                self.eval_metrics_iter = args.eval_metrics_iter
                
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
            
            if args.eval_metrics:
                self.eval_metrics = True
                self.eval_metrics_iter = args.eval_metrics_iter
                
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
            
            if args.eval_metrics:
                self.eval_metrics = True
                self.eval_metrics_iter = args.eval_metrics_iter
                
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
            
            if args.eval_metrics:
                self.eval_metrics = True
                self.eval_metrics_iter = args.eval_metrics_iter

        # networks and optimizers
        self.batch_size = args.batch_size
        self.z_dim = args.z_dim
        self.gamma = args.gamma
        self.lr_VAE = args.lr_VAE
        self.beta1_VAE = args.beta1_VAE
        self.beta2_VAE = args.beta2_VAE
        self.lr_D = args.lr_D
        self.beta1_D = args.beta1_D
        self.beta2_D = args.beta2_D
        
        # visdom setup
        self.viz_on = args.viz_on
        if self.viz_on:
            
            self.win_id = dict( 
                DZ='win_DZ', recon='win_recon', kl='win_kl', 
                kl_alpha='win_kl_alpha' )
            self.line_gather = DataGather( 
                'iter', 'p_DZ', 'p_DZ_perm', 'recon', 'kl', 'kl_alpha' )
            
            if self.eval_metrics:
                self.win_id['metrics']='win_metrics'
            
            import visdom
            
            self.viz_port = args.viz_port  # port number, eg, 8097
            self.viz = visdom.Visdom(port=self.viz_port)
            self.viz_ll_iter = args.viz_ll_iter
            self.viz_la_iter = args.viz_la_iter
            
            self.viz_init()
        
        # create dirs: "records", "ckpts", "outputs" (if not exist)
        mkdirs("records");  mkdirs("ckpts");  mkdirs("outputs")
        
        # set run id
        if args.run_id < 0:  # create a new id
            k = 0;  rfname = os.path.join("records", self.name + '_run_0.txt')
            while os.path.exists(rfname):
                k += 1
                rfname = os.path.join("records", self.name + '_run_%d.txt' % k)
            self.run_id = k
        else:  # user-provided id
            self.run_id = args.run_id
            
        # finalize name
        self.name = self.name + '_run_' + str(self.run_id)

        # records (text file to store console outputs)
        self.record_file = 'records/%s.txt' % self.name

        # checkpoints
        self.ckpt_dir = os.path.join("ckpts", self.name)

        # outputs
        self.output_dir_recon = os.path.join("outputs", self.name + '_recon')  
          # dir for reconstructed images
        self.output_dir_synth = os.path.join("outputs", self.name + '_synth')  
          # dir for synthesized images
        self.output_dir_trvsl = os.path.join("outputs", self.name + '_trvsl')  
          # dir for latent traversed images
        
        #### create a new model or load a previously saved model
        
        self.ckpt_load_iter = args.ckpt_load_iter
        
        if self.ckpt_load_iter == 0:  # create a new model
        
            # create a vae model
            if args.dataset.endswith('dsprites'):
                self.encoder = Encoder1(self.z_dim)
                self.decoder = Decoder1(self.z_dim)
            elif args.dataset == '3dfaces':
                self.encoder = Encoder3(self.z_dim)
                self.decoder = Decoder3(self.z_dim)
            elif args.dataset == 'celeba':
                self.encoder = Encoder4(self.z_dim)
                self.decoder = Decoder4(self.z_dim)
            elif args.dataset.endswith('teapots'):
                # self.encoder = Encoder4(self.z_dim)
                # self.decoder = Decoder4(self.z_dim)
                self.encoder = Encoder_ResNet(self.z_dim)
                self.decoder = Decoder_ResNet(self.z_dim)
            else:
                pass  #self.VAE = FactorVAE2(self.z_dim)
            
            # create a prior alpha model
            self.prior_alpha = PriorAlphaParams(self.z_dim)
            
            # create a posterior alpha model
            self.post_alpha = PostAlphaParams(self.z_dim)
            
            # create a discriminator model
            self.D = Discriminator(self.z_dim)
            
        else:  # load a previously saved model
            
            print('Loading saved models (iter: %d)...' % self.ckpt_load_iter)
            self.load_checkpoint()
            print('...done')
            
        if self.use_cuda:
            print('Models moved to GPU...')
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.prior_alpha = self.prior_alpha.cuda()
            self.post_alpha = self.post_alpha.cuda()
            self.D = self.D.cuda()
            print('...done')
        
        # get VAE parameters
        vae_params = list(self.encoder.parameters()) + \
            list(self.decoder.parameters()) + \
            list(self.prior_alpha.parameters()) + \
            list(self.post_alpha.parameters())
                    
        # get discriminator parameters
        dis_params = list(self.D.parameters())
    
        # create optimizers
        self.optim_vae = optim.Adam( vae_params, 
          lr=self.lr_VAE, betas=[self.beta1_VAE, self.beta2_VAE] )
        self.optim_dis = optim.Adam( dis_params, 
          lr=self.lr_D, betas=[self.beta1_D, self.beta2_D] )


    ####
    def train(self):
        
        self.set_mode(train=True)

        ones = torch.ones( self.batch_size, dtype=torch.long )
        zeros = torch.zeros( self.batch_size, dtype=torch.long )
        if self.use_cuda:
            ones = ones.cuda()
            zeros = zeros.cuda()
                    
        # prepare dataloader (iterable)
        print('Start loading data...')
        self.data_loader = create_dataloader(self.args)
        print('...done')
        
        # iterators from dataloader
        iterator1 = iter(self.data_loader)
        iterator2 = iter(self.data_loader)
        
        iter_per_epoch = min(len(iterator1), len(iterator2))
        
        start_iter = self.ckpt_load_iter + 1
        epoch = int(start_iter / iter_per_epoch)
        
        for iteration in range(start_iter, self.max_iter+1):

            # reset data iterators for each epoch
            if iteration % iter_per_epoch == 0:
                print('==== epoch %d done ====' % epoch)
                epoch+=1
                iterator1 = iter(self.data_loader)
                iterator2 = iter(self.data_loader)
            
            #============================================
            #          TRAIN THE VAE (ENC & DEC)
            #============================================
        
            # sample a mini-batch
            X, ids = next(iterator1)  # (n x C x H x W)
            if self.use_cuda:
                X = X.cuda()
                
            # enc(X)
            mu, std, logvar = self.encoder(X)
            
            # prior alpha params
            a, b = self.prior_alpha()
            
            # posterior alpha params
            ah, bh = self.post_alpha()
            
            # kl loss
            kls = 0.5 * ( \
                  (ah/bh)*(mu**2+std**2) - 1.0 + \
                  bh.log() - ah.digamma() - logvar )  # (n x z_dim)
            loss_kl = kls.sum(1).mean()
            
            # kl loss on alpha
            kls_alpha = ( \
                (ah-a)*ah.digamma() - ah.lgamma() + a.lgamma() + \
                a*(bh.log()-b.log()) + (ah/bh)*(b-bh) )  # z_dim-dim
            loss_kl_alpha = kls_alpha.sum() / self.N
            
            # reparam'ed samples
            if self.use_cuda:
                Eps = torch.cuda.FloatTensor(mu.shape).normal_()
            else:
                Eps = torch.randn(mu.shape)
            Z = mu + Eps*std
            
            # dec(Z) 
            X_recon = self.decoder(Z)
            
            # recon loss
            loss_recon = F.binary_cross_entropy_with_logits( 
                X_recon, X, reduction='sum' ).div(X.size(0))

            # dis(Z)
            DZ = self.D(Z)
            
            # tc loss
            loss_tc = (DZ[:,0] - DZ[:,1]).mean()
                        
            # total loss for vae
            vae_loss = loss_recon + loss_kl + loss_kl_alpha + \
                       self.gamma*loss_tc
            
            # update vae
            self.optim_vae.zero_grad()
            vae_loss.backward()
            self.optim_vae.step()

            
            #============================================
            #          TRAIN THE DISCRIMINATOR
            #============================================
            
            # sample a mini-batch
            X2, ids = next(iterator2)  # (n x C x H x W)
            if self.use_cuda:
                X2 = X2.cuda()
            
            # enc(X2)
            mu, std, _ = self.encoder(X2)
            
            # reparam'ed samples
            if self.use_cuda:
                Eps = torch.cuda.FloatTensor(mu.shape).normal_()
            else:
                Eps = torch.randn(mu.shape)
            Z = mu + Eps*std
                        
            # dis(Z)
            DZ = self.D(Z)
            
            # dim-wise permutated Z over the mini-batch
            perm_Z = []
            for zj in Z.split(1, 1):
                idx = torch.randperm(Z.size(0))
                perm_zj = zj[idx]
                perm_Z.append(perm_zj)
            Z_perm = torch.cat(perm_Z, 1)
            Z_perm = Z_perm.detach()

            # dis(Z_perm)
            DZ_perm = self.D(Z_perm)
            
            # discriminator loss
            dis_loss = 0.5*( F.cross_entropy(DZ, zeros) + 
                F.cross_entropy(DZ_perm, ones) )

            # update discriminator
            self.optim_dis.zero_grad()
            dis_loss.backward()
            self.optim_dis.step()
            
            ##########################################

            # print the losses
            if iteration % self.print_iter == 0:
                prn_str = ( '[iter %d (epoch %d)] vae_loss: %.3f | ' + \
                    'dis_loss: %.3f\n    ' + \
                    '(recon: %.3f, kl: %.3f, kl_alpha: %.3f, tc: %.3f)' \
                  ) % \
                  ( iteration, epoch, vae_loss.item(), dis_loss.item(), 
                    loss_recon.item(), loss_kl.item(), loss_kl_alpha.item(),
                    loss_tc.item() )
                prn_str += '\n    a = {}'.format(
                    a.detach().cpu().numpy().round(2) )
                prn_str += '\n    b = {}'.format(
                    b.detach().cpu().numpy().round(2) )
                prn_str += '\n    ah = {}'.format(
                    ah.detach().cpu().numpy().round(2) )
                prn_str += '\n    bh = {}'.format(
                    bh.detach().cpu().numpy().round(2) )
                print(prn_str)
                if self.record_file:
                    record = open(self.record_file, 'a')
                    record.write('%s\n' % (prn_str,))
                    record.close()

            # save model parameters
            if iteration % self.ckpt_save_iter == 0:
               self.save_checkpoint(iteration)
               
            # save output images (recon, synth, etc.)
            if iteration % self.output_save_iter == 0:
                
                # 1) save the recon images
                self.save_recon(iteration, X, torch.sigmoid(X_recon).data)   
                
                # 2) save the synth images
                self.save_synth(iteration, howmany=100)
                
                # 3) save the latent traversed images
                if self.dataset.lower() == '3dchairs':
                    self.save_traverse(iteration, limb=-2, limu=2, inter=0.5)
                else:
                    self.save_traverse(iteration, limb=-3, limu=3, inter=0.1)
                    

            # (visdom) insert current line stats
            if self.viz_on and (iteration % self.viz_ll_iter == 0):
                
                # compute discriminator accuracy
                p_DZ = F.softmax(DZ,1)[:,0].detach()
                p_DZ_perm = F.softmax(DZ_perm,1)[:,0].detach()
                
                # insert line stats
                self.line_gather.insert( iter=iteration, 
                  p_DZ=p_DZ.mean().item(), p_DZ_perm=p_DZ_perm.mean().item(),
                  recon=loss_recon.item(), kl=loss_kl.item(), 
                  kl_alpha=loss_kl_alpha.item()
                )

            # (visdom) visualize line stats (then flush out)
            if self.viz_on and (iteration % self.viz_la_iter == 0):
                self.visualize_line()
                self.line_gather.flush()

            # evaluate metrics
            if self.eval_metrics and (iteration % self.eval_metrics_iter == 0):
                
                metric1, _ = self.eval_disentangle_metric1()
                metric2, _ = self.eval_disentangle_metric2()
                
                prn_str = ( '********\n[iter %d (epoch %d)] ' + \
                  'metric1 = %.4f, metric2 = %.4f\n********' ) % \
                  (iteration, epoch, metric1, metric2)
                print(prn_str)
                if self.record_file:
                    record = open(self.record_file, 'a')
                    record.write('%s\n' % (prn_str,))
                    record.close()
                
                # (visdom) visulaize metrics
                if self.viz_on:
                    self.visualize_line_metrics(iteration, metric1, metric2)
                

    ####
    def eval_disentangle_metric1(self):
        
        # some hyperparams
        num_pairs = 800  # # data pairs (d,y) for majority vote classification
        bs = 50  # batch size
        nsamps_per_factor = 100  # samples per factor
        nsamps_agn_factor = 5000  # factor-agnostic samples
        
        self.set_mode(train=False)
        
        # 1) estimate variances of latent points factor agnostic
        
        dl = DataLoader( 
          self.data_loader.dataset, batch_size=bs,
          shuffle=True, num_workers=self.args.num_workers, pin_memory=True )
        iterator = iter(dl)
        
        M = []
        for ib in range(int(nsamps_agn_factor/bs)):
            
            # sample a mini-batch
            Xb, _ = next(iterator)  # (bs x C x H x W)
            if self.use_cuda:
                Xb = Xb.cuda()
            
            # enc(Xb)
            mub, _, _ = self.encoder(Xb)  # (bs x z_dim)

            M.append(mub.cpu().detach().numpy())
            
        M = np.concatenate(M, 0)
        
        # estimate sample vairance and mean of latent points for each dim
        vars_agn_factor = np.var(M, 0)

        # 2) estimatet dim-wise vars of latent points with "one factor fixed"

        factor_ids = range(0, len(self.latent_sizes))  # true factor ids
        vars_per_factor = np.zeros([num_pairs,self.z_dim])  
        true_factor_ids = np.zeros(num_pairs, np.int)  # true factor ids

        # prepare data pairs for majority-vote classification
        i = 0
        for j in factor_ids:  # for each factor

            # repeat num_paris/num_factors times
            for r in range(int(num_pairs/len(factor_ids))):

                # a true factor (id and class value) to fix
                fac_id = j
                fac_class = np.random.randint(self.latent_sizes[fac_id])

                # randomly select images (with the fixed factor)
                indices = np.where( 
                  self.latent_classes[:,fac_id]==fac_class )[0]
                np.random.shuffle(indices)
                idx = indices[:nsamps_per_factor]
                M = []
                for ib in range(int(nsamps_per_factor/bs)):
                    Xb, _ = dl.dataset[ idx[(ib*bs):(ib+1)*bs] ]
                    if Xb.shape[0]<1:  # no more samples
                        continue;
                    if self.use_cuda:
                        Xb = Xb.cuda()
                    mub, _, _ = self.encoder(Xb)  # (bs x z_dim)
                    M.append(mub.cpu().detach().numpy())
                M = np.concatenate(M, 0)
                                
                # estimate sample var and mean of latent points for each dim
                if M.shape[0]>=2:
                    vars_per_factor[i,:] = np.var(M, 0)
                else:  # not enough samples to estimate variance
                    vars_per_factor[i,:] = 0.0                
                
                # true factor id (will become the class label)
                true_factor_ids[i] = fac_id

                i += 1
                
        # 3) evaluate majority vote classification accuracy
 
        # inputs in the paired data for classification
        smallest_var_dims = np.argmin(
          vars_per_factor / (vars_agn_factor + 1e-20), axis=1 )
    
        # contingency table
        C = np.zeros([self.z_dim,len(factor_ids)])
        for i in range(num_pairs):
            C[ smallest_var_dims[i], true_factor_ids[i] ] += 1
        
        num_errs = 0  # # misclassifying errors of majority vote classifier
        for k in range(self.z_dim):
            num_errs += np.sum(C[k,:]) - np.max(C[k,:])
        
        metric1 = (num_pairs - num_errs) / num_pairs  # metric = accuracy
        
        self.set_mode(train=True)

        return metric1, C
    
    
    ####
    def eval_disentangle_metric2(self):
        
        # some hyperparams
        num_pairs = 800  # # data pairs (d,y) for majority vote classification
        bs = 50  # batch size
        nsamps_per_factor = 100  # samples per factor
        nsamps_agn_factor = 5000  # factor-agnostic samples     
        
        self.set_mode(train=False)
        
        # 1) estimate variances of latent points factor agnostic
        
        dl = DataLoader( 
          self.data_loader.dataset, batch_size=bs,
          shuffle=True, num_workers=self.args.num_workers, pin_memory=True )
        iterator = iter(dl)
        
        M = []
        for ib in range(int(nsamps_agn_factor/bs)):
            
            # sample a mini-batch
            Xb, _ = next(iterator)  # (bs x C x H x W)
            if self.use_cuda:
                Xb = Xb.cuda()
            
            # enc(Xb)
            mub, _, _ = self.encoder(Xb)  # (bs x z_dim)

            M.append(mub.cpu().detach().numpy())
            
        M = np.concatenate(M, 0)
        
        # estimate sample vairance and mean of latent points for each dim
        vars_agn_factor = np.var(M, 0)

        # 2) estimatet dim-wise vars of latent points with "one factor varied"

        factor_ids = range(0, len(self.latent_sizes))  # true factor ids
        vars_per_factor = np.zeros([num_pairs,self.z_dim])  
        true_factor_ids = np.zeros(num_pairs, np.int)  # true factor ids

        # prepare data pairs for majority-vote classification
        i = 0
        for j in factor_ids:  # for each factor

            # repeat num_paris/num_factors times
            for r in range(int(num_pairs/len(factor_ids))):
                                
                # randomly choose true factors (id's and class values) to fix
                fac_ids = list(np.setdiff1d(factor_ids,j))
                fac_classes = \
                  [ np.random.randint(self.latent_sizes[k]) for k in fac_ids ]

                # randomly select images (with the other factors fixed)
                if len(fac_ids)>1:
                    indices = np.where( 
                      np.sum(self.latent_classes[:,fac_ids]==fac_classes,1)
                      == len(fac_ids) 
                    )[0]
                else:
                    indices = np.where(
                      self.latent_classes[:,fac_ids]==fac_classes 
                    )[0]
                np.random.shuffle(indices)
                idx = indices[:nsamps_per_factor]
                M = []
                for ib in range(int(nsamps_per_factor/bs)):
                    Xb, _ = dl.dataset[ idx[(ib*bs):(ib+1)*bs] ]
                    if Xb.shape[0]<1:  # no more samples
                        continue;
                    if self.use_cuda:
                        Xb = Xb.cuda()
                    mub, _, _ = self.encoder(Xb)  # (bs x z_dim)
                    M.append(mub.cpu().detach().numpy())
                M = np.concatenate(M, 0)
                
                # estimate sample var and mean of latent points for each dim
                if M.shape[0]>=2:
                    vars_per_factor[i,:] = np.var(M, 0)
                else:  # not enough samples to estimate variance
                    vars_per_factor[i,:] = 0.0
                    
                # true factor id (will become the class label)
                true_factor_ids[i] = j

                i += 1
                
        # 3) evaluate majority vote classification accuracy
            
        # inputs in the paired data for classification
        largest_var_dims = np.argmax(
          vars_per_factor / (vars_agn_factor + 1e-20), axis=1 )
    
        # contingency table
        C = np.zeros([self.z_dim,len(factor_ids)])
        for i in range(num_pairs):
            C[ largest_var_dims[i], true_factor_ids[i] ] += 1
        
        num_errs = 0  # # misclassifying errors of majority vote classifier
        for k in range(self.z_dim):
            num_errs += np.sum(C[k,:]) - np.max(C[k,:])
    
        metric2 = (num_pairs - num_errs) / num_pairs  # metric = accuracy    
        
        self.set_mode(train=True)

        return metric2, C
    

    ####
    def save_recon(self, iters, true_images, recon_images):
        
        # make a merge of true and recon, eg, 
        #   merged[0,...] = true[0,...], 
        #   merged[1,...] = recon[0,...], 
        #   merged[2,...] = true[1,...], 
        #   merged[3,...] = recon[1,...], ...
        
        n = true_images.shape[0]
        perm = torch.arange(0,2*n).view(2,n).transpose(1,0)
        perm = perm.contiguous().view(-1)
        merged = torch.cat([true_images, recon_images], dim=0)
        merged = merged[perm,:].cpu()
              
        # save the results as image
        fname = os.path.join(self.output_dir_recon, 'recon_%s.jpg' % iters) 
        mkdirs(self.output_dir_recon)
        save_image( 
          tensor=merged, filename=fname, nrow=2*int(np.sqrt(n)), 
          pad_value=1
        )


    ####
    def save_synth(self, iters, howmany=100):
        
        self.set_mode(train=False)

        decoder = self.decoder
        
        Z = torch.randn(howmany, self.z_dim)
        if self.use_cuda:
            Z = Z.cuda()
    
        # do synthesis 
        X = torch.sigmoid(decoder(Z)).data.cpu()
    
        # save the results as image
        fname = os.path.join(self.output_dir_synth, 'synth_%s.jpg' % iters)
        mkdirs(self.output_dir_synth)
        save_image( 
          tensor=X, filename=fname, nrow=int(np.sqrt(howmany)), 
          pad_value=1
        )

        self.set_mode(train=True)
        
    
    ####
    def save_traverse(self, iters, limb=-3, limu=3, inter=2/3, loc=-1):
        
        self.set_mode(train=False)

        encoder = self.encoder
        decoder = self.decoder
        interpolation = torch.arange(limb, limu+0.001, inter)

        i = np.random.randint(self.N)
        random_img = self.data_loader.dataset.__getitem__(i)[0]
        if self.use_cuda:
            random_img = random_img.cuda()
        random_img = random_img.unsqueeze(0)
        random_img_zmu, _, _ = encoder(random_img)

        if self.dataset.lower() == 'dsprites':
            
            fixed_idx1 = 87040  # square
            fixed_idx2 = 332800  # ellipse
            fixed_idx3 = 578560  # heart

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            if self.use_cuda:
                fixed_img1 = fixed_img1.cuda()
            fixed_img1 = fixed_img1.unsqueeze(0)
            fixed_img_zmu1, _, _ = encoder(fixed_img1)

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            if self.use_cuda:
                fixed_img2 = fixed_img2.cuda()
            fixed_img2 = fixed_img2.unsqueeze(0)
            fixed_img_zmu2, _, _ = encoder(fixed_img2)

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            if self.use_cuda:
                fixed_img3 = fixed_img3.cuda()
            fixed_img3 = fixed_img3.unsqueeze(0)
            fixed_img_zmu3, _, _ = encoder(fixed_img3)
            
            IMG = {'fixed_square':fixed_img1, 'fixed_ellipse':fixed_img2,
                 'fixed_heart':fixed_img3, 'random_img':random_img}

            Z = {'fixed_square':fixed_img_zmu1, 'fixed_ellipse':fixed_img_zmu2,
                 'fixed_heart':fixed_img_zmu3, 'random_img':random_img_zmu}
            
        elif self.dataset.lower() == 'oval_dsprites':
            
            fixed_idx1 = 87040  # oval1
            fixed_idx2 = 220045  # oval2
            fixed_idx3 = 178560  # oval3

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            if self.use_cuda:
                fixed_img1 = fixed_img1.cuda()
            fixed_img1 = fixed_img1.unsqueeze(0)
            fixed_img_zmu1, _, _ = encoder(fixed_img1)

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            if self.use_cuda:
                fixed_img2 = fixed_img2.cuda()
            fixed_img2 = fixed_img2.unsqueeze(0)
            fixed_img_zmu2, _, _ = encoder(fixed_img2)

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            if self.use_cuda:
                fixed_img3 = fixed_img3.cuda()
            fixed_img3 = fixed_img3.unsqueeze(0)
            fixed_img_zmu3, _, _ = encoder(fixed_img3)
            
            IMG = {'fixed1':fixed_img1, 'fixed2':fixed_img2,
                 'fixed3':fixed_img3, 'random_img':random_img}

            Z = {'fixed1':fixed_img_zmu1, 'fixed2':fixed_img_zmu2,
                 'fixed3':fixed_img_zmu3, 'random_img':random_img_zmu}
            
        elif self.dataset.lower() == '3dfaces':
            
            fixed_idx1 = 6245  
            fixed_idx2 = 10205  
            fixed_idx3 = 68560  

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            if self.use_cuda:
                fixed_img1 = fixed_img1.cuda()
            fixed_img1 = fixed_img1.unsqueeze(0)
            fixed_img_zmu1, _, _ = encoder(fixed_img1)

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            if self.use_cuda:
                fixed_img2 = fixed_img2.cuda()
            fixed_img2 = fixed_img2.unsqueeze(0)
            fixed_img_zmu2, _, _ = encoder(fixed_img2)

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            if self.use_cuda:
                fixed_img3 = fixed_img3.cuda()
            fixed_img3 = fixed_img3.unsqueeze(0)
            fixed_img_zmu3, _, _ = encoder(fixed_img3)
            
            IMG = {'fixed1':fixed_img1, 'fixed2':fixed_img2,
                 'fixed3':fixed_img3, 'random_img':random_img}

            Z = {'fixed1':fixed_img_zmu1, 'fixed2':fixed_img_zmu2,
                 'fixed3':fixed_img_zmu3, 'random_img':random_img_zmu}
            
        elif self.dataset.lower() == 'celeba':
            
            fixed_idx1 = 191281
            fixed_idx2 = 143307
            fixed_idx3 = 101535

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            if self.use_cuda:
                fixed_img1 = fixed_img1.cuda()
            fixed_img1 = fixed_img1.unsqueeze(0)
            fixed_img_zmu1, _, _ = encoder(fixed_img1)

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            if self.use_cuda:
                fixed_img2 = fixed_img2.cuda()
            fixed_img2 = fixed_img2.unsqueeze(0)
            fixed_img_zmu2, _, _ = encoder(fixed_img2)

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            if self.use_cuda:
                fixed_img3 = fixed_img3.cuda()
            fixed_img3 = fixed_img3.unsqueeze(0)
            fixed_img_zmu3, _, _ = encoder(fixed_img3)
            
            IMG = {'fixed1':fixed_img1, 'fixed2':fixed_img2,
                 'fixed3':fixed_img3, 'random_img':random_img}

            Z = {'fixed1':fixed_img_zmu1, 'fixed2':fixed_img_zmu2,
                 'fixed3':fixed_img_zmu3, 'random_img':random_img_zmu}
            
        elif self.dataset.lower() == 'edinburgh_teapots':
            
            fixed_idx1 = 7040
            fixed_idx2 = 32800
            fixed_idx3 = 78560

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            if self.use_cuda:
                fixed_img1 = fixed_img1.cuda()
            fixed_img1 = fixed_img1.unsqueeze(0)
            fixed_img_zmu1, _, _ = encoder(fixed_img1)

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            if self.use_cuda:
                fixed_img2 = fixed_img2.cuda()
            fixed_img2 = fixed_img2.unsqueeze(0)
            fixed_img_zmu2, _, _ = encoder(fixed_img2)

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            if self.use_cuda:
                fixed_img3 = fixed_img3.cuda()
            fixed_img3 = fixed_img3.unsqueeze(0)
            fixed_img_zmu3, _, _ = encoder(fixed_img3)
            
            IMG = {'fixed1':fixed_img1, 'fixed2':fixed_img2,
                 'fixed3':fixed_img3, 'random_img':random_img}

            Z = {'fixed1':fixed_img_zmu1, 'fixed2':fixed_img_zmu2,
                 'fixed3':fixed_img_zmu3, 'random_img':random_img_zmu}

#        elif self.dataset.lower() == '3dchairs':
#            
#            fixed_idx1 = 40919 # 3DChairs/images/4682_image_052_p030_t232_r096.png
#            fixed_idx2 = 5172  # 3DChairs/images/14657_image_020_p020_t232_r096.png
#            fixed_idx3 = 22330 # 3DChairs/images/30099_image_052_p030_t232_r096.png
#
#            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
#            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
#            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]
#
#            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
#            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
#            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]
#
#            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
#            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
#            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]
#
#            Z = {'fixed_1':fixed_img_z1, 'fixed_2':fixed_img_z2,
#                 'fixed_3':fixed_img_z3, 'random':random_img_zmu}
#        
        else:
            
            raise NotImplementedError

        # do traversal and collect generated images 
        gifs = []
        for key in Z:
            z_ori = Z[key]
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:,row] = val
                    sample = torch.sigmoid(decoder(z)).data
                    gifs.append(sample)

        # save the generated files, also the animated gifs            
        out_dir = os.path.join(self.output_dir_trvsl, str(iters))
        mkdirs(self.output_dir_trvsl)
        mkdirs(out_dir)
        gifs = torch.cat(gifs)
        gifs = gifs.view( 
          len(Z), self.z_dim, len(interpolation), self.nc, 64, 64
        ).transpose(1,2)
        for i, key in enumerate(Z.keys()):            
            for j, val in enumerate(interpolation):
                I = torch.cat([IMG[key], gifs[i][j]], dim=0)
                save_image(
                  tensor=I.cpu(),
                  filename=os.path.join(out_dir, '%s_%03d.jpg' % (key,j)),
                  nrow=1+self.z_dim, pad_value=1 )
            # make animated gif
            grid2gif(
              out_dir, key, str(os.path.join(out_dir, key+'.gif')), delay=10
            )

        self.set_mode(train=True)
        

    ####
    def viz_init(self):
        
        self.viz.close(env=self.name+'/lines', win=self.win_id['DZ'])
        self.viz.close(env=self.name+'/lines', win=self.win_id['recon'])
        self.viz.close(env=self.name+'/lines', win=self.win_id['kl'])
        self.viz.close(env=self.name+'/lines', win=self.win_id['kl_alpha'])
        
        if self.eval_metrics:
            self.viz.close(env=self.name+'/lines', win=self.win_id['metrics'])
        

    ####
    def visualize_line(self):
        
        # prepare data to plot
        data = self.line_gather.data
        iters = torch.Tensor(data['iter'])
        recon = torch.Tensor(data['recon'])
        kl = torch.Tensor(data['kl'])
        kl_alpha = torch.Tensor(data['kl_alpha'])
        
        p_DZ = torch.Tensor(data['p_DZ'])
        p_DZ_perm = torch.Tensor(data['p_DZ_perm'])
        p_DZs = torch.stack([p_DZ, p_DZ_perm], -1)  # (#items x 2)

        self.viz.line(
          X=iters, Y=p_DZs, env=self.name+'/lines', win=self.win_id['DZ'],
          update='append',
          opts=dict(xlabel='iter', ylabel='D(z)', title='Discriminator-Z', 
                    legend=['D(z)','D(z_perm)',]) )
        
        self.viz.line(
          X=iters, Y=recon, env=self.name+'/lines', win=self.win_id['recon'],
          update='append',
          opts=dict(xlabel='iter', ylabel='recon loss', 
                    title='Reconstruction') )
        
        self.viz.line(
          X=iters, Y=kl, env=self.name+'/lines', win=self.win_id['kl'],
          update='append',
          opts=dict(xlabel='iter', 
                    ylabel='E_q(alpha)E_x[kl(q(z|x)||p(z|alpha)]', 
                    title='KL divergence') )
        
        self.viz.line(
          X=iters, Y=kl_alpha, env=self.name+'/lines', 
          win=self.win_id['kl_alpha'],
          update='append',
          opts=dict(xlabel='iter', ylabel='KL(q(alpha)||p(alpha)) / N', 
                    title='KL divergence on alpha') )
           

    ####
    def visualize_line_metrics(self, iters, metric1, metric2):
        
        # prepare data to plot
        iters = torch.tensor([iters], dtype=torch.int64).detach()
        metric1 = torch.tensor([metric1])
        metric2 = torch.tensor([metric2])
        metrics = torch.stack([metric1.detach(), metric2.detach()], -1)
        
        self.viz.line(
          X=iters, Y=metrics, env=self.name+'/lines', 
          win=self.win_id['metrics'], update='append',
          opts=dict( xlabel='iter', ylabel='metrics', 
            title='Disentanglement metrics', 
            legend=['metric1','metric2'] ) 
        )
 
    
    ####
    def set_mode(self, train=True):
        
        if train:
            self.encoder.train()
            self.decoder.train()
            self.D.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
            self.D.eval()
            

    ####
    def save_checkpoint(self, iteration):
        
        encoder_path = os.path.join( self.ckpt_dir, 
          'iter_%s_encoder.pt' % iteration )
        decoder_path = os.path.join( self.ckpt_dir, 
          'iter_%s_decoder.pt' % iteration )
        prior_alpha_path = os.path.join( self.ckpt_dir, 
          'iter_%s_prior_alpha.pt' % iteration )
        post_alpha_path = os.path.join( self.ckpt_dir, 
          'iter_%s_post_alpha.pt' % iteration )
        D_path = os.path.join( self.ckpt_dir, 
          'iter_%s_D.pt' % iteration )
        
        mkdirs(self.ckpt_dir)
        
        torch.save(self.encoder, encoder_path)
        torch.save(self.decoder, decoder_path)
        torch.save(self.prior_alpha, prior_alpha_path)
        torch.save(self.post_alpha, post_alpha_path)
        torch.save(self.D, D_path)
    

    ####
    def load_checkpoint(self):
        
        encoder_path = os.path.join( self.ckpt_dir, 
          'iter_%s_encoder.pt' % self.ckpt_load_iter )
        decoder_path = os.path.join( self.ckpt_dir, 
          'iter_%s_decoder.pt' % self.ckpt_load_iter )
        prior_alpha_path = os.path.join( self.ckpt_dir, 
          'iter_%s_prior_alpha.pt' % self.ckpt_load_iter )
        post_alpha_path = os.path.join( self.ckpt_dir, 
          'iter_%s_post_alpha.pt' % self.ckpt_load_iter )
        D_path = os.path.join( self.ckpt_dir, 
          'iter_%s_D.pt' % self.ckpt_load_iter )
        
        if self.use_cuda:
            self.encoder = torch.load(encoder_path)
            self.decoder = torch.load(decoder_path)
            self.prior_alpha = torch.load(prior_alpha_path)
            self.post_alpha = torch.load(post_alpha_path)
            self.D = torch.load(D_path)
        else:
            self.encoder = torch.load(encoder_path, map_location='cpu')
            self.decoder = torch.load(decoder_path, map_location='cpu')
            self.prior_alpha = torch.load(prior_alpha_path, map_location='cpu')
            self.post_alpha = torch.load(post_alpha_path, map_location='cpu')
            self.D = torch.load(D_path, map_location='cpu')

            

