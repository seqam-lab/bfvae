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
        self.gamma = args.gamma
        
        # what to do in this test
        self.num_recon = args.num_recon
        self.num_synth = args.num_synth
        self.num_trvsl = args.num_trvsl
        self.losses = args.losses
        self.num_eval_metric1 = args.num_eval_metric1
        self.num_eval_metric2 = args.num_eval_metric2
        
        # checkpoints
        self.ckpt_dir = os.path.join("ckpts", self.name)
        
        # create dirs: "records", "ckpts", "outputs" (if not exist)
        mkdirs("records");  mkdirs("outputs")
            
        # records
        self.record_file = 'records/%s.txt' % ("test_" + self.name)

        # outputs
        self.output_dir_recon = os.path.join( "outputs", 
                                              "test_" + self.name + '_recon' )
        self.output_dir_synth = os.path.join( "outputs",  
                                              "test_" + self.name + '_synth' )
        self.output_dir_trvsl = os.path.join( "outputs",  
                                              "test_" + self.name + '_trvsl' )
        
        # load a previously saved model
        self.ckpt_load_iter = args.ckpt_load_iter
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
        
        # iterator from dataloader
        iterator = iter(self.data_loader)
        iter_per_epoch = len(iterator)
        
        #----#
        
        # image synthesis
        if self.num_trvsl > 0:
            prn_str = 'Start doing image synthesis...'
            print(prn_str)
            self.dump_to_record(prn_str)
            for ii in range(self.num_synth):
                self.save_synth( str(self.ckpt_load_iter) + '_' + str(ii), 
                             howmany=100 )
        
        # latent traversal
        if self.num_trvsl > 0:
            prn_str = 'Start doing latent traversal...'
            print(prn_str)
            self.dump_to_record(prn_str)
            # self.save_traverse_new( self.ckpt_load_iter, self.num_trvsl, 
            #                         limb=-4, limu=4, inter=0.1 )
            self.save_traverse_new( self.ckpt_load_iter, self.num_trvsl, 
                                    limb=-16, limu=16, inter=0.2 )
        
        # metric1
        if self.num_eval_metric1 > 0:
            prn_str = 'Start evaluating metric1...'
            print(prn_str)
            self.dump_to_record(prn_str)
            #
            metric1s = np.zeros(self.num_eval_metric1)
            C1s = np.zeros([ self.num_eval_metric1, 
                             self.z_dim, len(self.latent_sizes) ])
            for ii in range(self.num_eval_metric1):
                metric1s[ii], C1s[ii] = self.eval_disentangle_metric1()
                prn_str = 'eval metric1: %d/%d done' % \
                          (ii+1, self.num_eval_metric1)
                print(prn_str)
                self.dump_to_record(prn_str)
            #
            prn_str = 'metric1:\n' + str(metric1s)
            prn_str += '\nC1:\n' + str(C1s)
            print(prn_str)
            self.dump_to_record(prn_str)
            
        
        # metric2
        if self.num_eval_metric2 > 0:
            prn_str = 'Start evaluating metric2...'
            print(prn_str)
            self.dump_to_record(prn_str)
            #
            metric2s = np.zeros(self.num_eval_metric2)
            C2s = np.zeros([ self.num_eval_metric2, 
                             self.z_dim, len(self.latent_sizes) ])
            for ii in range(self.num_eval_metric2):
                metric2s[ii], C2s[ii] = self.eval_disentangle_metric2()
                prn_str = 'eval metric2: %d/%d done' % \
                          (ii+1, self.num_eval_metric2)
                print(prn_str)
                self.dump_to_record(prn_str)
            #
            prn_str = 'metric2:\n' + str(metric2s)
            prn_str += '\nC2:\n' + str(C2s)
            print(prn_str)
            self.dump_to_record(prn_str)
            
            
        #----#
        
        if self.losses or self.num_recon>0:
            num_adds = 0
            loss_kl_inds = np.zeros(self.z_dim)
            losses = {}
            losses['vae_loss'] = 0.0
            losses['dis_loss'] = 0.0
            losses['recon'] = 0.0
            losses['kl'] = 0.0
            losses['tc'] = 0.0
            losses['kl_alpha'] = 0.0
            cntdn = self.num_recon
        else:
            return
        
        prn_str = 'Start going through the entire data...'
        print(prn_str)
        self.dump_to_record(prn_str)
            
        for iteration in range(1, 100000000):

            # reset data iterators for each epoch
            if iteration % iter_per_epoch == 0:
                
                # inidividual kls
                loss_kl_inds /= num_adds
                prn_str = "Individual kl's:\n" + str(loss_kl_inds)
                print(prn_str)
                self.dump_to_record(prn_str)
                
                # losses
                losses['vae_loss'] /= num_adds
                losses['dis_loss'] /= num_adds
                losses['recon'] /= num_adds
                losses['kl'] /= num_adds
                losses['tc'] /= num_adds
                losses['kl_alpha'] /= num_adds
                prn_str = "losses:\n" + str(losses)
                print(prn_str)
                self.dump_to_record(prn_str)
            
                break
            
            
            with torch.no_grad():
            
                # sample a mini-batch
                X, ids = next(iterator)  # (n x C x H x W)
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
                
                # kl loss on alpha
                kls_alpha = ( \
                    (ah-a)*ah.digamma() - ah.lgamma() + a.lgamma() + \
                    a*(bh.log()-b.log()) + (ah/bh)*(b-bh) )  # z_dim-dim
                loss_kl_alpha = kls_alpha.sum() / self.N
                
                # total loss for vae
                vae_loss = loss_recon + loss_kl + self.gamma*loss_tc + \
                           loss_kl_alpha
                
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
                
                if self.losses:
                    
                    loss_kl_ind = 0.5 * ( \
                        (ah/bh)*(mu**2+std**2) - 1.0 + \
                        bh.log() - ah.digamma() - logvar ).mean(0) 
                    loss_kl_inds += loss_kl_ind.cpu().detach().numpy()
                    #
                    losses['vae_loss'] += vae_loss.item()
                    losses['dis_loss'] += dis_loss.item()
                    losses['recon'] += loss_recon.item()
                    losses['kl'] += loss_kl.item()
                    losses['tc'] += loss_tc.item()
                    losses['kl_alpha'] += loss_kl_alpha.item()
                    #
                    num_adds += 1


                # print the losses
                if iteration % 100 == 0:
                    prn_str = ( '[%d/%d] vae_loss: %.3f | dis_loss: %.3f\n' + \
                      '    (recon: %.3f, kl: %.3f, tc: %.3f, kl_alpha: %.3f)' \
                      ) % \
                        ( iteration, iter_per_epoch, 
                          vae_loss.item(), dis_loss.item(), 
                          loss_recon.item(), loss_kl.item(), loss_tc.item(),
                          loss_kl_alpha.item() )
                    prn_str += '\n    a = {}'.format(
                        a.detach().cpu().numpy().round(2) )
                    prn_str += '\n    b = {}'.format(
                        b.detach().cpu().numpy().round(2) )
                    prn_str += '\n    ah = {}'.format(
                        ah.detach().cpu().numpy().round(2) )
                    prn_str += '\n    bh = {}'.format(
                        bh.detach().cpu().numpy().round(2) )
                    print(prn_str)
                    self.dump_to_record(prn_str)
                    
                # save reconstructed images
                if cntdn>0:
                    self.save_recon(iteration, X, torch.sigmoid(X_recon).data)
                    cntdn -= 1
                    if cntdn==0:
                        prn_str = 'Completed image reconstruction'
                        print(prn_str)
                        self.dump_to_record(prn_str)            
                        if not self.losses:
                            break


    ####
    def eval_disentangle_metric1(self):
        
        # some hyperparams
        num_pairs = 800  # # data pairs (d,y) for majority vote classification
        bs = 50  # batch size
        nsamps_per_factor = 100  # samples per factor
        nsamps_agn_factor = 5000  # factor-agnostic samples
        
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

        return metric1, C
    
    
    ####
    def eval_disentangle_metric2(self):
        
        # some hyperparams
        num_pairs = 800  # # data pairs (d,y) for majority vote classification
        bs = 50  # batch size
        nsamps_per_factor = 100  # samples per factor
        nsamps_agn_factor = 5000  # factor-agnostic samples     
        
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
        
    
    ####
    def save_traverse_new( self, iters, num_reps, 
                           limb=-3, limu=3, inter=2/3, loc=-1 ):
        
        encoder = self.encoder
        decoder = self.decoder
        interpolation = torch.arange(limb, limu+0.001, inter)

        np.random.seed(123)
        rii = np.random.randint(self.N, size=num_reps)
        #--#
        prn_str = '(TRAVERSAL) random image IDs = {}'.format(rii)
        print(prn_str)
        self.dump_to_record(prn_str)
        #--#
        random_imgs = [0]*num_reps
        random_imgs_zmu = [0]*num_reps
        for i, i2 in enumerate(rii):
            random_imgs[i] = self.data_loader.dataset.__getitem__(i2)[0]
            if self.use_cuda:
                random_imgs[i] = random_imgs[i].cuda()
            random_imgs[i] = random_imgs[i].unsqueeze(0)
            random_imgs_zmu[i], _, _ = encoder(random_imgs[i])

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
            
            IMG = { 'fixed_square': fixed_img1, 'fixed_ellipse': fixed_img2,
                    'fixed_heart': fixed_img3 }
            for i in range(num_reps):
                IMG['random_img'+str(i)] = random_imgs[i]

            Z = { 'fixed_square': fixed_img_zmu1, 
                  'fixed_ellipse': fixed_img_zmu2,
                  'fixed_heart': fixed_img_zmu3 }
            for i in range(num_reps):
                Z['random_img'+str(i)] = random_imgs_zmu[i]
            
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
            
            IMG = { 'fixed1': fixed_img1, 'fixed2': fixed_img2,
                    'fixed3': fixed_img3 }           
            for i in range(num_reps):
                IMG['random_img'+str(i)] = random_imgs[i]

            Z = { 'fixed1': fixed_img_zmu1, 'fixed2': fixed_img_zmu2,
                  'fixed3': fixed_img_zmu3}
            for i in range(num_reps):
                Z['random_img'+str(i)] = random_imgs_zmu[i]
                
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
            
            IMG = { 'fixed1': fixed_img1, 'fixed2': fixed_img2,
                    'fixed3': fixed_img3 }           
            for i in range(num_reps):
                IMG['random_img'+str(i)] = random_imgs[i]

            Z = { 'fixed1': fixed_img_zmu1, 'fixed2': fixed_img_zmu2,
                  'fixed3': fixed_img_zmu3}
            for i in range(num_reps):
                Z['random_img'+str(i)] = random_imgs_zmu[i]
            
        elif self.dataset.lower() == 'celeba':
            
            fixed_idx1 = 191282
            fixed_idx2 = 143308
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
            
            IMG = { 'fixed1': fixed_img1, 'fixed2': fixed_img2,
                    'fixed3': fixed_img3 }           
            for i in range(num_reps):
                IMG['random_img'+str(i)] = random_imgs[i]

            Z = { 'fixed1': fixed_img_zmu1, 'fixed2': fixed_img_zmu2,
                  'fixed3': fixed_img_zmu3}
            for i in range(num_reps):
                Z['random_img'+str(i)] = random_imgs_zmu[i]

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
            
            IMG = { 'fixed1': fixed_img1, 'fixed2': fixed_img2,
                    'fixed3': fixed_img3 }           
            for i in range(num_reps):
                IMG['random_img'+str(i)] = random_imgs[i]

            Z = { 'fixed1': fixed_img_zmu1, 'fixed2': fixed_img_zmu2,
                  'fixed3': fixed_img_zmu3}
            for i in range(num_reps):
                Z['random_img'+str(i)] = random_imgs_zmu[i]
            
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


    ####
    def dump_to_record(self, prn_str):
        
        record = open(self.record_file, 'a')
        record.write('%s\n' % (prn_str,))
        record.close()
        
        
        