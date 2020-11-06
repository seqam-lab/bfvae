import os, random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

###############################################################################
    
class CustomImageFolder(ImageFolder):
    
    '''
    Dataset when it is inefficient to load all data items to memory
    '''
    
    ####
    def __init__(self, root, transform=None):
        
        '''
        root = directory that contains data (images)
        '''
        
        super(CustomImageFolder, self).__init__(root, transform)


    ####
    def __getitem__(self, i):

        img = self.loader(self.imgs[i][0])
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, i


###############################################################################
        
class CustomTensorDataset(Dataset):
    
    '''
    Dataset when it is possible and efficient to load all data items to memory
    '''
    
    ####
    def __init__(self, data_tensor, transform=None):
        
        '''
        data_tensor = actual data items; (N x C x H x W)
        '''
        
        self.data_tensor = data_tensor
        self.transform = transform

    ####
    def __getitem__(self, i):
        
        img = self.data_tensor[i]
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, i

    ####
    def __len__(self):
        
        return self.data_tensor.size(0)


###############################################################################
        
def create_dataloader(args, williams=None):
    
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    assert image_size == 64, 'currently only image size of 64 is supported'

    transform = transforms.Compose([
      transforms.Resize((image_size, image_size)),
      transforms.ToTensor(),])

    drop_last = True
    
    if name.lower() == 'celeba':
        
        root = os.path.join(dset_dir, 'celeba') #/img_align_crop_64x64')
        train_kwargs = {'root': root, 'transform': transform}
        dset = CustomImageFolder
    
    elif name.lower() == '3dchairs':
    
        root = os.path.join(dset_dir, '3DChairs')
        train_kwargs = {'root': root, 'transform': transform}
        dset = CustomImageFolder
    
    elif name.lower() == 'dsprites':
    
        root = os.path.join( dset_dir, 'dsprites-dataset/imgs.npy' )
        imgs = np.load(root, encoding='latin1')
        data = torch.from_numpy(imgs).unsqueeze(1).float()
        train_kwargs = {'data_tensor': data}
        dset = CustomTensorDataset
    
    elif name.lower() == 'oval_dsprites':
    
        latent_classes = np.load( os.path.join( dset_dir, 
           'dsprites-dataset/latents_classes.npy'), encoding='latin1' )
        idx = np.where(latent_classes[:,1]==1)[0]  # "oval" shape only
        root = os.path.join( dset_dir, 'dsprites-dataset/imgs.npy' )
        imgs = np.load(root, encoding='latin1')
        imgs = imgs[idx]
        data = torch.from_numpy(imgs).unsqueeze(1).float()
        train_kwargs = {'data_tensor': data}
        dset = CustomTensorDataset
        
    elif name.lower() == '3dfaces':
    
        root = os.path.join( dset_dir, 
                            '3d_faces/rtqichen/basel_face_renders.pth' )
        data = torch.load(root).float().div(255)  # (50x21x11x11x64x64)
        data = data.view(-1,64,64).unsqueeze(1)  # (127050 x 1 x 64 x 64)
        train_kwargs = {'data_tensor': data}
        dset = CustomTensorDataset
        
    elif name.lower() == 'edinburgh_teapots':
        
#        roots = [ os.path.join( dset_dir, 'edinburgh_teapots/imgs_tr.npz' ), \
#            os.path.join( dset_dir, 'edinburgh_teapots/imgs_va.npz' ), \
#            os.path.join( dset_dir, 'edinburgh_teapots/imgs_te.npz' ) ]
#
#        imgs = np.zeros((0,64,64,3))
#        for root in roots:
#            imgs = np.concatenate((imgs, np.load(root)['data']), axis=0)
#            
#        data = torch.from_numpy(imgs).permute(0,3,1,2).float()
#        #data = 2.0 * (data/255.0 - 0.5)  # [-1,+1]-valued
#        
#        torch.save( data, 
#                    os.path.join( dset_dir, 'edinburgh_teapots/data_all.pt') )
        
        # NOTE: the above is already done
        
        data = torch.load( 
            os.path.join( dset_dir, 'edinburgh_teapots/data_all.pt') )
        
        data = data/255.0  # [0,1]-valued
        train_kwargs = {'data_tensor': data}
        dset = CustomTensorDataset
        # drop_last = False
        
    else:
        raise NotImplementedError

    dataset = dset(**train_kwargs)
    
    dataloader = DataLoader( dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, drop_last=drop_last )

    return dataloader
