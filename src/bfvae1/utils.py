import os
import imageio
import argparse
import subprocess

###############################################################################

class DataGather(object):

    '''
    create (array)lists, one for each category, eg, 
      self.data['recon'] = [2.3, 1.5, 0.8, ...],
      self.data['kl'] = [0.3, 1.8, 2.2, ...], 
      self.data['acc'] = [0.3, 0.4, 0.5, ...], ...
    '''
    
    def __init__(self, *args):
        self.keys = args
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return {arg:[] for arg in self.keys}

    def insert(self, **kwargs):
        for key in kwargs.keys():
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()

###############################################################################

def str2bool(v):
    
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#-----------------------------------------------------------------------------#

def grid2gif(img_dir, key, out_gif, delay=100, duration=0.1):

    '''
    make (moving) GIF from images
    '''
    
    if True:  #os.name=='nt':
        
        fnames = [ \
          str(os.path.join(img_dir, f)) for f in os.listdir(img_dir) \
            if (key in f) and ('jpg' in f) ]
        
        fnames.sort()
        
        images = []
        for filename in fnames:
            images.append(imageio.imread(filename))
        
        imageio.mimsave(out_gif, images, duration=duration)
        
    else:  # os.name=='posix'
        
        img_str = str(os.path.join(img_dir, key+'*.jpg'))
        cmd = 'convert -delay %s -loop 0 %s %s' % (delay, img_str, out_gif)
        subprocess.call(cmd, shell=True)

#-----------------------------------------------------------------------------#

def mkdirs(path):

    if not os.path.exists(path):
        os.makedirs(path)
