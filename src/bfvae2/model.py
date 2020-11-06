import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

###############################################################################

def kaiming_init(m):
    
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

#-----------------------------------------------------------------------------#
            
def normal_init(m):
    
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
    
###############################################################################

class PriorAlphaParams(nn.Module):
    
    def __init__(self, z_dim):
        
        super(PriorAlphaParams, self).__init__()
        
        self.epsilon = 0.001    
        self.rvlogit = nn.Parameter(0.001*torch.randn(z_dim))
        
    def forward(self):
        
        rv = torch.sigmoid(self.rvlogit)
        
        a = (1.0 + 2*self.epsilon) / (rv + self.epsilon)
        b = a - 1.0
        
        return a, b, self.rvlogit, rv

###############################################################################

class PostAlphaParams(nn.Module):
    
    def __init__(self, z_dim):
        
        super(PostAlphaParams, self).__init__()
        
        self.logah = nn.Parameter(0.01*torch.randn(z_dim))
        self.logbh = nn.Parameter(0.01*torch.randn(z_dim))
        
    def forward(self):
        
        ah = torch.exp(self.logah)
        bh = torch.exp(self.logbh)
        
        return ah, bh

###############################################################################
        
class Discriminator(nn.Module):
    
    '''
    returns (n x 2): Let D1 = 1st column, D2 = 2nd column, then the meaning is
      D(z) (\in [0,1]) = exp(D1) / ( exp(D1) + exp(D2) )
      
      so, it follows: log( D(z) / (1-D(z)) ) = D1 - D2
    '''
    
    ####
    def __init__(self, z_dim):
        
        super(Discriminator, self).__init__()
        
        self.z_dim = z_dim
        
        self.net = nn.Sequential(
          nn.Linear(z_dim, 1000), nn.LeakyReLU(0.2, True),
          nn.Linear(1000, 1000),  nn.LeakyReLU(0.2, True),
          nn.Linear(1000, 1000),  nn.LeakyReLU(0.2, True),
          nn.Linear(1000, 1000),  nn.LeakyReLU(0.2, True),
          nn.Linear(1000, 1000),  nn.LeakyReLU(0.2, True),
          nn.Linear(1000, 2),
        )
        
        self.weight_init()


    ####
    def weight_init(self, mode='normal'):

        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)


    ####
    def forward(self, z):
        
        return self.net(z)
    

#-----------------------------------------------------------------------------#
        
class Encoder1(nn.Module):
    
    '''
    encoder architecture for the "dsprites" data
    '''
    
    ####
    def __init__(self, z_dim=10):
        
        super(Encoder1, self).__init__()
        
        self.z_dim = z_dim

        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)
        self.fc5 = nn.Linear(64*4*4, 128)
        self.fc6 = nn.Linear(128, 2*z_dim) 
        
        # initialize parameters
        self.weight_init()


    ####
    def weight_init(self, mode='normal'):
        
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for m in self._modules:
            initializer(self._modules[m])


    ####
    def forward(self, x):
        
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc5(out))
        stats = self.fc6(out)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        std = torch.sqrt(torch.exp(logvar))
        
        return mu, std, logvar


#-----------------------------------------------------------------------------#
        
class Decoder1(nn.Module):
    
    '''
    decoder architecture for the "dsprites" data
    '''
    
    ####
    def __init__(self, z_dim=10):
        
        super(Decoder1, self).__init__()
        
        self.z_dim = z_dim
        
        self.fc1 = nn.Linear(z_dim, 128)
        self.fc2 = nn.Linear(128, 4*4*64)
        self.deconv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.deconv6 = nn.ConvTranspose2d(32, 1, 4, 2, 1)
       
        # initialize parameters
        self.weight_init()


    ####
    def weight_init(self, mode='normal'):
        
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for m in self._modules:
            initializer(self._modules[m])


    ####
    def forward(self, z):
        
        out = F.relu(self.fc1(z))
        out = F.relu(self.fc2(out))
        out = out.view(out.size(0), 64, 4, 4)
        out = F.relu(self.deconv3(out))
        out = F.relu(self.deconv4(out))
        out = F.relu(self.deconv5(out))
        x_recon = self.deconv6(out)
            
        return x_recon


#-----------------------------------------------------------------------------#
        
class Encoder3(nn.Module):
    
    '''
    encoder architecture for the "3dfaces" data
    '''
    
    ####
    def __init__(self, z_dim=10):
        
        super(Encoder3, self).__init__()
        
        self.z_dim = z_dim

        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)
        self.fc5 = nn.Linear(64*4*4, 256)
        self.fc6 = nn.Linear(256, 2*z_dim) 
        
        # initialize parameters
        self.weight_init()


    ####
    def weight_init(self, mode='normal'):
        
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for m in self._modules:
            initializer(self._modules[m])


    ####
    def forward(self, x):
        
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc5(out))
        stats = self.fc6(out)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        std = torch.sqrt(torch.exp(logvar))
        
        return mu, std, logvar


#-----------------------------------------------------------------------------#
        
class Decoder3(nn.Module):
    
    '''
    decoder architecture for the "3dfaces" data
    '''
    
    ####
    def __init__(self, z_dim=10):
        
        super(Decoder3, self).__init__()
        
        self.z_dim = z_dim
        
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 4*4*64)
        self.deconv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.deconv6 = nn.ConvTranspose2d(32, 1, 4, 2, 1)
       
        # initialize parameters
        self.weight_init()


    ####
    def weight_init(self, mode='normal'):
        
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for m in self._modules:
            initializer(self._modules[m])


    ####
    def forward(self, z):
        
        out = F.relu(self.fc1(z))
        out = F.relu(self.fc2(out))
        out = out.view(out.size(0), 64, 4, 4)
        out = F.relu(self.deconv3(out))
        out = F.relu(self.deconv4(out))
        out = F.relu(self.deconv5(out))
        x_recon = self.deconv6(out)
            
        return x_recon
    
    
#-----------------------------------------------------------------------------#
        
class Encoder4(nn.Module):
    
    '''
    encoder architecture for the "celeba" data
    '''
    
    ####
    def __init__(self, z_dim=10):
        
        super(Encoder4, self).__init__()
        
        self.z_dim = z_dim

        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)
        self.fc5 = nn.Linear(64*4*4, 256)
        self.fc6 = nn.Linear(256, 2*z_dim) 
        
        # initialize parameters
        self.weight_init()


    ####
    def weight_init(self, mode='normal'):
        
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for m in self._modules:
            initializer(self._modules[m])


    ####
    def forward(self, x):
        
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc5(out))
        stats = self.fc6(out)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        std = torch.sqrt(torch.exp(logvar))
        
        return mu, std, logvar


#-----------------------------------------------------------------------------#
        
class Decoder4(nn.Module):
    
    '''
    decoder architecture for the "celeba" data
    '''
    
    ####
    def __init__(self, z_dim=10):
        
        super(Decoder4, self).__init__()
        
        self.z_dim = z_dim
        
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 4*4*64)
        self.deconv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.deconv6 = nn.ConvTranspose2d(32, 3, 4, 2, 1)
       
        # initialize parameters
        self.weight_init()


    ####
    def weight_init(self, mode='normal'):
        
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for m in self._modules:
            initializer(self._modules[m])


    ####
    def forward(self, z):
        
        out = F.relu(self.fc1(z))
        out = F.relu(self.fc2(out))
        out = out.view(out.size(0), 64, 4, 4)
        out = F.relu(self.deconv3(out))
        out = F.relu(self.deconv4(out))
        out = F.relu(self.deconv5(out))
        x_recon = self.deconv6(out)
            
        return x_recon
    

###############################################################################

#
# resnet enc/dec models (for "teapot" dataset)
#

#-----------------------------------------------------------------------------#
# residual block with (3 x 3) filters and down-sampling

class ResidualBlockDownSamp(nn.Module):
    
    ####
    def __init__( self, in_channels, out_channels ):
    
        super(ResidualBlockDownSamp, self).__init__()
        
        self.conv_res = nn.Conv2d( in_channels, out_channels, kernel_size=1, 
            stride=1, padding=0, bias=True )
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        self.conv1 = nn.Conv2d( in_channels, in_channels, kernel_size=3, 
            stride=1, padding=1, bias=False )
        
        self.bn2 = nn.BatchNorm2d(in_channels)
        
        self.conv2 = nn.Conv2d( in_channels, out_channels, kernel_size=3, 
            stride=1, padding=1, bias=True )
        
        # initialize weights
        kaiming_init(self.conv1)
        kaiming_init(self.conv2)
        
        
    ####
    def forward(self, x):
        
        out = F.avg_pool2d(x, 2)
        residual = self.conv_res(out)
        
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.avg_pool2d(out, 2)
        
        return residual + out
    

## test 
#resblk_dn = ResidualBlockDownSamp(64, 128)
#out = resblk_dn(torch.randn(5,64,64,64))
        

#-----------------------------------------------------------------------------#

class Encoder_ResNet(nn.Module):
    
    '''
    encoder architecture for the "teapots" data
    '''
    
    ####
    def __init__(self, z_dim=10):
        
        super(Encoder_ResNet, self).__init__()
        
        self.z_dim = z_dim

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        
        self.rb2 = ResidualBlockDownSamp(64, 128)
        self.rb3 = ResidualBlockDownSamp(128, 256)
        self.rb4 = ResidualBlockDownSamp(256, 512)
        self.rb5 = ResidualBlockDownSamp(512, 512)
        
        self.fc6 = nn.Linear(4*4*512, 2*z_dim) 


    ####
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.rb2(out)
        out = self.rb3(out)
        out = self.rb4(out)
        out = self.rb5(out)
        out = out.view(out.size(0), -1)
        stats = self.fc6(out)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        std = torch.sqrt(torch.exp(logvar))
        
        return mu, std, logvar


## test 
#enc = Encoder_ResNet(10)
#out = enc(torch.randn(5,3,64,64))


#-----------------------------------------------------------------------------#
# residual block with (3 x 3) filters and up-sampling

class ResidualBlockUpSamp(nn.Module):
    
    ####
    def __init__( self, in_channels, out_channels, input_norm=True ):
    
        super(ResidualBlockUpSamp, self).__init__()
        
        self.input_norm = input_norm  # whether to apply BN and ReLU to input
        
        self.upsamp_res = nn.UpsamplingNearest2d(scale_factor=2)
        
        self.conv_res = nn.Conv2d( in_channels, out_channels, kernel_size=1, 
            stride=1, padding=0, bias=True )
        
        if self.input_norm:
            self.bn1 = nn.BatchNorm2d(in_channels)

        self.upsamp1 = nn.UpsamplingNearest2d(scale_factor=2)
          
        self.conv1 = nn.Conv2d( in_channels, out_channels, kernel_size=3, 
            stride=1, padding=1, bias=True )
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d( out_channels, out_channels, kernel_size=3, 
            stride=1, padding=1, bias=True )
        
        # initialize weights
        kaiming_init(self.conv1)
        kaiming_init(self.conv2)
        
        
    ####
    def forward(self, x):
        
        out = self.upsamp_res(x)
        residual = self.conv_res(out)
        
        if self.input_norm:
            out = F.relu(self.bn1(x))
        else:
            out = x
        out = self.upsamp1(out)
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        
        return residual + out
    

## test 
#resblk_up = ResidualBlockUpSamp(512, 512, input_norm=True)
#out = resblk_up(torch.randn(5,512,4,4))


#-----------------------------------------------------------------------------#

class Decoder_ResNet(nn.Module):
    
    '''
    decoder architecture for the "teapots" data
    '''
    
    ####
    def __init__(self, z_dim=10):
        
        super(Decoder_ResNet, self).__init__()
        
        self.z_dim = z_dim
        
        self.fc1 = nn.Linear(z_dim, 4*4*512)
        self.bn1 = nn.BatchNorm1d(4*4*512)

        self.rb2 = ResidualBlockUpSamp(512, 512, input_norm=False)
        self.rb3 = ResidualBlockUpSamp(512, 256)
        self.rb4 = ResidualBlockUpSamp(256, 128)
        self.rb5 = ResidualBlockUpSamp(128, 64)
        
        self.bn6 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)


    ####
    def forward(self, z):
        
        out = self.fc1(z)
        out = F.relu(self.bn1(out))
        out = out.view(out.size(0), 512, 4, 4)
        
        out = self.rb2(out)
        out = self.rb3(out)
        out = self.rb4(out)
        out = self.rb5(out)
        
        out = F.relu(self.bn6(out))
        x_recon = self.conv6(out)

        return x_recon


## test 
#dec = Decoder_ResNet(10)
#out = dec(torch.randn(5,10))



