model_name = 'horse2zebra'

# image settings
channels:int = 3
height:int = 256
width:int = 256

max_epochs:int = 200
decay_start_epoch:int = 100

init_gain:float = 0.02
init_type:str = 'normal'

lr:float = 0.0002
beta1:float = 0.5
lambda_cycle:float = 10.0
lambda_identity:float = 0.1

image_pool_size:int = 50
gan_mode:str = 'lsgan'

import torch.nn as nn
class generator:
    channels = channels
    height,width = height,width
    init_gain:float = 0.02

    norm_mode:str = 'InstanceNorm' # "BatchNorm" or "InstanceNorm"
    norm_layer:nn.Module= nn.InstanceNorm2d
    padding_mode = 'reflect'
    ngf:int = 64

    n_downsampling:int = 2
    n_reslayers:int = 6

class discriminator:
    channels = channels
    ndf:int = 64
    n_layers:int = 3

    init_gain:float = 0.02
    pass
