model_name = 'horse2zebra'

# image settings
channels:int = 3
height:int = 256
width:int = 256


init_gain:float = 0.02
init_type:str = 'normal'


import torch.nn as nn
class generator:
    channels = channels
    height,width = height,width
    
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
    norm_mode:str = "BatchNorm"
    norm_layer:nn.Module= nn.BatchNorm2d
    pass
