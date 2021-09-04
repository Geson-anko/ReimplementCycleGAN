import torch.nn as nn
import torch
from typing import Tuple,Union

class ConvNorm2d(nn.Module):
    def __init__(self,in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]],
                stride: Union[int, Tuple[int]] = 1, padding: Union[int, Tuple[int]] = 0, 
                dilation: Union[int, Tuple[int]]= 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros',
                norm_layer:nn.Module=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)
        self.norm = norm_layer(out_channels)

    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        return x

class ConvTransposeNorm2d(nn.Module):
    def __init__(self,in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]],
                stride: Union[int, Tuple[int]] = 1, padding: Union[int, Tuple[int]] = 0, output_padding: Union[int, Tuple[int]] = 0,
                dilation: Union[int, Tuple[int]]= 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros',
                norm_layer:nn.Module=nn.BatchNorm2d):
        super().__init__()
        self.dconv = nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding,output_padding,groups,bias,dilation,padding_mode)
        self.norm = norm_layer(out_channels)

    def forward(self,x):
        x = self.dconv(x)
        x = self.norm(x)
        return x

class ConvInstanceNorm2d(nn.Module):
    def __init__(self,in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]],
                stride: Union[int, Tuple[int]] = 1, padding: Union[int, Tuple[int]] = 0, 
                dilation: Union[int, Tuple[int]]= 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        return x

class ConvTransposeInstanceNorm2d(nn.Module):
    def __init__(self,in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]],
                stride: Union[int, Tuple[int]] = 1, padding: Union[int, Tuple[int]] = 0, output_padding: Union[int, Tuple[int]] = 0,
                dilation: Union[int, Tuple[int]]= 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'
    ):
        super().__init__()
        self.dconv = nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding,output_padding,groups,bias,dilation,padding_mode)
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self,x):
        x = self.dconv(x)
        x = self.norm(x)
        return x

class ResBlock2d(nn.Module):
    def __init__(self,in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]],
                channel_divsor:int=4, **kwargs):
        super().__init__()
        assert kernel_size % 2 == 1
        assert out_channels >= channel_divsor

        ch = out_channels//channel_divsor
        pad = kernel_size//2
        self.init_conv = ConvNorm2d(in_channels,ch,kernel_size=1,**kwargs)
        self.conv = ConvNorm2d(ch,ch,kernel_size,padding=pad,**kwargs)
        self.out_conv= ConvNorm2d(ch,out_channels,kernel_size=1,**kwargs)
        self.shortcut_conv = self._generate_shortcut(in_channels,out_channels,kwargs) # skip connection

    def forward(self,x:torch.Tensor):
        h = torch.relu(self.init_conv(x))
        h = torch.relu(self.conv(h))
        h = self.out_conv(h)
        s = self.shortcut_conv(x) 
        y = torch.relu(h+s) # skip connection
        return y

    def _generate_shortcut(self,in_channels: int,out_channels: int,kwargs:dict):
        if in_channels != out_channels:
            return ConvNorm2d(in_channels,out_channels,kernel_size=1,**kwargs)
        else:
            return lambda x:x
        
class ResBlocks2d(nn.Module):
    def __init__(self,in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]],
                nlayers: int, channel_divsor:int=4,**kwargs):
        super().__init__()
        assert nlayers >=1
        self.first = ResBlock2d(in_channels,out_channels,kernel_size,channel_divsor,**kwargs)
        layers = [ResBlock2d(out_channels,out_channels,kernel_size,channel_divsor,**kwargs) for _ in range(nlayers-1)]
        self.layers = nn.ModuleList(layers)

    def forward(self,x:torch.Tensor):
        x = self.first(x)
        for l in self.layers:
            x = l(x)
        return x

if __name__ == '__main__':
    from torchsummaryX import summary
    m = ResBlocks2d(8,16,3,3,4,padding_mode='zeros',norm_layer=nn.InstanceNorm2d)
    d = torch.randn(1,8,16,16)
    summary(m,d)