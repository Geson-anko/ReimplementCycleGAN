import torch.nn as nn
import torch
import pytorch_lightning as pl
from layers import ConvNorm2d,ConvTransposeNorm2d,ResBlocks2d
from hparams import horse2zebra as hparams
from torchsummaryX import summary

class Generator(nn.Module):

    def __init__(self,hparams:hparams.generator):
        super().__init__()
        self.hparams = hparams
        norm_layer = hparams.norm_layer
        padding_mode = hparams.padding_mode
        inch = hparams.channels
        ngf = hparams.ngf
        n_downsampling = hparams.n_downsampling
        n_reslayers = hparams.n_reslayers


        # first convolution
        layers = [
            nn.ReflectionPad2d(3),
            ConvNorm2d(inch,ngf,kernel_size=7,norm_layer=norm_layer),
            nn.ReLU(True)
        ]
        # defining down sampling layer
        down_chs = [hparams.ngf*(2**m) for m in range(n_downsampling + 1)]
        for idx in range(n_downsampling):
            ch_prev,ch = down_chs[idx], down_chs[idx + 1]
            l = [
                ConvNorm2d(ch_prev,ch,3,2,1,padding_mode=padding_mode,norm_layer=norm_layer),
                nn.ReLU(True),
            ]
            layers += l
        # define resblock layer
        res = ResBlocks2d(down_chs[-1],down_chs[-1],3,n_reslayers,padding_mode=padding_mode,norm_layer=norm_layer)
        layers.append(res)

        # define up samping layers
        up_chs = down_chs[::-1]
        for idx in range(n_downsampling):
            ch_prev,ch =up_chs[idx],up_chs[idx+1]
            l = [
                ConvTransposeNorm2d(ch_prev,ch,3,2,1,1,norm_layer=norm_layer),
                nn.ReLU(True),
            ]
            layers += l

        # define output convolution
        l = [
            nn.Conv2d(ngf,inch,kernel_size=7,padding=3,padding_mode=padding_mode),
            nn.Tanh()
        ]
        layers += l
        #
        self.layers = nn.Sequential(*layers)

        self.init_weight()

    def forward(self,x:torch.Tensor):
        h = self.layers(x) + 0.5
        return h
    """
    def get_adjust_conv(self,in_channels:int, out_channels:int) -> nn.Module:
        h0,w0 = self.hparams.height,self.hparams.width
        h,w = h0,w0
        for _ in range(self.hparams.n_downsampling):
            h,w = self.calculate_conv_outlen(h,w,pad=1,dilation=1,kernel=3,stride=2)
        for _ in range(self.hparams.n_downsampling):
            h,w = self.calculate_convT_outlen(h,w,pad=1,dilation=1,kernel=3,stride=2,outpad=1)
        
        dh,dw = h0 - h, w0 - w
        if dh == 0 and dw == 0:
            return lambda x:x
        elif dh > 0 and dw > 0:
            return nn.ConvTranspose2d(in_channels,out_channels,kernel_size=(dh+1,dw+1))
        elif dh < 0 and dw < 0:
            return nn.Conv2d(in_channels,out_channels,kernel_size=(abs(dh),abs(dw)))
        else:
            raise NotImplementedError(f'input img size is {h0,w0}, output img size is {h,w}')

    def calculate_conv_outlen(self,*lengths,pad,dilation,kernel,stride) -> tuple:
        out = []
        for lin in lengths:
            lout = int((lin + 2*pad - dilation * (kernel -1) + 1)/stride + 1)
            out.append(lout)
        return tuple(out)

    def calculate_convT_outlen(self,*lengths,pad,dilation,kernel,stride,outpad) -> tuple:
        out = []
        for lin in lengths:
            lout = (lin - 1)* stride - 2*pad + dilation * (kernel - 1) + outpad + 1
            out.append(lout)
        return tuple(out)
    """
    def summary(self):
        dummy = torch.randn(1,hparams.channels,hparams.height,hparams.width)
        summary(self,dummy)
    
    @torch.no_grad()
    def clamp_flow(self,x:torch.Tensor):
        out = self(x)
        out = torch.clamp(out,0.0,1.0)
        return out

    def init_weight(self):
        for p in self.parameters():
            nn.init.normal_(p.data,0.0,self.hparams.init_gain)
class Discriminator(nn.Module):
    def __init__(self,hparams:hparams.discriminator):
        super().__init__()
        self.hparams  = hparams
        inch = hparams.channels
        ndf = hparams.ndf
        n_layers = hparams.n_layers

        kw = 4
        padw = 1
        layers = [ConvNorm2d(inch,ndf,kw,2,padw),nn.LeakyReLU(0.2,True)]
        chs = [ndf * min((2**i),8) for i in range(n_layers + 1)]
        for idx in range(n_layers-1):
            ch_prev,ch = chs[idx],chs[idx+1]
            l = [
                ConvNorm2d(ch_prev,ch,kw,2,padw),
                nn.LeakyReLU(0.2,True)
            ]
            layers += l
        
        ch_prev,ch = chs[-2],chs[-1]
        layers += [
            ConvNorm2d(ch_prev,ch,kw,padding=padw),
            nn.LeakyReLU(0.2,True)
        ]
        layers += [nn.Conv2d(chs[-1],1,kw,padding=padw)]
        self.layers = nn.Sequential(*layers)

        self.init_weight()

    def forward(self,x:torch.Tensor):
        h = self.layers(x)
        return h
    
    def summary(self):
        dummy = torch.randn(1,hparams.channels,hparams.height,hparams.width)
        summary(self,dummy)
    
    def init_weight(self):
        for p in self.parameters():
            nn.init.normal_(p.data,0.0,self.hparams.init_gain)

class CycleGAN(pl.LightningModule):
    pass

if __name__ == '__main__':
    model = Discriminator(hparams.discriminator)
    model.summary()
    