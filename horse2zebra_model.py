import torch.nn as nn
import torch
import pytorch_lightning as pl
from layers import ConvNorm2d,ConvTransposeNorm2d,ResBlocks2d
from hparams import horse2zebra as hparams
from torchsummaryX import summary
from image_pool import ImagePool
import itertools
from collections import OrderedDict

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

class GANLoss(nn.Module):
    def __init__(self,gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.real_label = torch.tensor(target_real_label)
        self.fake_label = torch.tensor(target_fake_label)
        self.register_buffer("real_label",self.real_label)
        self.register_buffer("fake_label",self.fake_label)
        self.gan_mode = gan_mode

        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"gan mode {gan_mode} is not implemented.")
    
    def get_target_tensor(self, prediction:torch.Tensor, target_is_real:bool) -> torch.Tensor:
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction:torch.Tensor, target_is_real:bool) -> torch.Tensor:
        target_tensor = self.get_target_tensor(prediction,target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss
class CycleGAN(pl.LightningModule):
    
    def __init__(self,hparams:hparams):
        
        # define names
        self.loss_names = ['D_A', 'G_B2A', 'cycle_A', 'idt_A', 'D_B', 'G_A2B', 'cycle_B','idt_B']
        self.visual_names_A = ['real_A', 'fake_B', 'rec_A','idt_B']
        self.visual_names_B = ['real_B', 'fake_A', 'rec_B','idt_A']

        self.visual_names = self.visual_names_A + self.visual_names_B
        self.model_names = ['G_B2A', 'G_A2B', 'D_A', 'D_B']

        # set hyper parameter
        self.my_hparams = hparams
        self.model_name = hparams.model_name
        self.lr = hparams.lr
        self.beta1 = hparams.beta1
        self.lambda_cycle = hparams.lambda_cycle
        self.lambda_identity = hparams.lambda_identity
        self.max_epochs = hparams.max_epochs
        self.decay_start_epoch = hparams.decay_start_epoch

        # define models
        self.netG_B2A = Generator(hparams.generator)
        self.netG_A2B = Generator(hparams.generator)

        self.netD_A = Discriminator(hparams.discriminator)
        self.netD_B = Discriminator(hparams.discriminator)

        # define image pool
        self.fake_A_pool = ImagePool(hparams.image_pool_size)
        self.fake_B_pool = ImagePool(hparams.image_pool_size)

        # define loss functions
        self.criterionGAN = GANLoss(hparams.gan_mode)
        self.criterionCycle = nn.L1Loss()
        self.crtierionIdt = nn.L1Loss()

    def configure_optimizers(self):
        optG = torch.optim.Adam(
            itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
            self.lr,betas=(self.beta1, 0.999),
        )
        schG = torch.optim.lr_scheduler.LambdaLR(optG,self.lr_curve)
        optD = torch.optim.Adam(
            itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
            self.lr,betas=(self.beta1, 0.999),
        )
        schD = torch.optim.lr_scheduler.LambdaLR(optD,self.lr_curve)
        return [optG,optD],[schG, schD]

    def lr_curve(self,epoch:int) ->float:
        linspace = self.max_epochs - self.decay_start_epoch
        if self.decay_start_epoch < epoch:
            r = 1 - ((epoch - self.decay_start_epoch)/ linspace)
        else: r = 1.0
        return r
                    

    def training_step(self,batch, batch_idx, optimizer_idx):
        real_A, real_B = batch
        self.real_A, self.real_B = real_A, real_B
        if optimizer_idx == 0:
            return self.step_G(real_A, real_B)
        if optimizer_idx == 1:
            return self.step_D(real_A, real_B)
        
    def step_G(self, real_A:torch.Tensor, real_B:torch.Tensor) -> OrderedDict:
        fake_B, fake_A = self.forward(real_A, real_B)
        rec_A, rec_B = self.forward(fake_A, fake_B)
        self.fake_A,self.fake_B = fake_A, fake_B
        # Identity loss
        if self.lambda_identity > 0:
            idt_B,idt_A = self.forward(real_B, real_A)
            r = self.lambda_cycle * self.lambda_identity
            loss_idt_A = self.crtierionIdt(idt_A, real_A) * r
            loss_idt_B = self.crtierionIdt(idt_B, real_B) * r
            self.log('loss idt A',loss_idt_A)
            self.log('loss idt B',loss_idt_B)
        else:
            loss_idt_A,loss_idt_B = 0,0
        # GAN loss D_B(G_A2B(A))
        loss_G_A2B = self.criterionGAN(self.netD_B(fake_B),True)
        self.log('loss G_A2B', loss_G_A2B)
        # GAN loss D_A(G_B2A(B))
        loss_G_B2A = self.criterionGAN(self.netD_A(fake_A), True)
        self.log('loss G_B2A', loss_G_B2A)
        # Forward cycle loss || G_B2A(G_A2B(A)) - A ||
        loss_cycle_A = self.criterionCycle(rec_A, self.real_A) * self.lambda_cycle
        self.log('loss cycle A', loss_cycle_A)
        # Backward cycle loss || G_A2B(G_B2A(B)) - B ||
        loss_cycle_B = self.criterionCycle(rec_B, real_B) * self.lambda_cycle
        self.log('loss cycle B', loss_cycle_B)

        loss_G = loss_idt_A + loss_idt_B + loss_G_A2B + loss_G_B2A + loss_cycle_A + loss_cycle_B
        self.log('loss_G', loss_G)

        tqdm_dict = {'loss_G':loss_G}
        output = OrderedDict({'loss':loss_G,'progress_bar': tqdm_dict, 'log': tqdm_dict})
        return output

    def step_D(self,real_A:torch.Tensor, real_B:torch.Tensor)-> OrderedDict:
        # D_A
        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_A = self._step_D_basic(self.netD_A, real_A, fake_A)
        self.log('loss D_A', loss_D_A)
        # D_B
        fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D_B = self._step_D_basic(self.netD_B, real_B, fake_B)
        self.log('loss D_B',loss_D_B )

        loss_D = loss_D_A + loss_D_B
        tqdm_dict = {'loss D',loss_D}
        output = OrderedDict({'loss':loss_D, 'progress_bar': tqdm_dict, 'log': tqdm_dict})
        return output

    def _step_D_basic(self,netD:nn.Module, real:torch.Tensor, fake:torch.Tensor):

        # real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        #fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        return (loss_D_real+loss_D_fake) * 0.5
    def forward(self,real_A:torch.Tensor=None, real_B:torch.Tensor=None):
        if real_A is not None:
            real_A = self.netG_A2B(real_A)
        if real_B is not None:
            real_B = self.netG_B2A(real_B)
        fake_B, fake_A = real_A, real_B
        return fake_B, fake_A

if __name__ == '__main__':
    model = Discriminator(hparams.discriminator)
    model.summary()
    