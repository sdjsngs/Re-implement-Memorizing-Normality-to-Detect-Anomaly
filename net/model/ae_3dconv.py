# baseline AE models for video data
from __future__ import absolute_import, print_function
import torch
import torch.nn as nn
from net.model.build import MODEL_REGISTRY
from net.model.memory_module import MemModule

class Encoder(nn.Module):
    """
    encoder 
    channel 1 96 128 256 256
    """
    def __init__(self):
        super(Encoder,self).__init__()
        self.layer_1=self._make_layer(in_place=1,out_place=96,kernel_size=3,stride=(1, 2, 2),padding=1)
        self.layer_2 = self._make_layer(in_place=96, out_place=128, kernel_size=3, stride=2, padding=1)
        self.layer_3 = self._make_layer(in_place=128, out_place=256, kernel_size=3, stride=2, padding=1)
        self.layer_4 = self._make_layer(in_place=256, out_place=256, kernel_size=3, stride=2, padding=1)
    def forward(self,x):
        x_1=self.layer_1(x)
        x_2=self.layer_2(x_1)
        x_3=self.layer_3(x_2)
        x_4=self.layer_4(x_3)
        return x_4
    
    def _make_layer(self,in_place,out_place,kernel_size,stride,padding):
        
        layer=nn.Sequential(
            nn.Conv3d(in_place, out_place, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_place),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        return layer


class Decoder(nn.Module):
    """
    encoder
    channel 1 96 128 256 256
    """

    def __init__(self):
        super(Decoder, self).__init__()

        self.layer_1 = self._make_layer(in_place=256, out_place=256, kernel_size=3, stride=2, padding=1,out_padding=1)
        self.layer_2 = self._make_layer(in_place=256, out_place=128, kernel_size=3, stride=2, padding=1,out_padding=1)
        self.layer_3 = self._make_layer(in_place=128, out_place=96, kernel_size=3, stride=2, padding=1,out_padding=1)
        self.layer_4 = self._make_layer(in_place=96, out_place=1, kernel_size=3, stride=(1, 2, 2), padding=1,out_padding=(0,1,1))

    def forward(self, x):
        x_1 = self.layer_1(x)
        x_2 = self.layer_2(x_1)
        x_3 = self.layer_3(x_2)
        x_4 = self.layer_4(x_3)
        return x_4

    def _make_layer(self, in_place, out_place, kernel_size, stride, padding,out_padding):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_place, out_place, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=out_padding),
            nn.BatchNorm3d(out_place),
            nn.LeakyReLU(0.2, inplace=True),
        )

        return layer

@MODEL_REGISTRY.register()
class AutoEncoder(nn.Module):

    def __init__(self,cfg):
        super(AutoEncoder,self).__init__()
        self.cfg=cfg
        self.encoder=Encoder()
        self.decoder=Decoder()

    def forward(self, x):
        encoder_x=self.encoder(x)
        decoder_x=self.decoder(encoder_x)
        return decoder_x

@MODEL_REGISTRY.register()
class AutoEncoderMemory(nn.Module):

    def __init__(self,cfg):
        super(AutoEncoderMemory,self).__init__()
        self.cfg=cfg
        self.encoder=Encoder()
        self.decoder=Decoder()
        self.memory=MemModule(
            cfg.MODEL.MEMORY_DIM,cfg.MODEL.FEATURE_DIM,cfg.MODEL.SHRINK_THRES
        )

    def forward(self, x):
        encoder_x=self.encoder(x)
        memory_x,att_weight=self.memory(encoder_x)
        decoder_x=self.decoder(memory_x)
        return decoder_x ,att_weight


class AutoEncoderCov3D(nn.Module):
    def __init__(self, chnum_in):
        super(AutoEncoderCov3D, self).__init__()
        self.chnum_in = chnum_in # input channel number is 1;
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        self.encoder = nn.Sequential(
            nn.Conv3d(self.chnum_in, feature_num_2, (3,3,3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_2, feature_num, (3,3,3), stride=(2,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num, feature_num_x2, (3,3,3), stride=(2,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_x2, feature_num_x2, (3,3,3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(feature_num_x2, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num_x2, feature_num, (3,3,3), stride=(2,2,2), padding=(1,1,1), output_padding=(1,1,1)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num, feature_num_2, (3,3,3), stride=(2,2,2), padding=(1,1,1), output_padding=(1,1,1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num_2, self.chnum_in, (3,3,3), stride=(1,2,2), padding=(1,1,1), output_padding=(0,1,1))
        )

    def forward(self, x):
        f = self.encoder(x)
        out = self.decoder(f)
        return out




if __name__=="__main__":
    print("ae 3d conv ")
    x=torch.randn(size=[2,1,16,224,224]).cuda()
    model=AutoEncoder(cfg=None).cuda()
    y=model(x)
    print(y.shape)
