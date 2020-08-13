"""
model construction function
"""
import  torch
import torch.nn as nn
from  fvcore.common.registry import  Registry
from torch.nn import init
MODEL_REGISTRY=Registry("MODEL")



def weights_init_kaiming(m):
    """
    kaiming init
    :param m:
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
    """

    :param m:
    :return:
    """
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

def build_model(cfg,):
    """
    build model  auto-encoder
    :param cfg:
    :param model_name:  autoencoder
    :return:
    """
    model_name=cfg.MODEL.MODEL_NAME
    # print("MODEL_REGISTRY", MODEL_REGISTRY.__dict__)
    model = MODEL_REGISTRY.get(model_name)(cfg)
    # init model  with xavier
    # model.apply(weights_init_xavier)
    model = model.cuda()
    return model



if __name__=="__main__":
    print("model register")
    print("MODEL_REGISTRY", MODEL_REGISTRY.__dict__)