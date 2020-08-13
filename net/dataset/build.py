"""
model construction function
"""
import  torch
import torch.nn as nn
from  fvcore.common.registry import  Registry
from torch.nn import init
DATASET_REGISTRY=Registry("DATASET")


def build_dataset(dataset_name,mode,cfg,):
    """

    :param cfg:
    :param dataset_name: avenue
    :param mode:  train /test
    :return:
    """
    # print("MODEL_REGISTRY", MODEL_REGISTRY.__dict__)
    name=dataset_name.capitalize()
    # init model  with xavier

    return DATASET_REGISTRY.get(name)(cfg,mode)



if __name__=="__main__":
    print("dataset register")
