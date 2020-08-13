"""
load dataset
"""
import torch
from torch.utils.data import  DataLoader
from .build import  build_dataset


def construct_loader(mode,cfg,):
    """
    consturct data loader
    :param cfg:
    :param mode:
    :return:
    """
    assert mode in ["train","test"]
    if mode in ["train"]:
        dataset_name=cfg.TRAIN.DATASET
        batch_size=cfg.TRAIN.BATCH_SIZE
        shuffle=True
    elif mode in ["test"]:
        dataset_name=cfg.TEST.DATASET
        batch_size=cfg.TEST.BATCH_SIZE
        shuffle=False
    # get dataset in torch.util.data.Dataset
    dataset=build_dataset(dataset_name,mode,cfg)

    loader=torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=cfg.DATA_LOADER.DROP_LAST,

    )
    return loader

