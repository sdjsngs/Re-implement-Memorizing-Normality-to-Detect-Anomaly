"""
inference and save reconstruct loss to npy
"""
import torch
import os
import numpy as np
from net.utils.parser import load_config,parse_args
from net.utils.logging_tool import setup_logging
from net.dataset.loader import construct_loader
from net.model.build import build_model
import  net.utils.logging_tool as logging
import net.model.optimizer as optim
from net.utils.rng_seed import setup_seed
from net.model.losses import get_loss_func
from net.dataset import loader
from net.utils.meter import TestMeter
import net.utils.misc as misc
import net.utils.checkpoint as cu
import  net.utils.tensorboard_vis as Board

# logger
logger=logging.get_logger(__name__)

def Tensor2Numpy(input_tensor):
    """
    transform tensor to numpy
    [B,C,T,H,W] -> [B,T,H,W,C]
    :param input_tensor:
    :return:
    """
    numpy_x=input_tensor.permute(0,2,3,4,1).detach().cpu().numpy()
    return numpy_x

def init_recon_error_dict(cfg):
    """
    init recon error dict
    :return:
    """
    recon_error={}
    folder_num=np.arange(cfg.TEST.VIDEO_NUM)
    for num in folder_num:
        recon_error["%02d" % (num+1)]=[]
    return recon_error
def save_error_npy(recon_error_dict,cfg):
    """
    save error to npy file for auc calculate
    :return:
    """
    for key in recon_error_dict.keys():
        save_npy=np.array(recon_error_dict[key])
        np.save(os.path.join(cfg.TEST.SAVE_NPY_PATH,key+".npy"),save_npy)

def infer_epoch(test_loader,model,recon_error,cfg):
    """

    :param test_loader:
    :param model:
    :param test_meter:
    :param cfg:
    :return:
    """
    model.eval()


    for cur_iter ,(imgs,vidieo_index) in enumerate(test_loader):
        # batch size =1
        with torch.no_grad():
            imgs=imgs.cuda().float()

            pred_imgs,att_weight=model(imgs)

            imgs_numpy=Tensor2Numpy(imgs)
            pred_imgs_numpy=Tensor2Numpy(pred_imgs)

            mse_error=np.mean((imgs_numpy-pred_imgs_numpy)**2)

            recon_error[vidieo_index[0]].append(mse_error)

    save_error_npy(recon_error,cfg)


def infer(cfg):
    """
    infer  func in anomaly detection
    infer in one epoch and save the mse loss to npy
    :param cfg:
    :return:
    """

    logging.setup_logging(cfg.OUTPUT_DIR,cfg.TEST_LOGFILE_NAME)
    logger.info("infer and save score  with config")

    # build model
    model=build_model(cfg)

    optimizer=optim.construct_optimizer(model,cfg)

    # load checkpoint if exist
    if cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("load from last checkpoint")
        last_checkpoint=cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        checkpoint_epoch=cu.load_checkpoint(
            last_checkpoint,model,optimizer
        )

    elif cfg.TEST.CHECKPOINT_FILE_PATH !="":
        logger.info("Load from given checkpoint file")
        checkpoint_epoch=cu.load_checkpoint(
            cfg.TEST.CHECKPOINT_FILE_PATH,
            model,
            optimizer,
        )

    test_loader=loader.construct_loader("test",cfg)

    recon_error=init_recon_error_dict(cfg)

    # misc.log_model_info(model,cfg)

    infer_epoch(test_loader,model,recon_error,cfg)



if __name__=="__main__":
    """
    load argpare 
    model 
    data 
    infer and save score   
    
    """
    args=parse_args()
    cfg=load_config(args)
    # setup_seed(cfg.RNG_SEED)
    infer(cfg)
    recon=init_recon_error_dict(cfg)
    # print(recon)
    # recon["01"].append(100)
    # recon["01"].append(200)
    # print(recon)



