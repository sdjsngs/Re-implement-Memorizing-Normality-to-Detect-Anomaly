"""
train
"""
import torch
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
from net.utils.meter import TrainMeter
import net.utils.misc as misc
import net.utils.checkpoint as cu
import  net.utils.tensorboard_vis as Board

# logger
logger=logging.get_logger(__name__)



def train_epoch(train_loader,model,optimizer,train_meter,cur_epoch,writer,cfg):
    """
    train in one epoch
    :param train_loader:
    :param model:
    :param optimizer:
    :param train_meter:
    :param cur_epoch:
    :param cfg:
    :return:
    """
    model.train()
    train_meter.iter_start()
    for cur_iter ,(imgs) in enumerate(train_loader):
        imgs=imgs.cuda().float()

        pred_imgs,att_weight=model(imgs)


        # show_img=imgs[0,:,0,:,:]
        # show_pred_img=pred_imgs[0,:,0,:,:]

        # writer.add_image("img", show_img)
        # writer.add_image("pred_img", show_pred_img)
        # writer.add_image("mse_img", show_img-show_pred_img)

        lr=optim.get_epoch_lr(cur_epoch,cfg)
        optim.set_lr(optimizer,lr)

        # loss_func=get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
        loss_func=get_loss_func(cfg.MODEL.LOSS_FUNC)
        mse_loss,entropy_loss=loss_func(imgs,pred_imgs,att_weight)
        combine_loss=mse_loss+entropy_loss

        misc.check_nan_losses(mse_loss)
        misc.check_nan_losses(entropy_loss)
        misc.check_nan_losses(combine_loss)

        optimizer.zero_grad()
        combine_loss.backward()
        optimizer.step()

        mse_loss=mse_loss.item()
        entropy_loss=entropy_loss.item()
        combine_loss=combine_loss.item()


        train_meter.iter_stop()
        train_meter.update_stats(
            mse_loss,entropy_loss,combine_loss, lr, imgs.size(0)
        )
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_start()


    writer.add_scalar("lr",train_meter.lr,cur_epoch+1)

    writer.add_scalar("mse_loss",train_meter.mse_loss.get_win_avg(),cur_epoch+1)
    writer.add_scalar("entropy_loss", train_meter.entropy_loss.get_win_avg(), cur_epoch + 1)
    writer.add_scalar("combine_loss", train_meter.combine_loss.get_win_avg(), cur_epoch + 1)


    train_meter.log_epoch_stats(cur_epoch)

    train_meter.reset()


def train(cfg):
    """
    train func in anomaly detection
     train video or frame  for many epoch
    :param cfg:
    :return:
    """

    logging.setup_logging(cfg.OUTPUT_DIR,cfg.TRAIN_LOGFILE_NAME)
    logger.info("train with config")

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
        start_epoch=checkpoint_epoch+1
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH !="":
        logger.info("Load from given checkpoint file")
        checkpoint_epoch=cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            optimizer,
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch=0

    train_loader=loader.construct_loader("train",cfg)

    train_meter=TrainMeter(len(train_loader),cfg)

    logger.info("Start epoch {}".format(start_epoch))

    writer=Board.init_summary_writer(cfg.TENSORBOARD.PATH)
    for cur_epoch in range(start_epoch,cfg.SOLVER.MAX_EPOCH):
        train_epoch(train_loader,model,optimizer,train_meter,cur_epoch,writer,cfg)

        # save checkpoint
        if cu.is_checkpoint_epoch(cur_epoch,cfg.TRAIN.CHECKPOINT_PERIOD):
            cu.save_checkpoint(cfg.OUTPUT_DIR,model,optimizer,cur_epoch,cfg)

    writer.close()

if __name__=="__main__":
    """
    load argpare 
    model 
    data 
    train  
    save model and tensorboard 
    """
    args=parse_args()
    cfg=load_config(args)
    setup_seed(cfg.RNG_SEED)
    train(cfg)
