"""
checkpoints file
"""
import net.utils.logging_tool as logging
import pickle
import os
import  torch
from fvcore.common.file_io import PathManager

logger=logging.get_logger(__name__)

def make_checkpoint_dir(path_to_checkpoint):

    checkpoint_dir=os.path.join(path_to_checkpoint,"checkpoints")

    if not PathManager.exists(checkpoint_dir):
        PathManager.mkdirs(checkpoint_dir)
    return checkpoint_dir

def get_checkpoint_dir(path_to_checkpoint):

    return os.path.join(path_to_checkpoint,"checkpoints")

def get_path_to_checkpoints(path_to_checkpoint,epoch):

    name="checkpoint_epoch_{:05d}.pyth".format(epoch)
    return os.path.join(get_checkpoint_dir(path_to_checkpoint),name)
def get_last_checkpoint(path_to_checkpoint):
    """
    Get the last checkpoint from the checkpointing folder.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """

    d = get_checkpoint_dir(path_to_checkpoint)
    names = PathManager.ls(d) if PathManager.exists(d) else []
    names = [f for f in names if "checkpoint" in f]
    assert len(names), "No checkpoints found in '{}'.".format(d)
    # Sort the checkpoints by epoch.
    name = sorted(names)[-1]
    return os.path.join(d, name)

def get_special_checkpoint(path_to_checkpoint,special_epoch):
    """
    get one special checkpoint
    :param path_to_checkpoint:
    :return:
    """

    d = get_checkpoint_dir(path_to_checkpoint)

    names = PathManager.ls(d) if PathManager.exists(d) else []
    special_name = "checkpoint_epoch_{:.05d}.pyth".format(special_epoch)
    names = [f for f in names if special_name in f]
    name=names[0]
    logger.info("load mode in special epoch : {}".format(os.path.join(d, name)))
    return os.path.join(d, name)


def has_checkpoint(path_to_checkpoint):
    """
    check if checkpoint exist
    :param path_to_checkpoint:
    :return:
    """
    d=get_checkpoint_dir(path_to_checkpoint)
    files=PathManager.ls(d) if PathManager.exists(d) else []

    return any("checkpoint" in f for f in files)

def is_checkpoint_epoch(cur_epoch,checkpoint_period):
    """
    determine if a checkpoint should be saved in cur_epoch
    :param cur_epoch:
    :param checkpoint_period:
    :return:
    """
    return (cur_epoch+1)%checkpoint_period==0

def save_checkpoint(path_to_checkpoint,model,optimizer,epoch,cfg):
    """
    save a checkpoint
    :param path_to_checkpoint:
    :param mdoel:
    :param optimizer:
    :param epoch:
    :param cfg:
    :return:
    """
    logger.info("save checkpoint in epoch {}".format(epoch))

    PathManager.mkdirs(get_checkpoint_dir(path_to_checkpoint))
    sd=model.state_dict()
    #Recode the state
    checkpoint={
        "epoch":epoch,
        "model_state":sd,
        "optimizer_state":optimizer.state_dict(),
        "cfg":cfg.dump()
    }
    checkpoint_path=get_path_to_checkpoints(path_to_checkpoint,epoch+1)
    # if (epoch+1)%10==0 or (epoch+1)==cfg.SOLVER.MAX_EPOCH:
    with PathManager.open(checkpoint_path,"wb") as f:
        torch.save(checkpoint,f)
    return checkpoint_path



def load_checkpoint(path_to_checkpoint,model,optimizer):
    """
    load checkpoint
    :return:
    """
    assert  PathManager.exists(path_to_checkpoint), "checkpoint {}".format(path_to_checkpoint)
    # load checkpoint on cpu
    with PathManager.open(path_to_checkpoint,"rb") as f:
        checkpoint=torch.load(f,map_location="cpu")
        logger.info("checkpoint {} is load ".format(path_to_checkpoint.split("/")[-1]))
    ms=model
    ms.load_state_dict(checkpoint["model_state"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if "epoch" in checkpoint.keys():
        epoch=checkpoint["epoch"]
    else:
        epoch=-1
    return epoch




if __name__=="__main__":
    print("checkpoint")
