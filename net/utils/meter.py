"""
meter
"""
from fvcore.common.timer import Timer
import torch
import  json
import datetime
import  numpy as np
from collections import  defaultdict ,deque

import net.utils.logging_tool as logging


logger=logging.get_logger(__name__)





class ScalarMeter(object):
    """
    a scalar meter to calue mean and sum
    """
    def __init__(self,window_size):
        self.dequa=deque(maxlen=window_size)
        self.total=0.0
        self.count=0

    def reset(self):
        self.dequa.clear()
        self.total=0.0
        self.count=0

    def add_value(self,value):
        self.dequa.append(value)
        self.total+=value
        self.count+=1

    def get_win_median(self):
        """
        calculate the median value in current dequa
        :return:
        """
        return np.median(self.dequa)


    def get_win_avg(self):
        """
        calculate mean value in  dequa
        :return:
        """
        return np.mean(self.dequa)

    def get_global_avg(self):
        """
        calculate global mean value
        :return:
        """
        return self.total/self.count


class TrainMeter(object):

    def  __init__(self,epoch_iters,cfg):
        """

        :param epoch_iters: iters in one epoch
        :param cfg:
        """
        self._cfg=cfg
        self.epoch_iters=epoch_iters
        # self.loss=ScalarMeter(cfg.LOG_PERIOD)
        self.mse_loss=ScalarMeter(cfg.LOG_PERIOD)
        self.entropy_loss=ScalarMeter(cfg.LOG_PERIOD)
        self.combine_loss=ScalarMeter(cfg.LOG_PERIOD)
        self.iter_timer=Timer()
        self.lr=None
        # self.loss_total=0.0
        self.MAX_EPOCH=cfg.SOLVER.MAX_EPOCH * epoch_iters
        # self.num_samples=0

    def reset(self):

        """
        reset meter
        :return:
        """
        self.lr=None
        self.mse_loss.reset()
        self.entropy_loss.reset()
        self.combine_loss.reset()
        # self.loss_total=0.0
    def iter_start(self):
        """
        start to recode time
        :return:
        """
        self.iter_timer.reset()
    def iter_stop(self):
        """
        stop recode time
        :return:
        """
        self.iter_timer.pause()

    def update_stats(self,mse_loss,entropy_loss,combine_loss,lr,mb_size):

        self.mse_loss.add_value(mse_loss)
        self.entropy_loss.add_value(entropy_loss)
        self.combine_loss.add_value(combine_loss)
        self.lr=lr
        # self.loss_total+=loss*mb_size
        # self.num_samples+=mb_size

    def log_iter_stats(self,cur_epoch,cur_iter):
        """
        log the stats for cur iteration
        :param cur_epoch:
        :param cur_iter:
        :return:
        """
        if (cur_iter+1) % self._cfg.LOG_PERIOD!= 0:
            return
        eta_sec = self.iter_timer.seconds() * (
                self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))

        stats={
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch+1,self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter+1,self.epoch_iters),
            "time":self.iter_timer.seconds(),
            "eta":eta,
            "mse_loss":self.mse_loss.get_win_median(),
            "entropy_loss": self.entropy_loss.get_win_median(),
            "combine_loss": self.combine_loss.get_win_median(),
            "lr":self.lr,
            "gpu":"{:.2f}GB".format(torch.cuda.max_memory_allocated()/1024**3)
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self,cur_epoch):
        """

        :param cur_epoch:
        :return:
        """
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "mse_loss":self.mse_loss.get_win_avg(),
            "entropy_loss":self.entropy_loss.get_win_avg(),
            "combine_loss":self.combine_loss.get_win_avg(),
            "gpu_mem": "{:.2f} GB".format(torch.cuda.max_memory_allocated()/1024**3),
        }
        logging.log_json_stats(stats)


class TestMeter(object):

    def  __init__(self,epoch_iters,cfg):
        """

        :param epoch_iters: iters in one epoch
        :param cfg:
        """
        self._cfg=cfg
        self.epoch_iters=epoch_iters
        # self.loss=ScalarMeter(cfg.LOG_PERIOD)
        self.mse_loss=ScalarMeter(cfg.LOG_PERIOD)
        self.entropy_loss=ScalarMeter(cfg.LOG_PERIOD)
        self.combine_loss=ScalarMeter(cfg.LOG_PERIOD)
        self.iter_timer=Timer()
        self.lr=None
        # self.loss_total=0.0
        self.MAX_EPOCH=cfg.SOLVER.MAX_EPOCH * epoch_iters
        # self.num_samples=0

    def reset(self):

        """
        reset meter
        :return:
        """
        self.lr=None
        self.mse_loss.reset()
        self.entropy_loss.reset()
        self.combine_loss.reset()
        # self.loss_total=0.0
    def iter_start(self):
        """
        start to recode time
        :return:
        """
        self.iter_timer.reset()
    def iter_stop(self):
        """
        stop recode time
        :return:
        """
        self.iter_timer.pause()

    def update_stats(self,mse_loss,entropy_loss,combine_loss,lr,mb_size):

        self.mse_loss.add_value(mse_loss)
        self.entropy_loss.add_value(entropy_loss)
        self.combine_loss.add_value(combine_loss)
        self.lr=lr
        # self.loss_total+=loss*mb_size
        # self.num_samples+=mb_size

    def log_iter_stats(self,cur_epoch,cur_iter):
        """
        log the stats for cur iteration
        :param cur_epoch:
        :param cur_iter:
        :return:
        """
        if (cur_iter+1) % self._cfg.LOG_PERIOD!= 0:
            return
        eta_sec = self.iter_timer.seconds() * (
                self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))

        stats={
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch+1,self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter+1,self.epoch_iters),
            "time":self.iter_timer.seconds(),
            "eta":eta,
            "mse_loss":self.mse_loss.get_win_median(),
            "entropy_loss": self.entropy_loss.get_win_median(),
            "combine_loss": self.combine_loss.get_win_median(),
            "lr":self.lr,
            "gpu":"{:.2f}GB".format(torch.cuda.max_memory_allocated()/1024**3)
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self,cur_epoch):
        """

        :param cur_epoch:
        :return:
        """
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "mse_loss":self.mse_loss.get_win_avg(),
            "entropy_loss":self.entropy_loss.get_win_avg(),
            "combine_loss":self.combine_loss.get_win_avg(),
            "gpu_mem": "{:.2f} GB".format(torch.cuda.max_memory_allocated()/1024**3),
        }
        logging.log_json_stats(stats)





