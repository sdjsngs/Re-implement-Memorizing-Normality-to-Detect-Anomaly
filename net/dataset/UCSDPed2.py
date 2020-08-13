import torch
import  torch.nn as nn
from torch.utils.data import  Dataset,DataLoader
import os
import glob
import cv2
import numpy as np
from net.dataset.build import DATASET_REGISTRY
from net.utils.parser import parse_args,load_config

@DATASET_REGISTRY.register()
class Ucsdped2(Dataset):
    """
    Uscdped2 dataset

   It has 16 short clips for training, and another 12 clips for testing.
   Each clip has 150 to 200 frames, with a resolution of 360 × 240 pixels
    training
        -frames
            -01
                -000.jpg
    """
    def __init__(self,cfg,mode):
        assert  mode in ["train","training","test","testing"]
        self.data_root=cfg.PED.PATH_TO_DATA_DIR
        self.mode=mode+"ing"
        self.temporal_length=cfg.TEMPORAL_LENGTH

        self._consturct()
    def _consturct(self):
        """
        recode img path
        :return:
        """
        self.img_paths=[]
        for num_folder in os.listdir(os.path.join(self.data_root,self.mode,"frames")):
            folder_img=sorted(glob.glob(
                os.path.join(self.data_root,self.mode,"frames",num_folder,"*.jpg").replace("\\","/")
            ))
            self.img_paths+=folder_img[self.temporal_length//2:(-self.temporal_length//2+1)]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):

        video, video_idx = self.load_img_to_gray(self.img_paths[index])
        video = torch.from_numpy(video)
        video = video.unsqueeze(dim=0)

        if self.mode in ["train", "training"]:
            return video
        elif self.mode in ["test", "testing"]:
            return video, video_idx
        else:
            raise NotImplementedError(
                "mode {} is not supported".format(self.mode)
            )

    def load_img_to_gray(self, path):
        # resize h,w 192,128
        # print("path", path)
        img_num = int(path.split("\\")[-1].split(".")[0])
        video_idx = (path.split("\\")[0].split("/")[-1])
        for step, i in enumerate(range(-(self.temporal_length // 2), self.temporal_length // 2)):
            img_num_i = img_num + i
            str_img_num_i = "%03d" % img_num_i  # len 3 for each frame
            path_i = path.split("\\")[0] + "/" + str_img_num_i + ".jpg"

            img = cv2.imread(path_i)
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img / 255.0
            if step == 0:
                imgs = np.expand_dims(img, axis=0)
            else:
                imgs = np.concatenate((imgs, np.expand_dims(img, axis=0)), axis=0)

        return imgs, video_idx

if __name__=="__main__":
    args=parse_args()
    cfg=load_config(args)
    print(type(cfg))
    data_loader=DataLoader(Ucsdped2(cfg,mode="train"),batch_size=2,shuffle=False)

    for step ,(video) in enumerate(data_loader):
        print("step",step)
        print(video.shape)
