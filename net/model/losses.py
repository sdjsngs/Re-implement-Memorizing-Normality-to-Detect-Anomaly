"""
loss function
"""
import torch
import torch.nn as nn
import torch.nn.functional as F




def Combine_Loss(imgs,pred_imgs,att_weight):
    """
    loss = mse(pred_imgs-imgs)+ entropy_loss(att_weight)
    alpha=0.0002
    :return:
    """
    batch_size=imgs.size(0)
    mse_loss=nn.MSELoss()
    img_loss=mse_loss(imgs,pred_imgs)
    entropy_loss=EntropyLoss()
    weight_loss=0.0002*entropy_loss(att_weight,batch_size)

    return img_loss,weight_loss

_LOSSES={
    "MSE_LOSS":nn.MSELoss,
    "COMBINE_LOSS":Combine_Loss,
}

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()
    def forward(self, x,batch_size):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum() /batch_size
        return b



def get_loss_func(loss_name):

    if loss_name not in _LOSSES.keys():
        raise NotImplementedError(
            "loss {} is not in supported".format(loss_name)
        )
    return _LOSSES[loss_name]


if __name__=="__main__":
    x=torch.tensor([1,2,3,5]).cuda()
    img=torch.randn(size=[10,16])
    loss=EntropyLoss()
    print(loss(img,10))
