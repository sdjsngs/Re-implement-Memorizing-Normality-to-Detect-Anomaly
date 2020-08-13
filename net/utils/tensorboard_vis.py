"""
tesnorbaord vis py
"""
from tensorboardX import SummaryWriter


def init_summary_writer(summary_root):
    writer = SummaryWriter(summary_root)
    return writer

def loss_add(writer,loss_name,loss_item,cur_epoch):
    writer.add_scalar(loss_name, loss_item, cur_epoch)

def lr_add(writer,lr_name,lr_item,cur_epoch):
    writer.add_scalar(lr_name, lr_item, cur_epoch)

def show_img_and_flow(writer,img,pred_img):
    writer.add_image("raw img" ,img)
    writer.add_image("pred img", pred_img)
    writer.add_image("MSE img", img-pred_img)


if __name__=="__main__":
    print("this is tensorboard vis py ")