"""
eval auc curve
pred score in npy
ground true in mat
"""
import  matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve
from net.utils.parser import load_config,parse_args
import net.utils.logging_tool as logging
from net.utils.load_ground_true import load_gt_ucsd_ped2
logger=logging.get_logger(__name__)

def remove_edge(plt):
    """
    visual ground in non-line bar
    :param plt:
    :return:
    """
    fig, ax = plt.subplots()
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

def show_ground_true(y,score):
    # ax = plt.gca()  # 获取到当前坐标轴信息
    # ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
    # ax.invert_yaxis()  # 反转Y坐标轴
    plt.xlim((0, len(y)))
    plt.ylim((0, 1.01))
    x=np.arange(len(y))
    plt.plot(x, score,"r")
    plt.bar(x,y,width=1)

    plt.show()
def show_line_one_video(y_score):

    x=np.arange(len(y_score))
    plt.plot(x,y_score)
    plt.show()

def show_pred_score_and_ground_true(y_score,y_label):
    x = np.arange(len(y_score))
    plt.plot(x, y_score,"r--")
    plt.plot(x,y_label,"g--")
    plt.show()

def roc_draw(y_pred_score,y_label):
    """
    draw roc
    :param y_pred:
    :param y_score:
    :return:
    """
    fpr, tpr, thresholds =roc_curve(y_label, y_pred_score, pos_label=None, sample_weight=None,

                              drop_intermediate=True)

    plt.title("roc curve")
    plt.plot(fpr, tpr, marker='o')
    plt.show()

def cal_auc(y_pred,y_label):
    """
    calculate auc
    :param y_pred:
    :param y_label:
    :return:
    """
    assert len(y_pred)==len(y_label)
    auc=roc_auc_score(y_label,y_pred)
    return auc

def Normalize(scoer_array):
    """
    normalize score=1-((score-score.min)/score.max)
    :param list_file:
    :return:
    """
    score_norm=1-((scoer_array-scoer_array.min())/scoer_array.max())
    return score_norm

def load_npy(folder,num_npy):
    """
    load npy file
    :param folder:
    :param num_npy:
    :return: [singel_npy and total npy]
    """
    y_pred_score=[]
    for i in range(num_npy):
        filename = '%s/%02d.npy' % (folder, i + 1)
        single_npy=np.load(filename)
        # noramlize
        y_pred_score.append(list(Normalize(single_npy)))
    for npy in y_pred_score:
        print(len(npy))
    return y_pred_score


def eval_auc_roc(cfg):
    """
    load y_pred_score  len = list * cfg.TEST.VIDEO_NUM
    load y_label {0,1} 0 for abnormal  1 for normal
    :param cfg:
    :return:
    """
    logging.setup_logging(cfg.OUTPUT_DIR,cfg.AUC_LOGFILE_NAME)
    y_pred_score=load_npy(cfg.TEST.SAVE_NPY_PATH,cfg.TEST.VIDEO_NUM)
    y_label=load_gt_ucsd_ped2(cfg.PED.MAT_FILE)
    auc_values=[]
    assert len(y_pred_score)==len(y_label) ,"len{} and len{}not match".format("y_pred_score","y_label")
    logger.info("auc for each video and all video ")

    # for  i in range(len(y_pred_score)):
    #     print(y_label[i])
    #     auc_value=cal_auc(y_pred_score[i],y_label[i])
    #     logger.info("{}th auc_value{}".format(i+1,auc_value) )
    #     auc_values.append(auc_value)
    #
    # logger.info("mean auc value {}".format(np.mean(np.array(auc_values))))
    # calculate total auc

    total_y_pred=[i for item in y_pred_score for i in item ]
    total_y_label=[i for item in y_label for i in item ]

    auc_value = cal_auc(total_y_pred, total_y_label)
    roc_draw(total_y_pred, total_y_label)
    logger.info("total auc value:{}".format(auc_value))

if __name__=="__main__":
    """
    load pred score 
    score close to 0 mean anomaly 
    load ground true
    cal auc value 
    draw roc  
    """
    args=parse_args()
    cfg=load_config(args)
    # eval_auc_roc(cfg)
    y=list([0,0,0,0,1,1,1,1,0,0,])
    score=list([0.1,0.2,0.2,0.3,0.4,0.5,1,1,1,1])
    show_ground_true(y,score)



