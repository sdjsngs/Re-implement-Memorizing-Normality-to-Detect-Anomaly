"""
Configs
"""
from fvcore.common.config import CfgNode
from net.config import custom_config

# config definition
_C=CfgNode()


# train

_C.TRAIN=CfgNode()

_C.TRAIN.ENABLE=True

_C.TRAIN.DATASET=""

_C.TRAIN.BATCH_SIZE=10

# pre-train model path for fine tune
_C.TRAIN.CHECKPOINT_FILE_PATH= ""

_C.TRAIN.CHECKPOINT_PERIOD=10

_C.TRAIN.VIDEO_NUM=16

#test
_C.TEST=CfgNode()

_C.TEST.DATASET=""

_C.TEST.BATCH_SIZE=10

_C.TEST.CHECKPOINT_FILE_PATH=""

_C.TEST.ENABLE= True

_C.TEST.VIDEO_NUM=21

_C.TEST.SAVE_NPY_PATH=r""




#Optimizer  and LR
_C.SOLVER= CfgNode()

_C.SOLVER.BASE_LR=0.01

_C.SOLVER.LR_POLICY=""

_C.SOLVER.MAX_EPOCH= 40 # for avenue

_C.SOLVER.MOMENTUM=0.9

_C.SOLVER.WEIGHT_DECAY= 1e-4

_C.SOLVER.BIAS_WEIGHT_DECAY= 0

_C.SOLVER.OPTIMIZING_METHOD="sgd"

_C.SOLVER.GAMMA= 0.1

_C.SOLVER.DAMPEMING= 0.0

_C.SOLVER.NESTEROV= True


# Model
_C.MODEL=CfgNode()

_C.MODEL.MODEL_NAME="Autoencoder"

_C.MODEL.NUM_CLASSES=0

_C.MODEL.LOSS_FUNC="MSE_LOSS"

_C.MODEL.DROPOUT_RATE=0.2

_C.MODEL.MEMORY_DIM=2000

_C.MODEL.FEATURE_DIM=256

_C.MODEL.SHRINK_THRES=0.0025


# _C.DATA=CfgNode()
#
# _C.DATA.PATH_TO_DATA_DIR=""
#
# _C.DATA.INPUT_CHANNEL=1

# AVENUE

_C.AVENUE=CfgNode()

_C.AVENUE.PATH_TO_DATA_DIR=r"F:/avenue"

_C.AVENUE.TRAIN_CROP_SIZE= 224

_C.AVENUE.TEST_CROP_SIZE= 224

_C.AVENUE.INPUT_CHANNEL_NUM=1

_C.AVENUE.MAT_FOLDER=""

# USCD ped2
_C.PED=CfgNode()

_C.PED.PATH_TO_DATA_DIR=r"F:/ped2"

_C.PED.INPUT_CHANNEL_NUM=1

_C.PED.TRAIN_CROP_SIZE= 224

_C.PED.TEST_CROP_SIZE= 224

_C.PED.MAT_FILE=""




_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 1

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

_C.DATA_LOADER.DROP_LAST=False


_C.TENSORBOARD=CfgNode()

_C.TENSORBOARD.PATH=""

_C.TEMPORAL_LENGTH=16
#rng seed
_C.RNG_SEED=10

_C.LOG_PERIOD=10

# output dir
_C.OUTPUT_DIR=""

_C.TRAIN_LOGFILE_NAME=""

_C.TEST_LOGFILE_NAME=""

_C.AUC_LOGFILE_NAME=""

_C.NUM_GPUS=1


def get_cfg():

    """

    :return: get a cfg copy
    """
    return _C.clone()


if __name__=="__main__":
    print("defaults configs")

    cfg=get_cfg()
    print(cfg.TRAIN.BATCH_SIZE)
