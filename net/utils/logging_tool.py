"""
logging
"""


import  logging
import sys
import os
import decimal
import simplejson
from fvcore.common.file_io import PathManager


def _cached_log_stream(filename):
    return PathManager.open(filename, "a")

def setup_logging(output_dir=None,log_name=None):

    # Set up logging format. ???
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"

    logging.root.handlers=[]
    logging.basicConfig(
        level=logging.INFO,format=_FORMAT,stream=sys.stdout
    )

    logger=logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(plain_formatter)
    logger.addHandler(ch)

    filename = os.path.join(output_dir,log_name)
    # UCF101_split1_SF4X16UCF101_split1_tvsumAug3_train
    # _VASNet_TVsumAug3_SF4X16_test_with_index.logHMDB51_split1_Grid_Sample_train.log
    fh = logging.StreamHandler(_cached_log_stream(filename))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(plain_formatter)
    logger.addHandler(fh)

def get_logger(name):
    """
    get logging with the name
    :param name: (string)
    :return:
    """
    return logging.getLogger(name)


def log_json_stats(stats):

    stats={
        k:decimal.Decimal("{:.6f}".format(v)) if isinstance(v,float) else v
            for k ,v in stats.items()
    }
    json_stats=simplejson.dumps(stats,sort_keys=False,use_decimal=True)
    logger=get_logger(__name__)
    logger.info("json_stats:{:s}".format(json_stats))




if __name__=="__main__":
    print("logging file ")
    print(__name__)