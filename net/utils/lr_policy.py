"""
lr policy
"""



def get_lr_at_epoch(cfg,cur_epoch):
    """
    get lr in cur_epoch
    :param cfg:
    :param cur_epoch:
    :return:
    """
    lr=get_lr_func(cfg.SOLVER.LR_POLICY)(cfg,cur_epoch)

    return lr


def lr_func_steps(cfg,cur_ecpoh):
    """
    stpe lr  policy
    :param cfg:
    :param cur_ecpoh:
    :return:
    """
    ind=int(cur_ecpoh/10)
    return pow(0.1 ,ind)*cfg.SOLVER.BASE_LR


def lr_func_stack(cfg,cur_epoch):
    """
    get lr down when loss get stack
    :param cfg:
    :param cur_epoch:
    :return:
    """
    return


def get_lr_func(lr_policy):
    """
    give a lr_policy return a learning rate
    :param lr_policy:
    :return:
    """

    lr_func="lr_func_"+lr_policy
    if lr_func not  in globals():
        raise NotImplementedError(
            "Unknown LR policy: {}".format(lr_policy)
        )
    return  globals()[lr_func]


if __name__=="__main__":
    print(globals())