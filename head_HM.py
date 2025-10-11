import logging
import os

def init_logger(argdict):
    r"""Initializes a logging.Logger to save all the running parameters to a
    log file

    Args:
    	argdict: dictionary of parameters to be logged
    """
    from os.path import join
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    fh = logging.FileHandler(join(argdict.log_dir, 'log_train.txt'), mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # try:
    # 	logger.info("Commit: {}".format(get_git_revision_short_hash()))
    # except Exception as e:
    # 	logger.error("Couldn't get commit number: {}".format(e))
    logger.info("Arguments: ")
    for k in argdict.__dict__:
        logger.info("\t{}: {}".format(k, argdict.__dict__[k]))
    return logger



def init_logger_test(argdict):
    r"""Initializes a logging.Logger to save all the running parameters to a
    log file

    Args:
    	argdict: dictionary of parameters to be logged
    """
    from os.path import join
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    fh = logging.FileHandler(join(argdict.log_dir, 'log_test.txt'), mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # try:
    # 	logger.info("Commit: {}".format(get_git_revision_short_hash()))
    # except Exception as e:
    # 	logger.error("Couldn't get commit number: {}".format(e))
    logger.info("Arguments: ")
    for k in argdict.__dict__:
        logger.info("\t{}: {}".format(k, argdict.__dict__[k]))
    return logger



def log(args):
    # ——————————————————— use_gpu=0——————————————————————————
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)
    mkdir(args.log_dir)
    mkdir(args.img_dir)
    mkdir(args.model_dir)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  " + path + "  ---")
    else:
        print("---  There exsits folder " + path + " !  ---")

