import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from model.net_PromptCT import Gradient_Descent
from data_loader.dataset_blind import Phantom2dDatasettrain, Phantom2dDatasetval, Phantom2dDatasettest
from trainer.train import Trainer
from trainer.test import Tester
from head_HM import *
from config import get_config


if __name__ == '__main__':
    args = get_config()
    log(args)
    model = Gradient_Descent(args)
    if args.phase == 'tr':
        tr_dataset = Phantom2dDatasettrain(args, phase='tr', datadir=args.tr_dir, length=int(args.imagenum_train), angle=[60, 90, 120, 180])
        vl_dataset = Phantom2dDatasetval(args, phase='vl', datadir=args.vl_dir, length=int(args.imagenum_val), angle=[60, 90, 120, 180])
        train = Trainer(args, model, tr_dset=tr_dataset, vl_dset=vl_dataset)
        train.tr()

    elif args.phase == 'test':
        test_dir = args.test_img_dir
        test_dset = Phantom2dDatasettest(args, phase='test', datadir=test_dir, length=int(args.imagenum_test))
        test = Tester(args, model, test_dset=test_dset)
        test.test()
    print('[*] Finish!')







