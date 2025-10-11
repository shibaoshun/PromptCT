import argparse
import numpy as np
class get_config():
    def __init__(self):
        # Parse from command line
        self.parser = argparse.ArgumentParser(description='CT img Recon')
        self.parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
        self.parser.add_argument('--gpu_idx', type=int, default=0, help='idx of gpu')
        self.parser.add_argument('--data_type', default='IMA', help='dcm, IMA')
        self.parser.add_argument('--resume', default=False, help='resume training')
        self.parser.add_argument('--manualSeed', type=int, default=205, help='manual seed')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
        self.parser.add_argument('--eta', type=float, default=2e-4, help='eta')
        self.parser.add_argument('--tau', type=float, default=0.9, help='tau')
        self.parser.add_argument('--layers', type=int, default=3, help='stage')
        self.parser.add_argument('--log_dir', default='./result/logs/', help='tensorboard logs')
        # Training Parameters
        self.parser.add_argument("--milestone", type=int, default=[100, 200, 300], nargs='+',
                                 help="When to decay task")
        self.parser.add_argument('--epoch', type=int, default=100, help='#epoch ')
        self.parser.add_argument('--tr_batch', type=int, default=1, help='batch size')
        self.parser.add_argument('--vl_batch', type=int, default=1, help='val batch size')
        self.parser.add_argument('--ts_batch', type=int, default=1, help='batch size')
        self.parser.add_argument('--test_model', default='best.pth', help='dcm, png')

        self.parser.add_argument('--imagenum', type=int, default=4000, help='the number of batch')
        self.parser.add_argument('--img_size', default=[512,512], help='image size')
        self.parser.add_argument('--sino_size', nargs='*', default=[360, 800], help='sino size')
        self.parser.add_argument('--poiss_level',default=5e6, help='Poisson noise level')
        self.parser.add_argument('--gauss_level',default=[0.05], help='Gaussian noise level')
        self.parser.add_argument('--phase', type=str, default='tr', choices=['tr', 'test'],
                            help='Select running phase: tr for training, test for testing')
        self.parser.parse_args(namespace=self)


        self.mode= 'sparse'   #sparse,limited
        self.imagenum_train = '4000'
        self.imagenum_val = '400'
        self.imagenum_test = '500'


        # Result saving locations
        self.info = self.mode
        self.img_dir = './result/' + self.info + '/img/'
        self.model_dir = './result/' + self.info + '/ckp/'


        if self.phase == 'tr':
            print('!!!!!!training!!!!!!')
            print('This is mode %s' % self.mode)

            self.tr_dir = './data/train'
            self.vl_dir = './data/val'


            if self.resume == True:
                self.resume_ckp_dir = self.model_dir +'epoch14.pth'
                self.resume_ckp_resume = '14'
        elif self.phase == 'test':
            self.test_ckp_dir = self.model_dir + self.test_model
            print('!!!!!!testing!!!!!!')
            print('This is mode %s' %self.mode)
            print(self.test_ckp_dir)
            self.test_save_img = True
            self.test_img_dir = './data/test'





