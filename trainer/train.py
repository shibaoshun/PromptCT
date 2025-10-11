import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import torch
import time
import numpy as np
# from skimage.measure import compare_ssim as skssim
# from skimage.measure import compare_psnr as skpsnr
from skimage.metrics import peak_signal_noise_ratio as skpsnr
from skimage.metrics import structural_similarity as skssim
from torch.utils.tensorboard import SummaryWriter
import random
import torch.backends.cudnn as cudnn
from head_HM import init_logger
import torch.nn.functional as F


class Trainer():
    def __init__(self, args, model, tr_dset, vl_dset):
        self.args = args
        self.model = model
        self.epoch = args.epoch
        self.bat_size = args.tr_batch
        self.valbat_size = args.vl_batch
        self.tr_dset = tr_dset
        self.vl_dset = vl_dset
        self.training_params = {}
        self.best_psnr = 0
        self.best_psnr_epoch = 0


    def tr(self):
        # ——————————————————————随机种子—————————————————————————
        if self.args.manualSeed is None:
            self.args.manualSeed = random.randint(1, 10000)
        print("Random Seed: ", self.args.manualSeed)
        random.seed(self.args.manualSeed)
        torch.manual_seed(self.args.manualSeed)
        cudnn.benchmark = True
        # —————————————————————
        writer = SummaryWriter("./logs_train")
        logger = init_logger(self.args)

        self.model = self.model.cuda()
        num = self.print_network(self.model)
        logger.info("\tTotal number of parameters: {}".format(num))

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        # self.scheduler = MultiStepLR(self.optimizer, milestones=self.args.milestone, gamma=0.5)

        start = 0
        if self.args.resume == True:
            start = self.resume_tr()
        print('************train dataset length:{}************'.format(len(self.tr_dset)))
        DLoader = DataLoader(dataset=self.tr_dset, num_workers=0, drop_last=True, batch_size=self.bat_size,shuffle=True)#shuffle=True
        DLoaderval = DataLoader(dataset=self.vl_dset, num_workers=0, drop_last=True, batch_size=self.valbat_size,shuffle=True)
        # —————————————————————

        for epoch in range(start,self.epoch):
            t_start = time.time()
            # _________lr_________________
            if epoch > self.args.milestone[2]:
                current_lr = self.args.lr / 8
            elif self.args.milestone[1] < epoch and epoch < self.args.milestone[2] or epoch == self.args.milestone[2]:
                current_lr = self.args.lr / 4
            elif self.args.milestone[0] < epoch and epoch < self.args.milestone[1] or epoch == self.args.milestone[1]:
                current_lr = self.args.lr / 2
            else:
                current_lr = self.args.lr
            # current_lr = self.args.lr
            print(f"当前学习率：{current_lr}")
            loss_per_epoch_tr = 0
            psnr_per_epoch_tr = 0
            ssim_per_epoch_tr = 0

            for n_count, data_batch in enumerate(DLoader):
                bat_num = len(DLoader)
                if n_count % 50 == 0:
                    print('\n   ********** This is the training stage *************')
                batch_img, batch_u0, batch_sino, angle = [x.cuda() for x in data_batch]
                self.model.train()
                self.optimizer.zero_grad()
                Listout, batch_x_out, W = self.model(batch_sino, batch_u0, angle)
                loss_l1X1 = 0
                loss_l2X1 = 0
                # egt = batch_u0 - batch_img
                lossfunction = torch.nn.MSELoss(reduction='sum')
                for j in range(self.args.layers):  # 内部循环为T=10
                    loss_l1X1 = float(loss_l1X1) + 0.1 * torch.sum(torch.abs(Listout[j] - batch_img))
                    loss_l2X1 = float(loss_l2X1) + 0.1 * F.mse_loss(Listout[j], batch_img)
                loss1 = lossfunction(batch_x_out, batch_img)
                loss = loss1 + loss_l1X1 + loss_l2X1
                loss.backward()
                self.optimizer.step()
                loss_iter = loss.item()
                psnr_iter = self.aver_psnr(batch_x_out, batch_img)
                ssim_iter = self.aver_ssim(batch_x_out, batch_img)
                loss_per_epoch_tr += loss_iter
                psnr_per_epoch_tr += psnr_iter.item()
                ssim_per_epoch_tr += ssim_iter.item()

                if n_count % 5 == 0:
                    template = '[***] Epoch {} of {}, Batch {} of {}, loss={:5.2f}, psnr={:4.2f}, ssim={:5.4f}, lr={:.2e}'
                    print(template.format(epoch + 1, self.epoch, n_count, bat_num, loss_iter, psnr_iter, ssim_iter, current_lr))

            loss_per_epoch_tr /= (n_count + 1)
            psnr_per_epoch_tr /= (n_count + 1)
            ssim_per_epoch_tr /= (n_count + 1)
            print('Train: Loss={:+.2e} PSNR={:4.2f} SSIM={:5.4f}'.format(loss_per_epoch_tr, psnr_per_epoch_tr,
                                                                         ssim_per_epoch_tr))
            # ———————————————————— val stage————————————————————————————————————
            print('\n   ********** This is the validation stage *************')
            print('************val dataset length:{}************'.format(len(self.vl_dset)))
            rmse_per_epoch_vl = 0
            psnr_per_epoch_vl = 0
            ssim_per_epoch_vl = 0
            self.model.eval()
            for n_count, data_batch in enumerate(DLoaderval):
                val_img, val_u0, val_sino, angle = [x.cuda() for x in data_batch]
                with torch.no_grad():
                    Listout, val_x_db, W = self.model(val_sino, val_u0, angle)
                psnr_iter = self.aver_psnr(val_x_db, val_img)
                ssim_iter = self.aver_ssim(val_x_db, val_img)
                rmse_iter = torch.sqrt(torch.mean((val_x_db - val_img) ** 2))
                psnr_per_epoch_vl += psnr_iter.item()
                ssim_per_epoch_vl += ssim_iter.item()
                rmse_per_epoch_vl += rmse_iter.item()
                log_str = '[***] Epoch {} of {}, val:{:0>3d}, psnr={:4.2f}, ssim={:5.4f}, rmse={:5.4f} lr={:.2e} '
                print(log_str.format(epoch + 1, self.epoch, n_count + 1, psnr_iter, ssim_iter, rmse_iter,current_lr))

            psnr_per_epoch_vl /= (n_count + 1)
            ssim_per_epoch_vl /= (n_count + 1)
            rmse_per_epoch_vl /= (n_count + 1)
            print('val PSNR mean:{}'.format(psnr_per_epoch_vl))
            print('val SSIM mean:{}'.format(ssim_per_epoch_vl))
            print('val RMSE mean:{}'.format(rmse_per_epoch_vl))

            t_end = time.time()
            time_ = t_end - t_start
            print(' One Epoch consumes time= %2.2f' % (time_))

         # ————————————————————————————— save best model————————————————————————
            if (epoch + 1) % 1 == 0 or epoch == self.epoch - 1:
                #self.save_ckp(epoch)

                if psnr_per_epoch_vl > self.best_psnr:
                    self.best_psnr = psnr_per_epoch_vl
                    self.training_params['best_psnr'] = self.best_psnr
                    self.training_params['best_psnr_epoch'] = epoch + 1
                    self.best_psnr_epoch = epoch + 1
                    model_filename_best = self.args.model_dir + 'grad_best.pth'
                    self.save_ckp(model_filename_best)

                model_filename = self.args.model_dir + 'epoch%d.pth' % (epoch + 1)
                self.save_ckp(model_filename)

                logger.info(
                    "\tval: current_epoch:{}  psnr_val:{:.4f} ssim_val:{:.4f} rmse_val:{:.4f} time:{:.2f}  best_psnr:{:.4f}  best_psnr_epoch:{}"
                    .format(epoch + 1, psnr_per_epoch_vl, ssim_per_epoch_vl, rmse_per_epoch_vl, time_, self.best_psnr, self.best_psnr_epoch))
                print('-' * 100)

                print("best_psnr:{:.4f} best_psnr_epoch:{}".format(self.best_psnr, self.best_psnr_epoch))

                print('\n' + '--->' * 10 + 'Save CKP now！！!')

                writer.add_scalar('tr PSNR epoch', psnr_per_epoch_tr, epoch + 1)
                writer.add_scalar('tr SSIM epoch', ssim_per_epoch_tr, epoch + 1)
                writer.add_scalar('tr loss epoch', loss_per_epoch_tr, epoch + 1)
                # writer.add_scalar('tr RMSE_epoch', rmse_per_epoch_tr, epoch + 1)

                writer.add_scalar('val RMSE_epoch', rmse_per_epoch_vl, epoch + 1)
                writer.add_scalar('val PSNR epoch', psnr_per_epoch_vl, epoch + 1)
                writer.add_scalar('val SSIM epoch', ssim_per_epoch_vl, epoch + 1)
                # writer.add_scalar('val loss epoch', loss_per_epoch_vl, epoch + 1)
                writer.add_scalar('Learning rate', current_lr, epoch + 1)

            writer.close()
            # self.scheduler.step()
        print('Reach the maximal epoch! Finish training')



    def Myl2_reg_ortho(self,W):
        cols = W[0].numel()  # torch.numel() 查看元素个数
        cols = min(cols, W[0, 0, :, :].numel())  # lqs
        w1 = W.view(-1, cols)  # view()改变形状，与输入的总元素数一致
        wt = torch.transpose(w1, 0, 1)  # 交换一个张量的两个维度，只能有两个相关的位置交换
        m = torch.matmul(wt, w1)  # 矩阵相乘  torch.matmul()也是一种类似于矩阵相乘操作的tensor联乘操作。但是它可以利用python 中的广播机制，处理一些维度不同的tensor结构进行相乘操作。这也是该函数与torch.bmm()区别所在
        ident = torch.eye(cols, cols).cuda()
        w_tmp = m - ident
        l2_reg = torch.norm(w_tmp, 2) ** 2
        return l2_reg

    def resume_tr(self):
        ckp = torch.load(self.args.resume_ckp_dir)
        #print(ckp['model'])
        self.model.load_state_dict(ckp['model'], False)
        self.optimizer.load_state_dict(ckp['optimizer'])
        # self.scheduler.load_state_dict(ckp['scheduler'])
        self.training_params = ckp['training_params']
        print("self.training_params", self.training_params)
        self.best_psnr = self.training_params['best_psnr']
        self.best_psnr_epoch = self.training_params['best_psnr_epoch']
        return int(self.args.resume_ckp_resume)

    def print_network(self,net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('Total number of parameters: %d' % num_params)
        return num_params
    #打印wangluo参数个数

    def save_ckp(self, filename):
        #filename = self.model_dir + 'epoch%d.pth' % (epoch+1)
        #print(self.training_params)
        state = {'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 # 'scheduler': self.scheduler.state_dict(),
                 'training_params': self.training_params
                 }
        torch.save(state,filename)

    def psnr(self,img1, img2):
        if isinstance(img1, torch.Tensor):
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        psnr = skpsnr(img1.cpu().detach().numpy(), img2.cpu().detach().numpy(), data_range=1.0)
        #tensor转numpy计算
        return psnr

    def aver_psnr(self, img1, img2):
        PSNR = 0
        assert img1.shape == img2.shape
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                PSNR += self.psnr(img1[i, j:j + 1, ...], img2[i, j:j + 1, ...])
        return PSNR / (img1.shape[0] * img1.shape[1])

    def aver_ssim(self, img1, img2):
        '''used in the training'''
        # from skimage.measure import compare_ssim as ski_ssim
        SSIM = 0
        img1 = img1.detach().cpu().numpy().astype(np.float64)
        img2 = img2.detach().cpu().numpy().astype(np.float64)
        for i in range(len(img1)):
            for j in range(img1.shape[1]):
                SSIM += skssim(img1[i, j, ...], img2[i, j, ...], gaussian_weights=True, win_size=11, data_range=1.0, sigma=1.5)
        #a = len(img1)
        return SSIM / (len(img1) * img1.shape[1])

