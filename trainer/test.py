from torch.utils.data.dataloader import DataLoader
import torch
import time
import numpy as np
# from skimage.measure import compare_ssim as skssim
# from skimage.measure import compare_psnr as skpsnr
from skimage.metrics import peak_signal_noise_ratio as skpsnr
from skimage.metrics import structural_similarity as skssim
import os
import matplotlib.pyplot as plt
from head_HM import init_logger_test


def save_image(idx, dir, datalist):
    for i in range(len(datalist)):
        file_dir = dir[i] + str(idx)+'.png'
        plt.imsave(file_dir, datalist[i].data.cpu().numpy().squeeze(), cmap="gray")


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print("---  There exsits folder " + path + " !  ---")


class Tester():
    def __init__(self, args, model, test_dset):
        self.args = args
        self.model = model
        self.epoch = args.epoch
        self.bat_size = args.tr_batch
        self.tsbat_size = args.ts_batch
        self.test_dset = test_dset

    def test(self):
        # writer = SummaryWriter("./logs_test")
        logger = init_logger_test(self.args)
        self.model = self.model.cuda()
        ckp = torch.load(self.args.test_ckp_dir)
        self.model.load_state_dict(ckp['model'], False)
        self.model.eval()
        DLoadertest = DataLoader(dataset=self.test_dset, drop_last=True, batch_size=self.tsbat_size,
                                 shuffle=False)
        print('************test dataset length:{}************'.format(len(self.test_dset)))
        t_start = time.time()
        out_dir60 = self.args.img_dir + '/60/PromptNet/image/'
        mkdir(out_dir60)
        input_dir60 = self.args.img_dir + '/60/input/image/'
        mkdir(input_dir60)
        gt_dir60 = self.args.img_dir + '/60/gt/image/'
        mkdir(gt_dir60)

        out_dir90 = self.args.img_dir + '/90/PromptNet/image/'
        mkdir(out_dir90)
        input_dir90 = self.args.img_dir + '/90/input/image/'
        mkdir(input_dir90)
        gt_dir90 = self.args.img_dir + '/90/gt/image/'
        mkdir(gt_dir90)

        out_dir120 = self.args.img_dir + '/120/PromptNet/image/'
        mkdir(out_dir120)
        input_dir120 = self.args.img_dir + '/120/input/image/'
        mkdir(input_dir120)
        gt_dir120 = self.args.img_dir + '/120/gt/image/'
        mkdir(gt_dir120)

        out_dir180 = self.args.img_dir + '/180/PromptNet/image/'
        mkdir(out_dir180)
        input_dir180 = self.args.img_dir + '/180/input/image/'
        mkdir(input_dir180)
        gt_dir180 = self.args.img_dir + '/180/gt/image/'
        mkdir(gt_dir180)


        psnr_iter60s = 0
        ssim_iter60s = 0
        rmse_iter60s = 0

        psnr_iter90s = 0
        ssim_iter90s = 0
        rmse_iter90s = 0

        psnr_iter120s = 0
        ssim_iter120s = 0
        rmse_iter120s = 0

        psnr_iter180s = 0
        ssim_iter180s = 0
        rmse_iter180s = 0

        print("\t 60,90,120,180 ")
        for n_count, data_batch in enumerate(DLoadertest):
            n_count += 1
            Xgt, Xfbp60, S60, Xfbp90, S90, Xfbp120, S120, Xfbp180, S180 = [x.cuda() for x in data_batch]
            with torch.no_grad():
                list60, val60, W = self.model(S60, Xfbp60, angle=60)
                list90, val90, W = self.model(S90, Xfbp90, angle=90)
                list120, val120, W = self.model(S120, Xfbp120, angle=120)
                list180, val180, W = self.model(S180, Xfbp180, angle=180)


            if self.args.test_save_img == True:
                X_60 = [val60, Xgt, Xfbp60]
                dir_60 = [out_dir60, gt_dir60, input_dir60]
                save_image(n_count, dir_60, X_60)

                X_90 = [val90, Xgt, Xfbp90]
                dir_90 = [out_dir90, gt_dir90, input_dir90]
                save_image(n_count, dir_90, X_90)

                X_120 = [val120, Xgt, Xfbp120]
                dir_120 = [out_dir120, gt_dir120, input_dir120]
                save_image(n_count, dir_120, X_120)

                X_180 = [val180, Xgt, Xfbp180]
                dir_180 = [out_dir180, gt_dir180, input_dir180]
                save_image(n_count, dir_180, X_180)


            psnr_iter60 = self.aver_psnr(val60, Xgt)
            ssim_iter60 = self.aver_ssim(val60, Xgt)
            rmse_iter60 = torch.sqrt(torch.mean((val60 - Xgt) ** 2))
            psnr_iter90 = self.aver_psnr(val90, Xgt)
            ssim_iter90 = self.aver_ssim(val90, Xgt)
            rmse_iter90 = torch.sqrt(torch.mean((val90 - Xgt) ** 2))
            psnr_iter120 = self.aver_psnr(val120, Xgt)
            ssim_iter120 = self.aver_ssim(val120, Xgt)
            rmse_iter120 = torch.sqrt(torch.mean((val120 - Xgt) ** 2))
            psnr_iter180 = self.aver_psnr(val180, Xgt)
            ssim_iter180 = self.aver_ssim(val180, Xgt)
            rmse_iter180 = torch.sqrt(torch.mean((val180 - Xgt) ** 2))


            print(
                "\t image:{}  psnr:{:.4f},{:.4f},{:.4f},{:.4f}  ssim:{:.4f},{:.4f},{:.4f},{:.4f}  rmse:{:.4f},{:.4f},{:.4f},{:.4f} "
                .format(n_count, psnr_iter60, psnr_iter90, psnr_iter120, psnr_iter180,
                        ssim_iter60, ssim_iter90, ssim_iter120, ssim_iter180,
                        rmse_iter60, rmse_iter90, rmse_iter120,rmse_iter180))
            logger.info(
                "image:{}  psnr:{:.4f},{:.4f},{:.4f},{:.4f}  ssim:{:.4f},{:.4f},{:.4f},{:.4f}  rmse:{:.4f},{:.4f},{:.4f},{:.4f} "
                .format(n_count, psnr_iter60, psnr_iter90, psnr_iter120, psnr_iter180,
                        ssim_iter60, ssim_iter90, ssim_iter120, ssim_iter180,
                        rmse_iter60, rmse_iter90, rmse_iter120, rmse_iter180))



            psnr_iter60s += psnr_iter60.item()
            ssim_iter60s += ssim_iter60.item()
            rmse_iter60s += rmse_iter60.item()

            psnr_iter90s += psnr_iter90.item()
            ssim_iter90s += ssim_iter90.item()
            rmse_iter90s += rmse_iter90.item()

            psnr_iter120s += psnr_iter120.item()
            ssim_iter120s += ssim_iter120.item()
            rmse_iter120s += rmse_iter120.item()

            psnr_iter180s += psnr_iter180.item()
            ssim_iter180s += ssim_iter180.item()
            rmse_iter180s += rmse_iter180.item()

        print(100 * '*')



        psnr_iter60s /= n_count
        ssim_iter60s /= n_count
        rmse_iter60s /= n_count

        psnr_iter90s /= n_count
        ssim_iter90s /= n_count
        rmse_iter90s /= n_count

        psnr_iter120s /= n_count
        ssim_iter120s /= n_count
        rmse_iter120s /= n_count

        psnr_iter180s /= n_count
        ssim_iter180s /= n_count
        rmse_iter180s /= n_count

        avg_psnr = (psnr_iter60s + psnr_iter90s + psnr_iter120s + psnr_iter180s) / 4.0
        avg_ssim = (ssim_iter60s + ssim_iter90s + ssim_iter120s + ssim_iter180s) / 4.0
        avg_rmse = (rmse_iter60s + rmse_iter90s + rmse_iter120s + rmse_iter180s) / 4.0


        print('\t avg_psnr:{}  60:{}  90:{}  120:{} 180:{}'.format(avg_psnr, psnr_iter60s, psnr_iter90s, psnr_iter120s,
                                                            psnr_iter180s))
        print('\t avg_ssim:{}  60:{}  90:{}  120:{} 180:{}'.format(avg_ssim, ssim_iter60s, ssim_iter90s, ssim_iter120s,
                                                            ssim_iter180s))
        print('\t avg_rmse:{}  60:{}  90:{}  120:{} 180:{}'.format(avg_rmse, rmse_iter60s, rmse_iter90s, rmse_iter120s,
                                                            rmse_iter180s))
        print(100 * '*')

        print(100 * '*')

        logger.info("\t 4 avg_psnr:{:.4f}  avg_ssim:{:.4f}  avg_rmse:{:.4f}"
                    .format(avg_psnr, avg_ssim, avg_rmse))

        logger.info("\t 60 avg_psnr:{:.4f}  avg_ssim:{:.4f}  avg_rmse:{:.4f}"
                    .format(psnr_iter60s, ssim_iter60s, rmse_iter60s))


        logger.info("\t 90 avg_psnr:{:.4f}  avg_ssim:{:.4f}  avg_rmse:{:.4f}"
                    .format(psnr_iter90s, ssim_iter90s, rmse_iter90s))

        logger.info("\t 120 avg_psnr:{:.4f}  avg_ssim:{:.4f}  avg_rmse:{:.4f}"
                    .format(psnr_iter120s, ssim_iter120s, rmse_iter120s))

        logger.info("\t 180 avg_psnr:{:.4f}  avg_ssim:{:.4f}  avg_rmse:{:.4f}"
                    .format(psnr_iter180s, ssim_iter180s, rmse_iter180s))

    def psnr(self,img1, img2):
        if isinstance(img1, torch.Tensor):
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        psnr = skpsnr(img1.cpu().numpy(), img2.cpu().numpy(), data_range=1.0)
        return psnr

    def aver_psnr(self,img1, img2):
        PSNR = 0
        assert img1.size() == img2.size()
        for i in range(img1.size()[0]):
            for j in range(img1.size()[1]):
                PSNR += self.psnr(img1[i, j:j + 1, ...], img2[i, j:j + 1, ...])
        return PSNR / (img1.size()[0] * img1.size()[1])

    def aver_ssim(self,img1, img2):
        SSIM = 0
        img1 = img1.cpu().numpy().astype(np.float64)
        img2 = img2.cpu().numpy().astype(np.float64)
        for i in range(len(img1)):
            for j in range(img1.shape[1]):
                SSIM += skssim(img1[i, j, ...], img2[i, j, ...], gaussian_weights=True, win_size=11, data_range=1.0,
                               sigma=1.5) #
        return SSIM / (len(img1) * img1.shape[1])
