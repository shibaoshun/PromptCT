import os
from glob import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy.random import RandomState
import odl
from skimage.measure import compare_psnr as skpsnr
import random
from skimage import transform
from odl.contrib import torch as odl_torch
import pydicom
from sklearn.cluster import k_means
import scipy.io as sio
import scipy

class Phantom2dDatasetval():
    def __init__(self, args, phase, datadir, length, angle):
        self.phase = phase
        self.angles = angle
        self.length = length
        self.img_size = args.img_size
        self.args = args
        self.base_path = datadir
        self.rand_state = RandomState(66)
        self.sp_files = glob(os.path.join(self.base_path, '*.npy'))

    def __len__(self):
        return self.length

    def __getitem__(self, ii):
        if 'npy' in self.sp_files[ii]:
            data = np.load(self.sp_files[ii], allow_pickle=True)
            phantom = data[0]
            fbpu = data[1]
            sino_noisy = data[4]
            zero_rows_count = np.sum(np.all(sino_noisy == 0, axis=1))
            total_rows = sino_noisy.shape[0]
            if np.isclose((total_rows - zero_rows_count) / total_rows, 1 / 6):
                angle = int(60)
            elif np.isclose((total_rows - zero_rows_count) / total_rows, 1 / 4):
                angle = int(90)
            elif np.isclose((total_rows - zero_rows_count) / total_rows, 1 / 3):
                angle = int(120)
            elif np.isclose((total_rows - zero_rows_count) / total_rows, 1 / 2):
                angle = int(180)
            else:
                angle = None

            phantom = torch.from_numpy(phantom)
            phantom = phantom.unsqueeze(0)
            phantom = phantom.type(torch.FloatTensor)
            fbpu = torch.from_numpy(fbpu)
            fbpu = fbpu.unsqueeze(0)
            fbpu = fbpu.type(torch.FloatTensor)
            sino_noisy = torch.from_numpy(sino_noisy)
            sino_noisy = sino_noisy.unsqueeze(0)
            sino_noisy = sino_noisy.type(torch.FloatTensor)

        return phantom, fbpu, sino_noisy, angle

class Phantom2dDatasettrain():
    def __init__(self, args, phase, datadir, length, angle):
        self.phase = phase
        self.angles = angle
        self.length = length
        self.img_size = args.img_size
        self.args = args
        self.base_path = datadir
        self.rand_state = RandomState(66)
        self.sp_files = glob(os.path.join(self.base_path, '*.npy'))

    def __len__(self):
        return self.length

    def __getitem__(self, ii):
        if 'npy' in self.sp_files[ii]:
            data = np.load(self.sp_files[ii], allow_pickle=True)
            phantom = data[0]
            fbpu = data[1]
            sino_noisy = data[4]
            zero_rows_count = np.sum(np.all(sino_noisy == 0, axis=1))
            total_rows = sino_noisy.shape[0]
            if np.isclose((total_rows - zero_rows_count) / total_rows, 1 / 6):
                angle = int(60)
            elif np.isclose((total_rows - zero_rows_count) / total_rows, 1 / 4):
                angle = int(90)
            elif np.isclose((total_rows - zero_rows_count) / total_rows, 1 / 3):
                angle = int(120)
            elif np.isclose((total_rows - zero_rows_count) / total_rows, 1 / 2):
                angle = int(180)
            else:
                angle = None

            phantom = torch.from_numpy(phantom)
            phantom = phantom.unsqueeze(0)
            phantom = phantom.type(torch.FloatTensor)
            fbpu = torch.from_numpy(fbpu)
            fbpu = fbpu.unsqueeze(0)
            fbpu = fbpu.type(torch.FloatTensor)
            sino_noisy = torch.from_numpy(sino_noisy)
            sino_noisy = sino_noisy.unsqueeze(0)
            sino_noisy = sino_noisy.type(torch.FloatTensor)

        return phantom, fbpu, sino_noisy, angle

class Phantom2dDatasettest():
    def __init__(self, args, phase, datadir, length):
        self.phase = phase
        self.args = args
        self.base_path = datadir
        self.length = length
        self.sp_file = glob(os.path.join(self.base_path, '*.IMA'))
        self.radon, self.iradon, self.fbp, self.op_norm = self.radon_transform(num_view=360)

    def __len__(self):

        return self.length

    def radon_transform(self, num_view=120):
        xx = 200
        space = odl.uniform_discr([-xx, -xx], [xx, xx], [self.args.img_size[0], self.args.img_size[1]], dtype='float32')
        if num_view == 360 or self.args.mode == 'sparse':
            angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)
        elif self.args.mode == 'limited':
            angles = np.array(self.args.sino_size[0]).astype(int)
            angle_partition = odl.uniform_partition(0, 2 / 3 * np.pi, angles)

        detectors = np.array(self.args.sino_size[1]).astype(int)
        detector_partition = odl.uniform_partition(-480, 480, detectors)
        geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=600,
                                            det_radius=290)  # FanBeamGeometry
        operator = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

        op_norm = odl.operator.power_method_opnorm(operator)
        op_norm = torch.from_numpy(np.array(op_norm * 2 * np.pi)).double().cuda()

        op_layer = odl_torch.operator.OperatorModule(operator)
        op_layer_adjoint = odl_torch.operator.OperatorModule(operator.adjoint)
        fbp = odl.tomo.fbp_op(operator, filter_type='Ram-Lak', frequency_scaling=0.9) * np.sqrt(2)
        op_layer_fbp = odl_torch.operator.OperatorModule(fbp)

        return op_layer, op_layer_adjoint, op_layer_fbp, op_norm

    def __getitem__(self, ii):

        if 'IMA' in self.sp_file[ii]:
            dcm = pydicom.read_file(self.sp_file[ii])
            dcm.image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
            data = dcm.image
            data = np.array(data).astype(float)
            data = transform.resize(data, (self.args.img_size))
            data = (data - np.min(data)) / (np.max(data) - np.min(data))

        data = torch.from_numpy(data)
        Xgt = data.unsqueeze(0)
        sino = self.radon(Xgt)

        # -----------
        # add Poisson noise
        intensityI0 = self.args.poiss_level
        scale_value = torch.from_numpy(np.array(intensityI0).astype(np.float))
        normalized_sino = torch.exp(-sino / sino.max())
        th_data = np.random.poisson(scale_value * normalized_sino)
        sino_noisy = -torch.log(torch.from_numpy(th_data) / scale_value)
        sino_noisy = sino_noisy * sino.max()

        # add Gaussian noise
        noise_std = self.args.gauss_level
        noise_std = np.array(noise_std).astype(np.float)
        if self.args.mode == 'limited':
            nx, ny = np.array(self.args.sino_size[0]).astype(np.int), np.array(self.args.sino_size[1]).astype(np.int)
        if self.args.mode == 'sparse':
            nx, ny = np.array(360), np.array(self.args.sino_size[1]).astype(np.int)
        noise = noise_std * np.random.randn(nx, ny)
        noise = torch.from_numpy(noise)
        sino_noisy = sino_noisy + noise

        s60 = sino_noisy.clone()
        s60[:, 1:361:6, :] = 0
        s60[:, 2:362:6, :] = 0
        s60[:, 3:363:6, :] = 0
        s60[:, 4:364:6, :] = 0
        s60[:, 5:365:6, :] = 0
        Snoisy360_60 = s60
        Xfbp60 = self.fbp(Snoisy360_60)

        s90 = sino_noisy.clone()
        s90[:, 1:361:4, :] = 0
        s90[:, 2:362:4, :] = 0
        s90[:, 3:363:4, :] = 0
        Snoisy360_90 = s90
        Xfbp90 = self.fbp(Snoisy360_90)

        s120 = sino_noisy.clone()
        s120[:, 1:361:3, :] = 0
        s120[:, 2:362:3, :] = 0
        Snoisy360_120 = s120
        Xfbp120 = self.fbp(Snoisy360_120)

        s180 = sino_noisy.clone()
        s180[:, 1:361:2, :] = 0
        Snoisy360_180 = s180
        Xfbp180 = self.fbp(Snoisy360_180)

        Xgt = Xgt.type(torch.FloatTensor)
        Xfbp120 = Xfbp120.type(torch.FloatTensor)
        Snoisy360_120 = Snoisy360_120.type(torch.FloatTensor)

        Xfbp60 = Xfbp60.type(torch.FloatTensor)
        Snoisy360_60 = Snoisy360_60.type(torch.FloatTensor)

        Xfbp90 = Xfbp90.type(torch.FloatTensor)
        Snoisy360_90 = Snoisy360_90.type(torch.FloatTensor)

        Xfbp180 = Xfbp180.type(torch.FloatTensor)
        Snoisy360_180 = Snoisy360_180.type(torch.FloatTensor)

        return Xgt, Xfbp60, Snoisy360_60, Xfbp90, Snoisy360_90, Xfbp120, Snoisy360_120, Xfbp180, Snoisy360_180

