import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import odl
from odl.contrib import torch as odl_torch
import scipy.io
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_, DropPath


def radon_transform():
    xx = 200
    space = odl.uniform_discr([-xx, -xx], [xx, xx], [512,512], dtype='float32')
    angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)
    detectors = np.array(800).astype(int)
    detector_partition = odl.uniform_partition(-480, 480, detectors)
    geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=600,det_radius=290)  # FanBeamGeometry
    operator = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

    op_norm = odl.operator.power_method_opnorm(operator)
    op_norm = torch.from_numpy(np.array(op_norm * 2 * np.pi)).double().cuda()

    op_layer = odl_torch.operator.OperatorModule(operator)
    op_layer_adjoint = odl_torch.operator.OperatorModule(operator.adjoint)
    fbp = odl.tomo.fbp_op(operator, filter_type='Ram-Lak', frequency_scaling=0.9) * np.sqrt(2)
    op_layer_fbp = odl_torch.operator.OperatorModule(fbp)

    return op_layer, op_layer_adjoint, op_layer_fbp, op_norm

proj, back_proj, _, op_norm = radon_transform()

def radon(img,angle):
    if len(img.shape) == 4:
        img = img.squeeze(1)
        sino = proj(img)
    if angle == 60:
        sino[:, 1:361:6, :] = 0
        sino[:, 2:362:6, :] = 0
        sino[:, 3:363:6, :] = 0
        sino[:, 4:364:6, :] = 0
        sino[:, 5:365:6, :] = 0
    elif angle == 90:
        sino[:, 1:361:4, :] = 0
        sino[:, 2:362:4, :] = 0
        sino[:, 3:363:4, :] = 0
    elif angle == 120:
        sino[:, 1:361:3, :] = 0
        sino[:, 2:362:3, :] = 0
    elif angle == 180:
        sino[:, 1:361:2, :] = 0
    else:
        raise ValueError(f"Unsupported angle: {angle}")
    return sino.unsqueeze(1)

def iradon(sino):
    if len(sino.shape) == 4:
        sino = sino.squeeze(1)
    #sino360 = buling(sino).cuda()
    img = back_proj(sino)   #back_proj(sino / op_norm)
    return img.unsqueeze(1)

def rot180(d):  # 转置
    ###  the filtersize must be a 奇数
    d = d.permute(1, 0, 2, 3)
    filtersize = d.shape[3]
    a = d.clone()
    for i in range(1, (filtersize + 1)):
        a[:, :, (i - 1), :] = d[:, :, (filtersize - i), :]
    c = a.clone()
    for i in range(1, (filtersize + 1)):
        c[:, :, :, (i - 1)] = a[:, :, :, (filtersize - i)]
    return c

class SoftThreshold(nn.Module):
    ####### perfect soft function
    def __init__(self, size, init_threshold=1e-1):
        super(SoftThreshold, self).__init__()
        self.threshold = nn.Parameter(init_threshold * torch.ones(1,size,1,1))
        # self.temp= torch.ones(1, size, 1, 1).cuda()
        # self.size =size
    def forward(self, x, threshold):
        # threshold = nn.Parameter(self.threshold.cuda() *self.temp)
        mask1 = (x > threshold).float()
        mask2 = (x < -threshold).float()
        out = mask1.float() * (x - threshold)
        out += mask2.float() * (x + threshold)
        return out

def my_epsilon_hat(c, sigma):
    N, C, H, W = c.size()
    sigma_map = sigma.view(N, 1, 1, 1).repeat(1, C, H, W)
    out = c * sigma_map
    return out

"""STB+SFB"""
class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        # sqaure validation
        # assert h_windows == w_windows

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        # negative is allowed
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]

class STBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size,
                 drop_path=0., type='W', input_resolution=256):
        """ SwinTransformer Block
        """
        super(STBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'

        # print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x

class FourierUnit(nn.Module):
    def __init__(self, embed_dim, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.conv_layer = torch.nn.Conv2d(embed_dim * 2, embed_dim * 2, 1, 1, 0)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(ffted)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4,
                                                                       2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        return output

class SpectralTransform(nn.Module):
    def __init__(self, embed_dim, last_conv=False):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.last_conv = last_conv

        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.fu = FourierUnit(embed_dim // 2)

        self.conv2 = torch.nn.Conv2d(embed_dim // 2, embed_dim, 1, 1, 0)

        if self.last_conv:
            self.last_conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        output = self.fu(x)
        output = self.conv2(x + output)
        if self.last_conv:
            output = self.last_conv(output)
        return output

class ResB(nn.Module):
    def __init__(self, embed_dim, red=1):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // red, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim // red, embed_dim, 3, 1, 1),
        )

    def __call__(self, x):
        out = self.body(x)
        return out + x

class SFB(nn.Module):
    def __init__(self, embed_dim, red=1):
        super(SFB, self).__init__()
        self.S = ResB(embed_dim, red)
        self.F = SpectralTransform(embed_dim)
        self.fusion = nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0)

    def __call__(self, x):
        s = self.S(x)
        f = self.F(x)
        out = torch.cat([s, f], dim=1)
        out = self.fusion(out)
        return out

class CNet_Conv2D_RCAB_STB(nn.Module):
    def __init__(self, in_ch=81, out_ch=81, num_STBs=1,
                fea_RCAB=81, head_dim=27, window_size=8, drop_path=0.):
        super(CNet_Conv2D_RCAB_STB, self).__init__()

        self.conv_low = nn.Sequential(
            nn.Conv2d(in_ch, fea_RCAB, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_ch, fea_RCAB, 3, 1, 1, bias=True))

        self.STBs = nn.Sequential(
            *[STBlock(fea_RCAB, fea_RCAB, head_dim, window_size, drop_path, 'W' if not i%2 else 'SW') \
            for i in range(num_STBs)])
        self.SFB = SFB(fea_RCAB, red=1)
        self.F_ext_net1 = F_ext(in_nc=1, nf=fea_RCAB)
        self.prompt_scale1 = nn.Linear(fea_RCAB, fea_RCAB, bias=True)
        self.base_nf = fea_RCAB
        self.conv_last = nn.Conv2d(fea_RCAB, out_ch, 3, 1, 1, bias=True)

    def forward(self, Wx, angle_vector):

        # prompt-learning
        prompt = angle_vector
        prompt1 = self.F_ext_net1(prompt)
        scale1 = self.prompt_scale1(prompt1)
        # low-level
        feat = self.conv_low(Wx)
        feat = feat * scale1.view(-1, self.base_nf, 1, 1) + Wx
        x2 = feat
        # high-level
        x2 = Rearrange('b c h w -> b h w c')(x2)
        x3 = self.STBs(x2)
        x3 = Rearrange('b h w c -> b c h w')(x3)
        x2 = Rearrange('b h w c -> b c h w')(x2)
        feat = self.SFB(x3)
        x4 = feat
        #zuihou
        x5 = self.conv_last(x4+x2)
        return x5, prompt1

"""Prompt-learning"""
class F_ext(nn.Module):
    def __init__(self, in_nc=1, nf=81):
        super(F_ext, self).__init__()
        stride = 2
        pad = 0
        self.pad = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(in_nc, nf, 7, stride, pad, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.conv3 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1_out = self.act(self.conv1(self.pad(x)))
        conv2_out = self.act(self.conv2(self.pad(conv1_out)))
        conv3_out = self.act(self.conv3(self.pad(conv2_out)))
        out = torch.mean(conv3_out, dim=[2, 3], keepdim=False)

        return out

class SingleTFCnet(nn.Module):
    def __init__(self):
        super(SingleTFCnet, self).__init__()
        self.kernel_size = 9
        self.num_filters = self.kernel_size ** 2
        # self.C = nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
        self.thresholding = CNet_Conv2D_RCAB_STB()
        self.conv_w = nn.Conv2d(1, self.num_filters, self.kernel_size, stride=1, padding=0, bias=False)  ### 卷积不做padding
        self.soft_threshold = SoftThreshold(self.num_filters, 1)
        self.rpad = nn.ReflectionPad2d(self.kernel_size // 2)  ### 镜像填# weight initial

    def setdecfilter(self):
        mat_dict = scipy.io.loadmat('globalTF9.mat')
        tframedict = mat_dict['learnt_dict'].transpose()
        self.conv_w.weight.data = torch.Tensor(tframedict).cuda().view(self.conv_w.weight.shape).permute(0, 1, 3,
                                                                                                         2).contiguous()

    def forward(self, input, angle_vector, sigma):
        Wx = self.conv_w(self.rpad(input))
        Constantnet, prompt1 = self.thresholding(Wx, angle_vector)
        constant = torch.clamp(Constantnet, 0., 10.)  ### 0<C<10
        epsilon_hat = my_epsilon_hat(constant, sigma)
        z = self.soft_threshold(Wx, epsilon_hat)
        weightT = rot180(self.conv_w.weight)  ###  输出通道数 输入通道数  滤波器大小
        Wt = F.conv2d(self.rpad(z), weightT, stride=1, padding=0) / (self.kernel_size ** 2)  # 7*7
        return Wt, self.conv_w.weight, prompt1

"""DU-Single"""
class DU_STF(nn.Module):
    def __init__(self):
        super(DU_STF, self).__init__()

        self.SingleTFCnet1 = SingleTFCnet()
        self.SingleTFCnet2 = SingleTFCnet()

        self.beta1 = nn.Parameter(torch.FloatTensor([0.1]), requires_grad=True)
        self.rho1 = nn.Parameter(torch.FloatTensor([0.9]), requires_grad=True)

        self.beta2 = nn.Parameter(torch.FloatTensor([0.01]), requires_grad=True)
        self.rho2 = nn.Parameter(torch.FloatTensor([0.99]), requires_grad=True)

        self.tao1 = nn.Parameter(torch.FloatTensor([0.8]), requires_grad=True)


    def make_net(self, iters):
        layers = []
        for i in range(iters):
            layers.append(self.SingleTFCnet)
        return nn.Sequential(*layers)

    def make_body(self, iters):
        layers = []
        for i in range(iters):
            layers.append(self.body_tao)
        return nn.Sequential(*layers)


    def forward(self, input, angle_vector):
        sigma = torch.cuda.FloatTensor([10 / 255])
        x1, W, prompt1 = self.SingleTFCnet1(input, angle_vector, sigma)
        x1 = self.beta1 * input + self.rho1 * x1
        tao1 = torch.clamp(self.tao1, 0., 1.)  ### 0<tao<1
        sigma1 = tao1 * sigma
        x2, W, prompt2 = self.SingleTFCnet2(x1, angle_vector, sigma1)
        out = self.beta2 * input + self.rho2 * x2
        return out, W, prompt2

"""PromptCT"""
class Gradient_Descent(nn.Module):
    def __init__(self, args):
        super(Gradient_Descent, self).__init__()
        self.args = args
        self.etaconst = torch.tensor(self.args.eta).float()
        self.eta = nn.Parameter(data=self.etaconst, requires_grad=True)
        self.T = self.args.layers
        self.CNN = DU_STF().cuda()
        self.CNN.SingleTFCnet1.setdecfilter()
        self.CNN.SingleTFCnet2.setdecfilter()
        self.proxNet = self.make_net(self.T)


    def make_net(self, iters):
        layers = []
        for j in range(iters):
            layers.append(self.CNN)
        return nn.Sequential(*layers)

    def forward(self, sino, fbpu, angle):
        Listout = []
        x = fbpu
        angle_vector = create_prompt(angle)
        for i in range(self.T):
            res = sino - radon(x, angle)
            grad1 = iradon(res)
            x = x + self.eta * grad1
            x, W, prompt = self.proxNet[i](x, angle_vector)
            Listout.append(x)
        return Listout, x, W

def create_prompt(angle):
    prompt = torch.ones((1, 1, 360, 800), dtype=torch.float32).cuda()
    if angle == 60:
        prompt[:, :, 1:361:6, :] = 0
        prompt[:, :, 2:362:6, :] = 0
        prompt[:, :, 3:363:6, :] = 0
        prompt[:, :, 4:364:6, :] = 0
        prompt[:, :, 5:365:6, :] = 0
        return prompt
    elif angle == 90:
        prompt[:, :, 1:361:4, :] = 0
        prompt[:, :, 2:362:4, :] = 0
        prompt[:, :, 3:363:4, :] = 0
        return prompt
    elif angle == 120:
        prompt[:, :, 1:361:3, :] = 0
        prompt[:, :, 2:362:3, :] = 0
        return prompt
    elif angle == 180:
        prompt[:, :, 1:361:2, :] = 0
        return prompt
    else:
        raise ValueError(f"Unsupported angle: {angle}")
