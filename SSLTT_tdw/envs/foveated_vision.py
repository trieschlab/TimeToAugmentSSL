"""
This work is based on the code written by Zhibo Yang.
The original version is available at
https://github.com/ouyangzhibo/Image_Foveation_Python
For more information go to
https://www3.cs.stonybrook.edu/~zhibyang/
"""

import cv2
import numpy as np
import sys

import torch
from PIL import Image


class FoveatedVision(torch.nn.Module):
    """
    FoveatedVision is a torchvision transform that picks a random focus point
    and procedurally blurs the input image outwards.
    """

    def __init__(self, sigma=0.248, prNum=6, p=7.5, k=3, alpha=2.5):
        super().__init__()
        self.sigma = sigma
        self.prNum = prNum
        # compute coefficients
        self.p = p
        self.k = k
        self.alpha = alpha

    def __repr__(self):
        return self.__class__.__name__ + ''

    def forward(self, img):
        img = np.array(img)

        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # xc, yc = np.random.randint(0,img.shape[0],1), np.random.randint(0,img.shape[1],1)
        # draw from a gaussian to focus more on center of image
        xc, yc = np.clip(np.int(np.random.normal(img.shape[0] // 2, img.shape[0] // 4)), 0, img.shape[0]), np.clip(
            np.int(np.random.normal(img.shape[1] // 2, img.shape[1] // 4)), 0, img.shape[1])
        # xc, yc = img.shape[0] // 2, img.shape[0] // 2
        img = self.foveat_img(img, [(xc, yc)], self.p, self.k, self.alpha)
        if len(img.shape) == 2:
            img = img[:, :, 0]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # return Image.fromarray(img)
        return img

    def genGaussiankernel(self, width, sigma):
        x = np.arange(-int(width / 2), int(width / 2) + 1, 1, dtype=np.float32)
        x2d, y2d = np.meshgrid(x, x)
        kernel_2d = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
        kernel_2d = kernel_2d / np.sum(kernel_2d)
        return kernel_2d

    def pyramid(self, im, sigma=1, prNum=6):
        height_ori, width_ori, ch = im.shape
        G = im.copy()
        pyramids = [G]

        # gaussian blur
        Gaus_kernel2D = self.genGaussiankernel(5, sigma)

        # downsample
        for i in range(1, prNum):
            G = cv2.filter2D(G, -1, Gaus_kernel2D)
            height, width = G.shape[0], G.shape[1]
            G = cv2.resize(G, (int(width / 2), int(height / 2)))
            pyramids.append(G)

        # upsample
        for i in range(1, prNum):
            curr_im = pyramids[i]
            for j in range(i):
                if j < i - 1:
                    im_size = (curr_im.shape[1] * 2, curr_im.shape[0] * 2)
                else:
                    im_size = (width_ori, height_ori)
                curr_im = cv2.resize(curr_im, im_size)
                curr_im = cv2.filter2D(curr_im, -1, Gaus_kernel2D)
            pyramids[i] = curr_im

        return pyramids

    def foveat_img(self, im, fixs, p, k, alpha):
        """
        im: input image
        fixs: sequences of fixations of form [(x1, y1), (x2, y2), ...]
        defaults: p = 7.5, k = 3, alpha = 2.5
        This function outputs the foveated image with given input image and fixations.
        """
        sigma = self.sigma
        prNum = self.prNum
        As = self.pyramid(im, sigma, prNum)
        height, width, channels = im.shape

        x = np.arange(0, width, 1, dtype=np.float32)
        y = np.arange(0, height, 1, dtype=np.float32)
        x2d, y2d = np.meshgrid(x, y)
        theta = np.sqrt((x2d - fixs[0][0]) ** 2 + (y2d - fixs[0][1]) ** 2) / p
        for fix in fixs[1:]:
            theta = np.minimum(theta, np.sqrt((x2d - fix[0]) ** 2 + (y2d - fix[1]) ** 2) / p)
        R = alpha / (theta + alpha)

        Ts = []
        for i in range(1, prNum):
            Ts.append(np.exp(-((2 ** (i - 3)) * R / sigma) ** 2 * k))
        Ts.append(np.zeros_like(theta))

        # omega
        omega = np.zeros(prNum)
        for i in range(1, prNum):
            omega[i - 1] = np.sqrt(np.log(2) / k) / (2 ** (i - 3)) * sigma

        omega[omega > 1] = 1

        # layer index
        layer_ind = np.zeros_like(R)
        for i in range(1, prNum):
            ind = np.logical_and(R >= omega[i], R <= omega[i - 1])
            layer_ind[ind] = i

        # B
        Bs = []
        for i in range(1, prNum):
            Bs.append((0.5 - Ts[i]) / (Ts[i - 1] - Ts[i] + 1e-5))

        # M
        Ms = np.zeros((prNum, R.shape[0], R.shape[1]))

        for i in range(prNum):
            ind = layer_ind == i
            if np.sum(ind) > 0:
                if i == 0:
                    Ms[i][ind] = 1
                else:
                    Ms[i][ind] = 1 - Bs[i - 1][ind]

            ind = layer_ind - 1 == i
            if np.sum(ind) > 0:
                Ms[i][ind] = Bs[i][ind]

        # print('[INFO] num of full-res pixel', np.sum(Ms[0] == 1))
        # generate periphery image
        im_fov = np.zeros_like(As[0], dtype=np.float32)
        for M, A in zip(Ms, As):
            A = np.expand_dims(A, -1) if len(A.shape) == 2 else A
            for i in range(channels):
                im_fov[:, :, i] += np.multiply(M, A[:, :, i])

        im_fov = im_fov.astype(np.uint8)
        return im_fov
