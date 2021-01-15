import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BlazeBlock, self).__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        # TFLite uses slightly different padding than PyTorch
        # on the depthwise conv layer when the stride is 2.
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=in_channels, bias=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.act = nn.PReLU(num_parameters=out_channels)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x  # 构造

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(self.convs(h) + x)


class face_mesh(nn.Module):
    """The BlazeFace face detection model from MediaPipe.

    The version from MediaPipe is simpler than the one in the paper;
    it does not use the "double" BlazeBlocks.

    Because we won't be training this model, it doesn't need to have
    batchnorm layers. These have already been "folded" into the conv
    weights by TFLite.

    The conversion to PyTorch is fairly straightforward, but there are
    some small differences between TFLite and PyTorch in how they handle
    padding on conv layers with stride 2.

    This version works on batches, while the MediaPipe version can only
    handle a single image at a time.

    Based on code from https://github.com/tkat0/PyTorch_BlazeFace/ and
    https://github.com/google/mediapipe/
    """

    def __init__(self):
        super(face_mesh, self).__init__()

        # These are the settings from the MediaPipe example graph
        # mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt
        self.score_clipping_thresh = 100.0
        self.min_score_thresh = 0.75

        self._define_layers()

    def _define_layers(self):
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                      stride=2, padding=0, bias=True),
            nn.PReLU(num_parameters=16),
            BlazeBlock(16, 16, kernel_size=3),
            BlazeBlock(16, 16, kernel_size=3),
        )
        self.stage2 = nn.Sequential(
            BlazeBlock(16, 32, kernel_size=3, stride=2),
            BlazeBlock(32, 32, kernel_size=3),
            BlazeBlock(32, 32, kernel_size=3),
        )
        self.stage3 = nn.Sequential(
            BlazeBlock(32, 64, kernel_size=3, stride=2),
            BlazeBlock(64, 64, kernel_size=3),
            BlazeBlock(64, 64, kernel_size=3),
        )
        self.stage4 = nn.Sequential(
            BlazeBlock(64, 128, kernel_size=3, stride=2),
            BlazeBlock(128, 128, kernel_size=3),
            BlazeBlock(128, 128, kernel_size=3),
        )
        self.stage5 = nn.Sequential(
            BlazeBlock(128, 128, kernel_size=3, stride=2),
            BlazeBlock(128, 128, kernel_size=3),
            BlazeBlock(128, 128, kernel_size=3),
        )
        self.stage6 = nn.Sequential(
            BlazeBlock(128, 128, kernel_size=3, stride=2),
            BlazeBlock(128, 128, kernel_size=3),
            BlazeBlock(128, 128, kernel_size=3),
            nn.Conv2d(in_channels=128, out_channels=32,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.PReLU(num_parameters=32),
            BlazeBlock(32, 32, kernel_size=3),
            nn.Conv2d(in_channels=32, out_channels=1404,
                      kernel_size=3, stride=3, padding=0, bias=True),
        )
        self.stage7 = nn.Sequential(
            BlazeBlock(128, 128, kernel_size=3, stride=2),
            nn.Conv2d(in_channels=128, out_channels=32,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.PReLU(num_parameters=32),
            BlazeBlock(32, 32, kernel_size=3),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3,
                      stride=3, padding=0, bias=True),
        )

        #self.handflag = nn.Conv2d(192, 1, kernel_size=2)
        #self.ld_21_3d = nn.Conv2d(192, 63, kernel_size=2)

    def forward(self, x):
        # TFLite uses slightly different padding on the first conv layer
        # than PyTorch, so do it manually.
        x = F.pad(x, (1, 2, 1, 2), "constant", 0)
        b = x.shape[0]      # batch size, needed for reshaping later

        x = self.stage1(x)           # (b, 32, 128, 128)
        print('stage1 shape: ', x.shape)
        x = self.stage2(x)           # (b, 64, 64, 64)
        print('stage2 shape: ', x.shape)
        x = self.stage3(x)           # (b, 128, 32, 32)
        print('stage3 shape: ', x.shape)
        x = self.stage4(x)           # (b, 192, 16, 16)
        print('stage4 shape: ', x.shape)
        x = self.stage5(x)           # (b, 192, 8, 8)
        print('stage5 shape: ', x.shape)

    def _device(self):
        """Which device (CPU or GPU) is being used by this model?"""
        return self.classifier_8.weight.device

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def _preprocess(self, x):
        """Converts the image pixels to the range [-1, 1]."""
        return x.float() / 127.5 - 1.0
