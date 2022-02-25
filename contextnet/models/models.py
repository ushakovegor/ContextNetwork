import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from contextnet.utils.losses import GaussianFocalLoss, MSELoss
from contextnet.utils.utils import WBF
import torch.nn.functional as F
from endoanalysis.targets import KeypointsBatch, Keypoints, keypoints_list_to_batch
import segmentation_models_pytorch as smp
import torchvision.models.segmentation as seg_models


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Block(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, identity_downsample=None, stride=1):
        assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"
        super(Block, self).__init__()
        self.num_layers = num_layers
        if self.num_layers > 34:
            self.expansion = 4
        else:
            self.expansion = 1
        # ResNet50, 101, and 152 include additional layer of 1x1 kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if self.num_layers > 34:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        if self.num_layers > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, num_layers, block, image_channels):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 18, 34, 50, 101, or 152 '
        super(ResNet, self).__init__()
        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels*self.expansion))
        layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion # 256
        for i in range(num_residual_blocks - 1):
            layers.append(block(num_layers, self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)


class CTResNetNeck(nn.Module):
    """The neck used in `CenterNet <https://arxiv.org/abs/1904.07850>`_ for
    object classification and box regression.
    Args:
         in_channel (int): Number of input channels.
         num_deconv_filters (tuple[int]): Number of filters per stage.
         num_deconv_kernels (tuple[int]): Number of kernels per stage.
         use_dcn (bool): If True, use DCNv2. Default: True.
         init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channel,
                 num_deconv_filters,
                 num_deconv_kernels,
                 use_dcn=True):
        assert len(num_deconv_filters) == len(num_deconv_kernels)
        super(CTResNetNeck, self).__init__()

        self.fp16_enabled = False
        self.use_dcn = use_dcn
        self.in_channel = in_channel
        self.deconv_layers = self._make_deconv_layer(num_deconv_filters,
                                                     num_deconv_kernels)

    def _make_deconv_layer(self, num_deconv_filters, num_deconv_kernels):
        """use deconv layers to upsample backbone's output."""
        layers = []
        for i in range(len(num_deconv_filters)):
            feat_channel = num_deconv_filters[i]
            conv_module = ConvModule(
                self.in_channel,
                feat_channel,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN'))
            layers.append(conv_module)
            upsample_module = ConvModule(
                feat_channel,
                feat_channel,
                num_deconv_kernels[i],
                stride=2,
                padding=1,
                conv_cfg=dict(type='deconv'),
                norm_cfg=dict(type='BN'))
            layers.append(upsample_module)
            self.in_channel = feat_channel

        return nn.Sequential(*layers)
    
    def forward(self, inputs):
        outs = self.deconv_layers(inputs)
        return outs

class ContextNetHead(nn.Module):
    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 loss_heatmap=GaussianFocalLoss(),
                 loss_weight=1.0):
        super(ContextNetHead, self).__init__()
        self.num_classes = num_classes
        self.heatmap_head = self._build_head(in_channel, feat_channel,
                                             num_classes)

        self.loss_heatmap = loss_heatmap

    def _build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def forward(self, x):
        """
        """
        center_heatmap_pred = self.heatmap_head(x).sigmoid()
        # offset_pred = self.offset_head(x)
        return center_heatmap_pred

    def get_loss(self, pd_heatmaps, gt_heatmaps):
        """Compute losses of the head.
        """
        loss_center_heatmap = self.loss_heatmap(pd_heatmaps, gt_heatmaps)
        return loss_center_heatmap


class ContextNet(nn.Module):
    def __init__(self, img_channels=3, num_classes=2, img_shape=None, kp_shape=None):
        super(ContextNet, self).__init__()
        self.backbone = seg_models.fcn_resnet50(pretrained=True)
        self.backbone.classifier = ContextNetHead(2048, 512, num_classes=num_classes)
        self.backbone.aux_classifier = nn.Identity()
        # self.backbone.avgpool = Identity()
        # self.backbone.fc = Identity()
        # self.neck = CTResNetNeck(2048, num_deconv_filters=(512, 256, 128, 64), num_deconv_kernels=(3, 3, 3, 3))
        # self.head = ContextNetHead(64, 64, num_classes=num_classes)
        print(self.backbone)
        if img_shape is not None:
            self.img_shape = img_shape
        else:
            self.img_shape = (600, 600)
        
        if kp_shape is not None:
            self.kp_shape = kp_shape
        else:
            self.kp_shape = (200, 200)
        

    def forward(self, x):
        x = self.backbone(x)
        # x = self.neck(x['out'])
        # x = self.head(x)
        return x['out']
    
    def get_loss(self, pd_heatmaps, gt_heatmaps):
        loss = self.backbone.classifier.get_loss(pd_heatmaps, gt_heatmaps)
        return self.compute_loss(loss)

    def compute_loss(self, loss):
        return loss

    def get_keypoints(self, pd_heatmaps, rescales=None, with_wbf=False):
        # scale_factors = [img_meta['scale_factor'] for img_meta in img_metas]

        batch_bboxes, batch_scores = self._decode_heatmap(pd_heatmaps)
        batch_bboxes = keypoints_list_to_batch(batch_bboxes)
        batch_scores = batch_scores.reshape(-1)
        if with_wbf:
            det_results = []
            for det_keypoints in zip(batch_bboxes, batch_scores): # PLS FIX IT. NEED TO UPDATE KEYPOINTS STACKING FOR BATCH
                det_bbox = WBF(det_keypoints)
                det_results.append(det_bbox)
        else:
            det_results = batch_bboxes, batch_scores
        return det_results

    def _decode_heatmap(self, center_heatmap_pred, kernel=3, k=400):
        """
        Transform heatmaps into predicted keypoints.
        """
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = self.img_shape
        out_h, out_w = self.kp_shape

        center_heatmap_pred = self._get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = self._get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_topk_labels = batch_dets

        x = topk_xs * (out_w / width)
        y = topk_ys * (out_h / height)

        batch_bboxes = torch.stack([x, y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_topk_labels[..., None]),
                                 dim=-1)
        batch_bboxes = batch_bboxes.numpy()
        batch_scores = batch_scores.numpy()
        return batch_bboxes, batch_scores
    

    def _get_local_maximum(self, heat, kernel=3):
        """Extract local maximum pixel with given kernal.
        Args:
            heat (Tensor): Target heatmap.
            kernel (int): Kernel size of max pooling. Default: 3.
        Returns:
            heat (Tensor): A heatmap where local maximum pixels maintain its
                own value and other positions are 0.
        """
        pad = (kernel - 1) // 2
        hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep


    def _get_topk_from_heatmap(self, scores, k=500):
        """Get top k positions from heatmap.
        Args:
            scores (Tensor): Target heatmap with shape
                [batch, num_classes, height, width].
            k (int): Target number. Default: 20.
        Returns:
            tuple[torch.Tensor]: Scores, indexes, categories and coords of
                topk keypoint. Containing following Tensors:
            - topk_scores (Tensor): Max scores of each topk keypoint.
            - topk_inds (Tensor): Indexes of each topk keypoint.
            - topk_clses (Tensor): Categories of each topk keypoint.
            - topk_ys (Tensor): Y-coord of each topk keypoint.
            - topk_xs (Tensor): X-coord of each topk keypoint.
        """
        batch, _, height, width = scores.size()
        topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
        topk_clses = topk_inds // (height * width)
        topk_inds = topk_inds % (height * width)
        topk_ys = topk_inds // width
        topk_xs = (topk_inds % width).int().float()
        return topk_scores, topk_clses, topk_ys, topk_xs


class ContextNetTry2(nn.Module):
    def __init__(self, img_channels=3, num_classes=2, img_shape=None, kp_shape=None):
        super(ContextNetTry2, self).__init__()
        self.backbone = smp.Unet(in_channels=img_channels,
                                classes=num_classes)
        # print(self.backbone)
        # self.backbone.segmentation_head = ContextNetHead(16, 16, num_classes=num_classes)
        if img_shape is not None:
            self.img_shape = img_shape
        else:
            self.img_shape = (512, 512)
        
        if kp_shape is not None:
            self.kp_shape = kp_shape
        else:
            self.kp_shape = (512, 512)
        

    def forward(self, x):
        x = self.backbone(x)
        # x = self.head(x)
        return x


# class Unet(SegmentationModel):
#     """Unet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder*
#     and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial
#     resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *concatenation*
#     for fusing decoder blocks with skip connections.

#     Args:
#         encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
#             to extract features of different spatial resolution
#         encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
#             two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
#             with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
#             Default is 5
#         encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
#             other pretrained weights (see table with available weights for each encoder_name)
#         decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
#             Length of the list should be the same as **encoder_depth**
#         decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
#             is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
#             Available options are **True, False, "inplace"**
#         decoder_attention_type: Attention module used in decoder of the model. Available options are
#             **None** and **scse** (https://arxiv.org/abs/1808.08127).
#         in_channels: A number of input channels for the model, default is 3 (RGB images)
#         classes: A number of classes for output mask (or you can think as a number of channels of output mask)
#         activation: An activation function to apply after the final convolution layer.
#             Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
#                 **callable** and **None**.
#             Default is **None**
#         aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
#             on top of encoder if **aux_params** is not **None** (default). Supported params:
#                 - classes (int): A number of classes
#                 - pooling (str): One of "max", "avg". Default is "avg"
#                 - dropout (float): Dropout factor in [0, 1)
#                 - activation (str): An activation function to apply "sigmoid"/"softmax"
#                     (could be **None** to return logits)

#     Returns:
#         ``torch.nn.Module``: Unet

#     .. _Unet:
#         https://arxiv.org/abs/1505.04597

#     """

#     def __init__(
#         self,
#         encoder_name = "resnet34",
#         encoder_depth = 5,
#         encoder_weights = "imagenet",
#         decoder_use_batchnorm = True,
#         decoder_channels = (256, 128, 64, 32, 16),
#         decoder_attention_type = None,
#         in_channels = 3,
#         classes = 1,
#         activation = None,
#         aux_params = None,
#     ):
#         super().__init__()

#         self.encoder = get_encoder(
#             encoder_name,
#             in_channels=in_channels,
#             depth=encoder_depth,
#             weights=encoder_weights,
#         )

#         self.decoder = UnetDecoder(
#             encoder_channels=self.encoder.out_channels,
#             decoder_channels=decoder_channels,
#             n_blocks=encoder_depth,
#             use_batchnorm=decoder_use_batchnorm,
#             center=True if encoder_name.startswith("vgg") else False,
#             attention_type=decoder_attention_type,
#         )

#         self.segmentation_head = SegmentationHead(
#             in_channels=decoder_channels[-1],
#             out_channels=classes,
#             activation=activation,
#             kernel_size=3,
#         )