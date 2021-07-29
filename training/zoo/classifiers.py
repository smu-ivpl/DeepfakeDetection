from functools import partial

import numpy as np
import torch
from timm.models import skresnext50_32x4d
from timm.models.dpn import dpn92, dpn131
from timm.models.efficientnet import tf_efficientnet_b4_ns, tf_efficientnet_b3_ns, \
    tf_efficientnet_b5_ns, tf_efficientnet_b2_ns, tf_efficientnet_b6_ns, tf_efficientnet_b7_ns
from timm.models.vision_transformer import vit_small_patch16_224, vit_base_patch16_224, \
    vit_base_patch16_384, vit_base_patch32_384, vit_large_patch16_224, vit_large_patch16_384, \
    vit_large_patch32_384, vit_huge_patch16_224, vit_huge_patch32_384
from timm.models.senet import legacy_seresnext50_32x4d
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from facebook_deit import deit_base_patch16_224, deit_distill_large_patch16_384, deit_distill_large_patch32_384
#from taming_transformer import Decoder, VUNet, ActNorm
import functools
#from vit_pytorch.distill import DistillableViT, DistillWrapper, DistillableEfficientViT
import re

encoder_params = {
    "deit_distill_large_patch32_384":{
        "features": 1024,
        "init_op":partial(deit_distill_large_patch32_384, pretrained=True, drop_path_rate=0.2)
    },
    "deit_distill_large_patch16_384":{
        "features": 1024,
        "init_op":partial(deit_distill_large_patch16_384, pretrained=True, drop_path_rate=0.2)
    },
    "deit_base_patch16_224":{
        "features": 768,
        "init_op":partial(deit_base_patch16_224, pretrained=True, drop_path_rate=0.2)
    },
    "vit_huge_patch16_224": {
        "features": 1280,
        "init_op": partial(vit_huge_patch16_224, pretrained=True, drop_path_rate=0.2)
    },
    "vit_huge_patch32_384": {
        "features": 1280,
        "init_op": partial(vit_huge_patch32_384, pretrained=True, drop_path_rate=0.2)
    },
    "vit_large_patch32_384": {
        "features": 1024,
        "init_op": partial(vit_large_patch32_384, pretrained=True, drop_path_rate=0.2)
    },
    "vit_large_patch16_384": {
        "features": 1024,
        "init_op": partial(vit_large_patch16_384, pretrained=True, drop_path_rate=0.2)
    },
    "vit_large_patch16_224": {
        "features": 1024,
        "init_op": partial(vit_large_patch16_224, pretrained=True, drop_path_rate=0.2)
    },
    "vit_base_patch_16_384": {
        "features": 768,
        "init_op": partial(vit_base_patch16_384, pretrained=True, drop_path_rate=0.2)
    },
    "vit_base_patch_32_384": {
        "features": 768,
        "init_op": partial(vit_base_patch32_384, pretrained=True, drop_path_rate=0.2)
    },
    "dpn92": {
        "features": 2688,
        "init_op": partial(dpn92, pretrained=True)
    },
    "dpn131": {
        "features": 2688,
        "init_op": partial(dpn131, pretrained=True)
    },
    "tf_efficientnet_b3_ns": {
        "features": 1536,
        "init_op": partial(tf_efficientnet_b3_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b2_ns": {
        "features": 1408,
        "init_op": partial(tf_efficientnet_b2_ns, pretrained=False, drop_path_rate=0.2)
    },
    "tf_efficientnet_b4_ns": {
        "features": 1792,
        "init_op": partial(tf_efficientnet_b4_ns, pretrained=True, drop_path_rate=0.5)
    },
    "tf_efficientnet_b5_ns": {
        "features": 2048,
        "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b4_ns_03d": {
        "features": 1792,
        "init_op": partial(tf_efficientnet_b4_ns, pretrained=True, drop_path_rate=0.3)
    },
    "tf_efficientnet_b5_ns_03d": {
        "features": 2048,
        "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.3)
    },
    "tf_efficientnet_b5_ns_04d": {
        "features": 2048,
        "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.4)
    },
    "tf_efficientnet_b6_ns": {
        "features": 2304,
        "init_op": partial(tf_efficientnet_b6_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b7_ns": {
        "features": 2560,
        "init_op": partial(tf_efficientnet_b7_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b6_ns_04d": {
        "features": 2304,
        "init_op": partial(tf_efficientnet_b6_ns, pretrained=True, drop_path_rate=0.4)
    },
    "se50": {
        "features": 2048,
        "init_op": partial(legacy_seresnext50_32x4d, pretrained=True)
    },
    "sk50": {
        "features": 2048,
        "init_op": partial(skresnext50_32x4d, pretrained=True)
    },
}

def setup_srm_weights(input_channels: int = 3) -> torch.Tensor:
    """Creates the SRM kernels for noise analysis."""
    # note: values taken from Zhou et al., "Learning Rich Features for Image Manipulation Detection", CVPR2018
    srm_kernel = torch.from_numpy(np.array([
        [  # srm 1/2 horiz
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
            [0., 1., -2., 1., 0.],  # noqa: E241,E201
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
        ], [  # srm 1/4
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
            [0., -1., 2., -1., 0.],  # noqa: E241,E201
            [0., 2., -4., 2., 0.],  # noqa: E241,E201
            [0., -1., 2., -1., 0.],  # noqa: E241,E201
            [0., 0., 0., 0., 0.],  # noqa: E241,E201
        ], [  # srm 1/12
            [-1., 2., -2., 2., -1.],  # noqa: E241,E201
            [2., -6., 8., -6., 2.],  # noqa: E241,E201
            [-2., 8., -12., 8., -2.],  # noqa: E241,E201
            [2., -6., 8., -6., 2.],  # noqa: E241,E201
            [-1., 2., -2., 2., -1.],  # noqa: E241,E201
        ]
    ])).float()
    srm_kernel[0] /= 2
    srm_kernel[1] /= 4
    srm_kernel[2] /= 12
    return srm_kernel.view(3, 1, 5, 5).repeat(1, input_channels, 1, 1)


def setup_srm_layer(input_channels: int = 3) -> torch.nn.Module:
    """Creates a SRM convolution layer for noise analysis."""
    weights = setup_srm_weights(input_channels)
    conv = torch.nn.Conv2d(input_channels, out_channels=3, kernel_size=5, stride=1, padding=2, bias=False)
    with torch.no_grad():
        conv.weight = torch.nn.Parameter(weights, requires_grad=False)
    return conv


class DeepFakeClassifierSRM(nn.Module):
    def __init__(self, encoder, dropout_rate=0.5) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.srm_conv = setup_srm_layer(3)
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params[encoder]["features"], 1)

    def forward(self, x):
        noise = self.srm_conv(x)
        x = self.encoder.forward_features(noise)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class GlobalWeightedAvgPool2d(nn.Module):
    """
    Global Weighted Average Pooling from paper "Global Weighted Average
    Pooling Bridges Pixel-level Localization and Image-level Classification"
    """

    def __init__(self, features: int, flatten=False):
        super().__init__()
        self.conv = nn.Conv2d(features, 1, kernel_size=1, bias=True)
        self.flatten = flatten

    def fscore(self, x):
        m = self.conv(x)
        m = m.sigmoid().exp()
        return m

    def norm(self, x: torch.Tensor):
        return x / x.sum(dim=[2, 3], keepdim=True)

    def forward(self, x):
        input_x = x
        x = self.fscore(x)
        x = self.norm(x)
        x = x * input_x
        x = x.sum(dim=[2, 3], keepdim=not self.flatten)
        return x


class DeepFakeClassifier(nn.Module):
    def __init__(self, encoder, dropout_rate=0.0) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params[encoder]["features"], 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, channel = 3, n_strided=6):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channel, 64, 4, 2, 1, bias=False), #384 -> 192
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  #192->96
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), # 96->48
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), #48->24
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False), #24->12
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 4, 2, 1, bias=False), #12->6
        )
        self.last = nn.Sequential(
            #(B, 6*6)
            nn.Linear(6*6, 1),
            #nn.Sigmoid()
        )

        def discriminator_block(in_filters, out_filters):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1), nn.LeakyReLU(0.01)]
            return layers

        layers = discriminator_block(channel, 32)
        curr_dim = 32
        for _ in range(n_strided-1):
            layers.extend(discriminator_block(curr_dim, curr_dim*2))
            curr_dim *= 2
        layers.extend(discriminator_block(curr_dim,curr_dim))
        self.model = nn.Sequential(*layers)
        self.out1 = nn.Conv2d(curr_dim, 1, 3, stride=1, padding=0, bias=False)
    def forward(self, x):
        #x = self.main(x).view(-1,6*6)
        feature_repr = self.model(x)
        x = self.out1(feature_repr)
        return x.view(-1, 1)#self.last(x)

##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Pre_training(nn.Module):
    def __init__(self, encoder, channel=3, res_blocks=5, dropout_rate=0.0, patch_size=16) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"]()
        self.emb_ch = encoder_params[encoder]["features"]

        '''
        self.teacher = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to("cuda")
        checkpoint = torch.load('weights/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36', map_location='cpu')
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.teacher.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)
        '''
        '''
        self.deconv = nn.Sequential(
            nn.Conv2d(self.emb_ch, self.emb_ch//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.emb_ch // 2),
            nn.ReLU(True),
            nn.Conv2d(self.emb_ch//2, self.emb_ch //4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.emb_ch //4),
            nn.ReLU(True),
        )
        '''
        '''
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.emb_ch, self.emb_ch//2 , kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.emb_ch//2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.emb_ch//2, self.emb_ch // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.emb_ch // 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.emb_ch//4, self.emb_ch // 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.emb_ch // 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.emb_ch//8, channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        '''
        #self.deconv = nn.ConvTranspose2d(self.emb_ch, 3, kernel_size=16, stride=16)
        #self.decoder = Decoder(double_z = False, z_channels = 1024, resolution= 384, in_channels=3, out_ch=3, ch=64
        #                       , ch_mult=[1,1,2,2], num_res_blocks = 0, attn_resolutions=[16], dropout=0.0)
        #nn.ConvTranspose2d(encoder_params[encoder]["features"], channel, kernel_size=patch_size, stride=patch_size)
        channels = self.emb_ch
        model = [
            nn.ConvTranspose2d(channels, channels, 7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(channels, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        ]
        curr_dim = channels

        for _ in range(2):
            model+=[
                nn.ConvTranspose2d(curr_dim, curr_dim//2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim //= 2

        #Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(curr_dim)]
        #Upsampling
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(curr_dim, curr_dim//2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim = curr_dim //2
        #output layer
        model += [nn.Conv2d(curr_dim, channel, 7, stride=1, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*model)
        self.fc = Linear(encoder_params[encoder]["features"], 1)
        self.dropout = Dropout(dropout_rate)
    '''
    def generator(self, x, freeze):
        if freeze:
            with torch.no_grad():
                _, z = self.encoder.pre_training(x)
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            #with torch.enable_grad():
            for param in self.encoder.parameters():
                param.requires_grad = True
            _, z = self.encoder.pre_training(x)
        x = self.model(z)
        return x
    def discriminator(self, x ,freeze):
        if freeze:
            with torch.no_grad():
                cls_token, _ = self.encoder.pre_training(x)
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            #with torch.enable_grad():
            for param in self.encoder.parameters():
                param.requires_grad = True
            cls_token, _ = self.encoder.pre_training(x)
        x = self.dropout(cls_token)
        cls = self.fc(x)
        return cls
    '''
    def get_class(self,x):
        for param in self.teacher.parameters():
            param.requires_grad = False
        teacher_logits = self.teacher(x)
        return teacher_logits

    def forward(self, x):
        cls_token, z = self.encoder.pre_training(x)
        #with torch.no_grad():
        #    teacher_logits = self.teacher(x)
        #x = self.deconv(x)
        #x = self.decoder(x)
        #cls = self.dropout(cls_token)
        #cls_token = self.fc(cls)

        x = self.model(z)
        return x#, cls_token, teacher_logits#, labels
class DeepFakeClassifier_Distill(nn.Module):
    def __init__(self, encoder, dropout_rate=0.0) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"]()
        #'''0524
        self.teacher = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to("cuda")
        checkpoint = torch.load('/home/yjheo/Deepfake/dfdc_deepfake_challenge/weights/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36', map_location='cpu')
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.teacher.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)
        #'''
        self.backbone = encoder_params["tf_efficientnet_b7_ns"]["init_op"]()
        self.temperature = 1.
        self.alpha = 0.5
        self.bce = nn.BCEWithLogitsLoss()
        #assert (isinstance(self.encoder,(DistillableViT, DistillableEfficientViT)))
        self.avg_pool = AdaptiveAvgPool2d((None,1024))
        dim = encoder_params[encoder]["features"] #self.encoder.dim
        num_classes = self.encoder.num_classes
        self.distill_token = nn.Parameter(torch.zeros(1, 1, dim)) #0524
        self.fc = Linear(encoder_params[encoder]["features"], 1)
        self.dropout = Dropout(dropout_rate)
        #self.proj = nn.Conv2d(3, encoder_params[encoder]["features"], kernel_size=32, stride=32) #0524 eye
        #self.pos_embed = nn.Parameter(torch.zeros(1, 1, encoder_params[encoder]["features"])) #0524 eye
    def forward(self, x): #eye
        b, *_ = x.shape
        #'''0524
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        #'''

        #backbone
        feature = self.backbone.forward_features(x)
        feature = feature.flatten(2).transpose(1, 2)
        '''#eye
        eye = self.proj(eye)
        eye = eye.flatten(2).transpose(1,2)
        '''#
        cls, distill = self.encoder.forward_features(x, feature, self.distill_token, self.avg_pool)
        #cls, distill = self.encoder.forward_features(x, eye, feature, self.distill_token, self.avg_pool, self.pos_embed) #0524 eye, pos_embed
        #cls = self.encoder.forward_features(x, feature, self.avg_pool)
        cls = self.dropout(cls)
        cls = self.fc(cls)

        #'''0524
        distill = self.dropout(distill)
        distill = self.fc(distill)
        #'''

        return cls, distill,teacher_logits #, teacher_logits#student_logits, distill_logits, teacher_logits #loss * alpha + distill_loss * (1 - alpha)

class DeepFakeClassifier_Video_Distill(nn.Module):
    def __init__(self, encoder, dropout_rate=0.0) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"]()
        self.teacher = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to("cuda")
        checkpoint = torch.load('weights/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36', map_location='cpu')
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.teacher.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)
        self.backbone = encoder_params["tf_efficientnet_b7_ns"]["init_op"]()
        self.temperature = 1.
        self.alpha = 0.5
        self.bce = nn.BCEWithLogitsLoss()
        self.avg_pool = AdaptiveAvgPool2d((None,1))
        self.avg_pool2 = AdaptiveAvgPool2d((None, 1024))
        dim = encoder_params[encoder]["features"] #self.encoder.dim
        self.distill_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.fc = Linear(encoder_params[encoder]["features"], 1)
        self.dropout = Dropout(dropout_rate)

        self.pos_embed = nn.Parameter(torch.zeros(1, 15 + 2, dim))
        self.pos_drop = nn.Dropout(p=0.0)

    def forward(self, x):
        b, *_, w, h = x.shape
        x = x.view(-1, 3, w, h)
        if len(x) > 15:
            x = x[:15, :, :, :]  # video
        elif len(x) < 15:
            temp = x[-1].repeat(15 - len(x), 1, 1).view(-1, 3, 384, 384)
            x = torch.cat([x, temp], dim=0)  # 42
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        #backbone
        x = self.backbone.forward_features(x)
        x = x.flatten(2)
        x = self.avg_pool(x)
        x = x.permute(2,0,1)
        x = self.avg_pool2(x)
        cls, distill = self.encoder.forward_for_video(x, self.distill_token, self.pos_embed, self.pos_drop)
        cls = self.dropout(cls)
        cls = self.fc(cls)

        distill = self.dropout(distill)
        distill = self.fc(distill)

        fakes = torch.count_nonzero(torch.sigmoid(teacher_logits) > 0.8)
        # 11 frames are detected as fakes with high probability
        with torch.no_grad():
            if fakes >= 7:
                teacher_logits = torch.mean(teacher_logits[torch.sigmoid(teacher_logits) > 0.8])
            elif torch.count_nonzero(torch.sigmoid(teacher_logits) < 0.2) >= 10:
                teacher_logits = torch.mean(teacher_logits[torch.sigmoid(teacher_logits) < 0.2])
            else:
                teacher_logits = torch.mean(teacher_logits)

        return distill #cls, distill, teacher_logits.view(1,1)

class DeepFakeClassifierGWAP(nn.Module):
    def __init__(self, encoder, dropout_rate=0.5) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"]()
        self.avg_pool = GlobalWeightedAvgPool2d(encoder_params[encoder]["features"])
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params[encoder]["features"], 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x