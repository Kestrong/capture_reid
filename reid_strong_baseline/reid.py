import datetime
import logging
import os
from collections import defaultdict

import PIL
import numpy as np
import torch
import torchvision.transforms as T
from torch import nn

from cdp.cdp import cluster
from reid_strong_baseline.senet import SENet, SEBottleneck, SEResNeXtBottleneck, SEResNetBottleneck
from reid_strong_baseline.resnet import ResNet, Bottleneck, BasicBlock
from reid_strong_baseline.resnet_ibn_a import resnet50_ibn_a
from util.global_lock import mx_lock

logger = logging.getLogger("app")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    m, n = input1.size(0), input2.size(0)
    distmat = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, input1, input2.t())
    return distmat.cpu().numpy()


def transform_img(transforms, imgs, device):
    result = []
    if transforms is not None:
        for img in imgs:
            if type(img) is np.ndarray:
                img = PIL.Image.fromarray(img.astype(np.uint8)).convert('RGB')
            img = transforms(img)
            result.append(img)
    imgs = torch.stack(result, dim=0)
    if torch.cuda.is_available():
        imgs = imgs.to(device)
    return imgs


def build_transforms(cfg):
    normalize_transform = T.Normalize(mean=cfg['INPUT']['PIXEL_MEAN'], std=cfg['INPUT']['PIXEL_STD'])
    transform = T.Compose([
        T.Resize(cfg['INPUT']['SIZE_TEST']),
        T.ToTensor(),
        normalize_transform
    ])
    return transform


def default_cfg():
    cfg = {
        "MODEL": {
            "LAST_STRIDE": 1,
            "NECK": "bnneck",
            "NAME": "se_resnext101"
        },
        "TEST": {
            "WEIGHT": "./reid_model.pth",
            "FEAT_NORM": "yes"
        },
        "INPUT": {
            "PIXEL_MEAN": [0.485, 0.456, 0.406],
            "PIXEL_STD": [0.229, 0.224, 0.225],
            "SIZE_TEST": [384, 128]
        },
        "DEVICE": 0
    }
    return cfg


class Reid(nn.Module):
    in_planes = 2048
    _instance_lock = mx_lock

    @classmethod
    def instance(cls, cfg=None):
        if hasattr(Reid, "_instance"):
            return Reid._instance
        with Reid._instance_lock:
            if hasattr(Reid, "_instance"):
                return Reid._instance
            reid = Reid(cfg)
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 1:
                    reid = torch.nn.DataParallel(reid)
                reid.to(reid.device)
            Reid._instance = reid
        return Reid._instance

    def __init__(self, cfg):
        super(Reid, self).__init__()
        if cfg is None:
            cfg = default_cfg()
            model_name = cfg['MODEL']['NAME']
            last_stride = cfg['MODEL']['LAST_STRIDE']
            if model_name == 'resnet18':
                self.in_planes = 512
                self.base = ResNet(last_stride=last_stride,
                                   block=BasicBlock,
                                   layers=[2, 2, 2, 2])
            elif model_name == 'resnet34':
                self.in_planes = 512
                self.base = ResNet(last_stride=last_stride,
                                   block=BasicBlock,
                                   layers=[3, 4, 6, 3])
            elif model_name == 'resnet50':
                self.base = ResNet(last_stride=last_stride,
                                   block=Bottleneck,
                                   layers=[3, 4, 6, 3])
            elif model_name == 'resnet101':
                self.base = ResNet(last_stride=last_stride,
                                   block=Bottleneck,
                                   layers=[3, 4, 23, 3])
            elif model_name == 'resnet152':
                self.base = ResNet(last_stride=last_stride,
                                   block=Bottleneck,
                                   layers=[3, 8, 36, 3])

            elif model_name == 'se_resnet50':
                self.base = SENet(block=SEResNetBottleneck,
                                  layers=[3, 4, 6, 3],
                                  groups=1,
                                  reduction=16,
                                  dropout_p=None,
                                  inplanes=64,
                                  input_3x3=False,
                                  downsample_kernel_size=1,
                                  downsample_padding=0,
                                  last_stride=last_stride)
            elif model_name == 'se_resnet101':
                self.base = SENet(block=SEResNetBottleneck,
                                  layers=[3, 4, 23, 3],
                                  groups=1,
                                  reduction=16,
                                  dropout_p=None,
                                  inplanes=64,
                                  input_3x3=False,
                                  downsample_kernel_size=1,
                                  downsample_padding=0,
                                  last_stride=last_stride)
            elif model_name == 'se_resnet152':
                self.base = SENet(block=SEResNetBottleneck,
                                  layers=[3, 8, 36, 3],
                                  groups=1,
                                  reduction=16,
                                  dropout_p=None,
                                  inplanes=64,
                                  input_3x3=False,
                                  downsample_kernel_size=1,
                                  downsample_padding=0,
                                  last_stride=last_stride)
            elif model_name == 'se_resnext50':
                self.base = SENet(block=SEResNeXtBottleneck,
                                  layers=[3, 4, 6, 3],
                                  groups=32,
                                  reduction=16,
                                  dropout_p=None,
                                  inplanes=64,
                                  input_3x3=False,
                                  downsample_kernel_size=1,
                                  downsample_padding=0,
                                  last_stride=last_stride)
            elif model_name == 'se_resnext101':
                self.base = SENet(block=SEResNeXtBottleneck,
                                  layers=[3, 4, 23, 3],
                                  groups=32,
                                  reduction=16,
                                  dropout_p=None,
                                  inplanes=64,
                                  input_3x3=False,
                                  downsample_kernel_size=1,
                                  downsample_padding=0,
                                  last_stride=last_stride)
            elif model_name == 'senet154':
                self.base = SENet(block=SEBottleneck,
                                  layers=[3, 8, 36, 3],
                                  groups=64,
                                  reduction=16,
                                  dropout_p=0.2,
                                  last_stride=last_stride)
            elif model_name == 'resnet50_ibn_a':
                self.base = resnet50_ibn_a(last_stride)

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.neck = cfg['MODEL']['NECK']
        self.feat_norm = cfg['TEST']['FEAT_NORM']

        if self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift

            self.bottleneck.apply(weights_init_kaiming)

        self.transforms = build_transforms(cfg)
        self.load_param(cfg['TEST']['WEIGHT'])
        self.device = cfg['DEVICE']

    def forward(self, x):
        """
        :param x:img
        :type PIL image array
        :return:
        """
        if not isinstance(x, torch.Tensor):
            x = transform_img(self.transforms, x, self.device)

        with torch.no_grad():
            global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
            global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

            if self.neck == 'bnneck':
                global_feat = self.bottleneck(global_feat)  # normalize for angular softmax
            if self.feat_norm == 'yes':
                global_feat = nn.functional.normalize(global_feat, dim=1, p=2)

        return global_feat.cpu().numpy()

    def load_param(self, trained_path):
        if not os.path.exists(trained_path):
            logger.info("reid model init fail:%s", trained_path)
            raise ValueError("reid model not yet init")
        logger.info('Loading pretrained reid model begin......')
        param_dict = torch.load(trained_path)
        logger.info('Loading pretrained reid model end......')
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def distance_cluster(self, feats_all, pids, threshold=0.5):
        """
        :param feats_all: all features
        :type torch.Tensor
        :param threshold:
        :type float
        :return: group  {'label':[index_of_pids]}
        """
        if feats_all is None or len(feats_all) == 0:
            return {}
        time1 = datetime.datetime.now()
        pre_labels = cluster(features=feats_all, th_knn=threshold, max_size=300, labels=[-1] * pids.__len__())
        groups = defaultdict(list)
        for i in range(len(feats_all)):
            groups[pre_labels[i]].append(pids[i])
        if -1 in groups.keys():  # remove invalid group
            groups.pop(-1)
        logger.info("get groups cost:%s", (datetime.datetime.now() - time1).total_seconds())
        return groups


if __name__ == "__main__":
    print(torch.__version__, torch.cuda.is_available())
    from PIL import Image

    model = Reid.instance()
    model.eval()
    img = Image.open("/home/lijc08/deeplearning/Data/ReID/dukemtmc-reid/DukeMTMC-reID/query1/0005_c2_f0046985.jpg").convert('RGB')
    img2 = Image.open("/home/lijc08/deeplearning/Data/ReID/dukemtmc-reid/DukeMTMC-reID/bounding_box_test/0005_c2_f0047105.jpg").convert('RGB')
    img3 = Image.open("/home/lijc08/deeplearning/Data/ReID/dukemtmc-reid/DukeMTMC-reID/bounding_box_test/0005_c2_f0047105.jpg").convert('RGB')
    img4 = Image.open("/home/lijc08/deeplearning/Data/ReID/dukemtmc-reid/DukeMTMC-reID/bounding_box_test/5856_c8_f0005425.jpg").convert("RGB")
    feat = model([img, img2, img3, img4])
    print(feat.shape, feat)
    group = model.distance_cluster(feats_all=feat, pids=[1, 2, 3, 4], threshold=0.3)
    print(group)
