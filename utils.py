import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import cv2
from concurrent import futures
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from template import imagenet_templates
from PIL import Image
import PIL

from config import CFG

def load_image(path, img_size, plot_hist=False):
    img = Image.open(path)
    img = img.resize((img_size, img_size))
    img = np.array(img).transpose(2,0,1)
    img = torch.tensor(img/255).float()
    img = img.unsqueeze(0)
    if plot_hist:
        plt.hist(img.flatten())
        plt.savefig("./outs/load_image.png")
    return img

def get_transforms_patch(img_size=224, crop_size=128):
    trans = transforms.Compose([
        transforms.RandomCrop(crop_size),
        # transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
        transforms.Resize(img_size),
    ])
    return trans

def img_denormalize(img, device, plot_hist=False):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)
    img = img*std + mean
    if plot_hist:
        plt.hist(img.detach().cpu().numpy().flatten())
        plt.savefig("./outs/img_denormalize.png")
    return img


def img_normalize_vgg(img, device, plot_hist=False):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)
    img = (img-mean)/std
    if plot_hist:
        plt.hist(img.detach().cpu().numpy().flatten())
        plt.savefig("./outs/img_normalize.png")
    return img

def img_normalize_clip(img, device, plot_hist=False):
    img = F.interpolate(img, size=224, mode='bicubic')
    mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)
    img = (img-mean)/std
    if plot_hist:
        plt.hist(img.detach().cpu().numpy().flatten())
        plt.savefig("./outs/img_normalize.png")
    return img


def get_features(img, model, layers=None):
    if layers is None:
        layers = {
            "0":"conv1_1",
            "5":"conv2_1",
            "10":"conv3_1",
            "19":"conv4_1",
            "21":"conv4_2",
            "28":"conv5_1",
            "31":"conv5_2"
            }
    features = {}
    x = img
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def compose_text_with_templates(text, templates=imagenet_templates):
    return [template.format(text) for template in templates]
