import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import random
import copy
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

def get_transforms(img_size):
    trans = transforms.Compose([
        # transforms.RandomResizedCrop(size=img_size, ratio=(0.85, 0.95), ),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomHorizontalFlip(),
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

    
def calc_clip_similarity_matrix(img_tokens, device):
    K = img_tokens.shape[1]
    cosine_similarity_matrix = torch.zeros(size=(1, K, K)).to(device) #shape=(1,K,K)
    for i in range(K):
        for j in range(K):
            cosine_similarity_matrix[0,i,j] = 1 - F.cosine_similarity(img_tokens[:,i,:], img_tokens[:,j,:])
    return cosine_similarity_matrix

def compose_text_with_templates(text, templates=imagenet_templates):
    rand = torch.randint(high=len(templates)-1, size=(1,))
    template_text = templates[rand].format(text)
    return template_text
