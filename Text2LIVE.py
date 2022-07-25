from sched import scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.transforms.functional import adjust_contrast
import albumentations as A
from albumentations.pytorch import ToTensorV2

import utils

from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import clip
import wandb
from config import CFG
import argparse
import sys
from template import imagenet_templates

from PIL import Image
import PIL

import EditNet

class Text2Live():
    def __init__(self, args):
        self.wandb = args.wandb
        self.save_weight = args.save_weight
        if torch.cuda.is_available():
            self.device="cuda"
        else:
            self.device="cpu"
        self.lr = CFG.lr
        self.content_path = CFG.content_path
        self.img_size = CFG.img_size
        self.text = CFG.text
        self.source = CFG.source
        self.step = CFG.step
        self.save_inference_path = CFG.save_inference_path


    def build_model(self):
        self.Net = EditNet.UNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.Net.parameters(), self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=60, gamma=0.5)
        self.VGG = models.vgg19(pretrained=True).features
        self.VGG.to(self.device)
        for parameter in self.VGG.parameters():
            parameter.requires_grad_(False)


    def train(self):
        #wandb
        if self.wandb:
            run = wandb.init(**CFG.wandb, settings=wandb.Settings(code_dir="."))
            wandb.watch(models=(self.Net), log_freq=10)

        with torch.no_grad():
            self.content_img = utils.load_image(self.content_path, self.img_size).to(self.device)
            content_features = utils.get_features(utils.img_normalize_vgg(self.content_img, self.device), self.VGG)

            clip_model, preprocess = clip.load("ViT-B/32", self.device, jit=False)
            template_text_target = utils.compose_text_with_templates(self.text)
            tokens_features = clip.tokenize(template_text_target).to(self.device)
            text_features = clip_model.encode_text(tokens_features).detach()
            text_features = text_features.mean(axis=0, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            template_text_source = utils.compose_text_with_templates(self.source)
            tokens_source = clip.tokenize(template_text_source).to(self.device)
            text_source = clip_model.encode_text(tokens_source).detach()
            text_source = text_source.mean(axis=0, keepdim=True)
            text_source /= text_source.norm(dim=-1, keepdim=True)

            source_features = clip_model.encode_image(utils.img_normalize_clip(self.content_img, self.device))
            source_features /= (source_features.clone().norm(dim=-1, keepdim=True))

        
        random_patching  = utils.get_transforms_patch()
        for step in range(CFG.step):
            self.scheduler.step()
            losses = {"total_loss":0, "patch_loss":0, "grobal_loss":0, "content_loss":0}
            self.optimizer.zero_grad()
            content_loss = 0
            patch_loss = 0

            self.Net.train()
            self.target = self.Net(self.content_img)
            self.target.requires_grad_(True)
            target_features = utils.get_features(utils.img_normalize_vgg(self.target, self.device), self.VGG)
            content_loss += torch.mean((target_features["conv4_2"]-content_features["conv4_2"])**2)
            content_loss += torch.mean((target_features["conv5_2"]-content_features["conv5_2"])**2)

            img_aug = []
            for n in range(self.crop_n):
                temp_patch = random_patching(self.target)
                img_aug.append(temp_patch)
            img_aug = torch.cat(img_aug, dim=0).to(self.device)
            img_features = clip_model.encode_image(utils.img_normalize_clip(img_aug, self.device))
            img_features /= img_features.clone().norm(dim=-1, keepdim=True)
            img_direction = img_features - source_features
            img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)
            
            text_direction = (text_features - text_source).repeat(img_direction.size(0), 1)
            text_direction /= text_direction.norm(dim=-1, keepdim=True)

            loss_temp = 1 - torch.cosine_similarity(img_direction, text_direction, dim=1)
            # loss_temp[loss_temp<self.patch_threshold] = 0
            patch_loss = loss_temp.mean()

            target_features = clip_model.encode_image(utils.img_normalize_clip(self.target, self.device))
            target_features /= (target_features.clone().norm(dim=-1, keepdim=True))
            grob_direction = target_features - source_features
            grob_direction /= grob_direction.clone().norm(dim=-1, keepdim=True)
            grob_loss = 1 - torch.cosine_similarity(grob_direction, text_direction, dim=1)
            grob_loss = grob_loss.mean()

            loss = self.lambda_grob*grob_loss + self.lambda_content*content_loss + self.lambda_patch*patch_loss
            loss.backward()
            self.optimizer.step()

            losses["total_loss"] = loss.item()
            losses["content_loss"] = content_loss.item()
            losses["grobal_loss"] = grob_loss.item()
            losses["patch_loss"] = patch_loss.item()
            if self.wandb:
                wandb.log(losses)

        # self.save()
        self.inference()
        if self.wandb:
            run.finish()


    def save_weight(self):
        torch.save(self.Net.state_dict(), CFG.MODEL_SAVE_PATH)


    def inference(self):
        fig,ax = plt.subplots(1,2)
        content_img_arr = self.content_img.detach().cpu().numpy()
        target_img_arr = adjust_contrast(self.target, 1.5).detach().cpu().numpy()
        content_img_arr = content_img_arr.squeeze(0).transpose(1,2,0)
        target_img_arr = target_img_arr.squeeze(0).transpose(1,2,0)
        ax[0].imshow(content_img_arr)
        ax[0].set_title("Original")
        ax[0].set_axis_off()
        ax[1].imshow(target_img_arr)
        ax[1].set_title(self.text)
        ax[1].set_axis_off()
        plt.savefig(self.save_inference_path)
