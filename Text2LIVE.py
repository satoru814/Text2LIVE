from re import template
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
from CLIP import clip
import wandb
from config import CFG
import argparse
import sys
from template import imagenet_templates

from PIL import Image
import PIL

import EditNet
import madgrad

class Text2LIVE():
    def __init__(self, args):
        self.wandb = args.wandb
        self.save_weight = args.save_weight
        self.log_picture = args.log_picture
        if torch.cuda.is_available():
            self.device="cuda"
        else:
            self.device="cpu"
        self.lr = CFG.lr
        self.content_path = CFG.content_path
        self.img_size = CFG.img_size
        self.text = CFG.text
        self.screen = CFG.screen
        self.ROI = CFG.ROI
        self.step = CFG.step
        self.save_inference_path = CFG.save_inference_path
        self.lambda_composition = CFG.lambda_composition
        self.lambda_screen = CFG.lambda_screen
        self.lambda_structure = CFG.lambda_structure
        self.optimizer_params = CFG.optimizer_params


    def build_model(self):
        self.Net = EditNet.UNet().to(self.device)
        self.optimizer = madgrad.MADGRAD(self.Net.parameters(), **self.optimizer_params)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

    def train(self):
        #wandb
        if self.wandb:
            run = wandb.init(**CFG.wandb, settings=wandb.Settings(code_dir="."))
            wandb.watch(models=(self.Net), log_freq=10)

        with torch.no_grad():
            self.content_img = utils.load_image(self.content_path, self.img_size).to(self.device)
            self.green_img = torch.zeros_like(self.content_img)
            self.green_img[0,1,:,:] = torch.ones(size=(self.content_img.shape[2], self.content_img.shape[3])).to(self.device)
            clip_model, preprocess = clip.load("ViT-B/32", self.device, jit=False)
            # template_text_roi = self.ROI
            tokens_roi = clip.tokenize(self.ROI).to(self.device)
            text_features_roi = clip_model.encode_text(tokens_roi).detach()
            text_features_roi /= text_features_roi.norm(dim=-1, keepdim=True)

            # template_text_screen = self.screen
            tokens_screen = clip.tokenize(self.screen).to(self.device)
            text_features_screen = clip_model.encode_text(tokens_screen).detach()
            text_features_screen /= text_features_screen.norm(dim=-1, keepdim=True)

        augmentation = utils.get_transforms(self.img_size,)
        for step in range(CFG.step):
            print(step)
            self.scheduler.step()
            losses = {"total_loss":0, "composition_loss":0, "screen_loss":0, "structure_loss":0}
            self.optimizer.zero_grad()
            loss = 0
            screen_loss = 0
            structure_loss = 0
            composition_loss = 0


            self.content_img_aug = augmentation(self.content_img)
            source_features = clip_model.encode_image(
                utils.img_normalize_clip(self.content_img_aug, self.device)
                )
            source_features /= (source_features.clone().norm(dim=-1, keepdim=True))
            template_text_target = utils.compose_text_with_templates(self.text)
            tokens_features = clip.tokenize(template_text_target).to(self.device)
            text_features = clip_model.encode_text(tokens_features).detach()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_direction = text_features - text_features_roi
            text_direction /= text_direction.norm(dim=-1, keepdim=True)

            #Input to Net
            self.Net.train()
            edit_colormap, edit_opacity = self.Net(self.content_img_aug)

            #Extract features
            #self.requires_grad_(True)
            self.editedimg = edit_opacity*edit_colormap + (1-edit_opacity)*self.content_img_aug
            out_features = clip_model.encode_image(utils.img_normalize_clip(self.editedimg, self.device))
            out_features /= out_features.clone().norm(dim=-1, keepdim=True)
            img_direction = out_features - source_features
            img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

            #Composition loss
            cosine_loss = 1 - torch.cosine_similarity(out_features, text_features)
            direction_loss = 1 - torch.cosine_similarity(img_direction, text_direction)
            composition_loss = cosine_loss + direction_loss

            #Screen loss
            self.editedimg_screen = edit_opacity*edit_colormap + (1-edit_opacity)*self.green_img
            out_features_screen = clip_model.encode_image(
                utils.img_normalize_clip(self.editedimg_screen, self.device),
                )
            out_features_screen /= out_features_screen.clone().norm(dim=-1, keepdim=True)
            screen_loss = 1 - torch.cosine_similarity(out_features_screen, text_features_screen)

            #Structure loss
            source_tokens = clip_model.encode_image(
                utils.img_normalize_clip(self.content_img_aug, self.device), 
                return_spatial_tokens=True,
                )
            out_tokens = clip_model.encode_image(
                utils.img_normalize_clip(self.editedimg, self.device), 
                return_spatial_tokens=True,
                )
            source_similarity_matrix = utils.calc_clip_similarity_matrix(source_tokens, self.device)
            source_similarity_matrix /= source_similarity_matrix.clone().norm(dim=-1, keepdim=True)
            out_similarity_matrix = utils.calc_clip_similarity_matrix(out_tokens, self.device)
            out_similarity_matrix /= out_similarity_matrix.clone().norm(dim=-1, keepdim=True)
            # print(source_similarity_matrix.shape)
            Frobenius_norm_distance = torch.sum((source_similarity_matrix - out_similarity_matrix)**2)
            structure_loss = Frobenius_norm_distance

            loss = composition_loss*self.lambda_composition+screen_loss*self.lambda_screen+structure_loss*self.lambda_structure

            loss.backward()
            self.optimizer.step()

            losses["total_loss"] = loss.item()
            losses["screen_loss"] = screen_loss.item()
            losses["structure_loss"] = structure_loss.item()
            losses["composition_loss"] = composition_loss.item()
            print(self.scheduler.get_last_lr())
            losses["lr"] = self.scheduler.get_last_lr()[0]
            if self.wandb:
                editedimg = wandb.Image(self.editedimg.detach().cpu().numpy().squeeze(0).transpose(1,2,0))
                contentimg = wandb.Image(self.content_img_aug.detach().cpu().numpy().squeeze(0).transpose(1,2,0))
                wandb.log(losses)
                if self.log_picture:
                    if step % 30==0:
                        wandb.log({"content":contentimg,"output":editedimg})

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
