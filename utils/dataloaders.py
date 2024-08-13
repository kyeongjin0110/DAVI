import os
import torch.utils.data as data
from PIL import Image
from utils import transforms as tr
import glob
import numpy as np

import cv2
import torch

from PIL import ImageFilter
import random

'''
Load all training and validation data paths
''' 

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

    
class xBD(data.Dataset):

    def __init__(self, dataset_dir, set_name, aug=False, sam=True):

        set_list = set_name.split('/')
        self.aug = aug
        self.sam = sam
        
        init_dataset_dir = dataset_dir

        self.pre_img_list = []
        for dir_name in set_list:
            
            if dir_name == 'sunda-tsunami' or dir_name == 'lower-puna-volcano' or dir_name == 'nepal-flooding' or dir_name == 'pinery-bushfire' or dir_name == 'portugal-wildfire' or dir_name == 'woolsey-fire':
                dataset_dir = dataset_dir.replace('tier1', 'tier3')
            else:
                dataset_dir = init_dataset_dir
                
            img_path = os.path.join(dataset_dir, 'images_256/{}'.format(dir_name))
            pre_img_list_temp = glob.glob(img_path + '/*_pre_*')
            self.pre_img_list += pre_img_list_temp

        pre_img_dir = self.pre_img_list[0]
        post_img_dir = pre_img_dir.replace('pre', 'post')
        mask_dir = pre_img_dir.replace('images_256', 'targets_256')
        mask_dir = mask_dir.replace('disaster', 'disaster_b2')
        damage_dir = mask_dir.replace('pre', 'post')
        
    def __getitem__(self, index):
        
        pre_img_dir = self.pre_img_list[index]
        post_img_dir = pre_img_dir.replace('pre', 'post')
        mask_dir = pre_img_dir.replace('images_256', 'targets_256')
        mask_dir = mask_dir.replace('disaster', 'disaster_b2')
        pre_damage_dir = mask_dir
        damage_dir = mask_dir.replace('pre', 'post')
        
        pre_confidence_img_dir = pre_img_dir.replace('pre_disaster', 'pre_disaster_confidence')
        pre_confidence_img_dir = pre_confidence_img_dir.replace('images_256', 'images_256/sam')
        post_confidence_img_dir = post_img_dir.replace('post_disaster', 'post_disaster_confidence')
        post_confidence_img_dir = post_confidence_img_dir.replace('images_256', 'images_256/sam')
        
        img1 = Image.open(pre_img_dir)
        img2 = Image.open(post_img_dir)
        pre_label = Image.open(pre_damage_dir)
        label = Image.open(damage_dir)
        
        img1_conf = Image.open(pre_confidence_img_dir)
        img2_conf = Image.open(post_confidence_img_dir)
        
        damage_class = np.asarray(label)
        pre_damage_class = damage_class 
        # replace non-classified pixels with background
        damage_class = np.where(damage_class==0, 0, damage_class)
        damage_class = np.where(damage_class==1, 0, damage_class)
        damage_class = np.where(damage_class==2, 1, damage_class)
        damage_class = np.where(damage_class==3, 1, damage_class)
        damage_class = np.where(damage_class==4, 1, damage_class)
        damage_class = np.where(damage_class==5, 0, damage_class)
        
        img1_conf = np.asarray(img1_conf)
        img2_conf = np.asarray(img2_conf)
        
        damage_class = Image.fromarray(np.uint8(damage_class), 'L')
        pre_damage_class = Image.fromarray(np.uint8(pre_damage_class), 'L')
        
        img1_conf = Image.fromarray(np.uint8(img1_conf), 'L')
        img2_conf = Image.fromarray(np.uint8(img2_conf), 'L')
        
        sample = {'image': (img1, img2), 'label': (damage_class, pre_damage_class, img1_conf, img2_conf)}
        
        if self.aug:
            sample = tr.train_transforms(sample)  
            img1_aug = tr.strong_transforms(sample['image'][0])
            img2_aug = tr.strong_transforms(sample['image'][1])
        else:
            sample = tr.test_transforms(sample)
            img1_aug = tr.no_transforms(sample['image'][0])
            img2_aug = tr.no_transforms(sample['image'][1])
            
        img1_ori = sample['image'][0]
        img2_ori = sample['image'][1]
        label = sample['label'][0]    
        pre_label = sample['label'][1]
        img1_conf = sample['label'][2]
        img2_conf = sample['label'][3]
        
        img_dir = pre_img_dir
        
        return img1_ori, img2_ori, img1_aug, img2_aug, img1_conf, img2_conf, label, pre_label, img_dir
    
    def __len__(self):
        return len(self.pre_img_list)