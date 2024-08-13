import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms
import torch.nn.functional as F

import albumentations as at
import albumentations.pytorch


def random_affine(img, min_rot=None, max_rot=None, min_shear=None,max_shear=None, min_scale=None, max_scale=None):

    assert (len(img.shape) == 3)
    a = np.radians(np.random.rand() * (max_rot - min_rot) + min_rot)
    shear = np.radians(np.random.rand() * (max_shear - min_shear) + min_shear)
    scale = np.random.rand() * (max_scale - min_scale) + min_scale
  
    affine1_to_2 = torch.FloatTensor([[np.cos(a) * scale, - np.sin(a + shear) * scale, 0.],
                             [np.sin(a) * scale, np.cos(a + shear) * scale, 0.],
                             [0., 0., 1.]])  # 3x3
  
    affine2_to_1 = torch.linalg.inv(affine1_to_2)
  
    affine1_to_2, affine2_to_1 = affine1_to_2[:2, :], affine2_to_1[:2, :]  # 2x3
  
    img = perform_affine_tf(img.unsqueeze(dim=0).float(), affine1_to_2.unsqueeze(dim=0).float())
    img = img.squeeze(dim=0).cpu()
  
    return img, affine1_to_2.cpu(), affine2_to_1.cpu()


def perform_affine_tf(data, tf_matrices):
    # expects 4D tensor, we preserve gradients if there are any
  
    n_i, k, h, w = data.shape
    n_i2, r, c = tf_matrices.shape
    assert (n_i == n_i2)
    assert (r == 2 and c == 3)
  
    grid = F.affine_grid(tf_matrices, data.shape)  # output should be same size
    data_tf = F.grid_sample(data, grid, padding_mode="zeros")  # this can ONLY do bilinear
  
    return data_tf

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask1 = sample['label'][0]
        mask2 = sample['label'][1]
        
        conf_mask1 = sample['label'][2]
        conf_mask2 = sample['label'][3]
        
        img1 = np.array(img1).astype(np.float32).transpose((2, 0, 1))
        img2 = np.array(img2).astype(np.float32).transpose((2, 0, 1))
        mask1 = np.array(mask1).astype(np.float32) / 255.0
        mask2 = np.array(mask2).astype(np.float32) / 255.0
        
        conf_mask1 = np.array(conf_mask1).astype(np.float32) / 255.0
        conf_mask2 = np.array(conf_mask2).astype(np.float32) / 255.0

        img1 = torch.from_numpy(img1).float()
        img2 = torch.from_numpy(img2).float()
        mask1 = torch.from_numpy(mask1).float()
        mask2 = torch.from_numpy(mask2).float()
        
        conf_mask1 = torch.from_numpy(conf_mask1).float()
        conf_mask2 = torch.from_numpy(conf_mask2).float()

        return {'image': (img1, img2),
                'label': (mask1, mask2, conf_mask1, conf_mask2)}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask1 = sample['label'][0]
        mask2 = sample['label'][1]
        conf_mask1 = sample['label'][2]
        conf_mask2 = sample['label'][3]
        if random.random() < 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            mask1 = mask1.transpose(Image.FLIP_LEFT_RIGHT)
            mask2 = mask2.transpose(Image.FLIP_LEFT_RIGHT)
            conf_mask1 = conf_mask1.transpose(Image.FLIP_LEFT_RIGHT)
            conf_mask2 = conf_mask2.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': (img1, img2),
                'label': (mask1, mask2, conf_mask1, conf_mask2)}

    
class RandomVerticalFlip(object):
    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask1 = sample['label'][0]
        mask2 = sample['label'][1]
        conf_mask1 = sample['label'][2]
        conf_mask2 = sample['label'][3]
        if random.random() < 0.5:
            img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
            img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
            mask1 = mask1.transpose(Image.FLIP_TOP_BOTTOM)
            mask2 = mask2.transpose(Image.FLIP_TOP_BOTTOM)
            conf_mask1 = conf_mask1.transpose(Image.FLIP_TOP_BOTTOM)
            conf_mask2 = conf_mask2.transpose(Image.FLIP_TOP_BOTTOM)

        return {'image': (img1, img2),
                'label': (mask1, mask2, conf_mask1, conf_mask2)}

    
class RandomFixRotate(object):
    def __init__(self):
        self.degree = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask1 = sample['label'][0]
        mask2 = sample['label'][1]
        conf_mask1 = sample['label'][2]
        conf_mask2 = sample['label'][3]
        if random.random() < 0.75:
            rotate_degree = random.choice(self.degree)
            img1 = img1.transpose(rotate_degree)
            img2 = img2.transpose(rotate_degree)
            mask1 = mask1.transpose(rotate_degree)
            mask2 = mask2.transpose(rotate_degree)
            conf_mask1 = conf_mask1.transpose(rotate_degree)
            conf_mask2 = conf_mask2.transpose(rotate_degree)

        return {'image': (img1, img2),
                'label': (mask1, mask2, conf_mask1, conf_mask2)}

    
class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask1 = sample['label'][0]
        mask2 = sample['label'][1]
        conf_mask1 = sample['label'][2]
        conf_mask2 = sample['label'][3]
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img1 = img1.rotate(rotate_degree, Image.BILINEAR)
        img2 = img2.rotate(rotate_degree, Image.BILINEAR)
        mask1 = mask1.rotate(rotate_degree, Image.NEAREST)
        mask2 = mask2.rotate(rotate_degree, Image.NEAREST)
        conf_mask1 = conf_mask1.rotate(rotate_degree, Image.NEAREST)
        conf_mask2 = conf_mask2.rotate(rotate_degree, Image.NEAREST)

        return {'image': (img1, img2),
                'label': (mask1, mask2, conf_mask1, conf_mask2)}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask1 = sample['label'][0]
        mask2 = sample['label'][1]
        conf_mask1 = sample['label'][2]
        conf_mask2 = sample['label'][3]
        if random.random() < 0.5:
            img1 = img1.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            img2 = img2.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': (img1, img2),
                'label': (mask1, mask2, conf_mask1, conf_mask2)}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

    
class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask1 = sample['label'][0]
        mask2 = sample['label'][1]
        
        conf_mask1 = sample['label'][2]
        conf_mask2 = sample['label'][3]

        assert img1.size == mask1.size and img2.size == mask2.size

        img1 = img1.resize(self.size, Image.BILINEAR)
        img2 = img2.resize(self.size, Image.BILINEAR)
        mask1 = mask1.resize(self.size, Image.NEAREST)
        mask2 = mask2.resize(self.size, Image.NEAREST)
        conf_mask1 = conf_mask1.resize(self.size, Image.NEAREST)
        conf_mask2 = conf_mask2.resize(self.size, Image.NEAREST)

        return {'image': (img1, img2),
                'label': (mask1, mask2, conf_mask1, conf_mask2)}

    
'''
We don't use Normalize here, because it will bring negative effects.
the mask of ground truth is converted to [0,1] in ToTensor() function.
'''
train_transforms = transforms.Compose([
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomFixRotate(),
            FixedResize(256),
            ToTensor()])

test_transforms = transforms.Compose([
            FixedResize(256),
            ToTensor()])

strong_transforms = transforms.Compose([
            transforms.RandomGrayscale(p=0.5),          
            transforms.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 0.2)), # sigma=(0.1, 0.2), sigma=(9, 11)           
            ])

no_transforms = transforms.Compose([
            transforms.RandomGrayscale(p=0.0),])