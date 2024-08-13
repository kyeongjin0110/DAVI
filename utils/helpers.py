import logging
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from utils.dataloaders import xBD
# from models.siamese_unet import SNUNet_ECAM
from models.Models_sw import SNUNet_ECAM

import os
import glob
from torch.utils.data import random_split

logging.basicConfig(level=logging.INFO)


def get_loaders(opt):


    logging.info('STARTING Dataset Creation')
    
    if opt.dataset == 'xBD':
        opt.train_dataset = xBD(opt.dataset_dir, opt.source_disaster, aug=True, sam=opt.sam)
        opt.val_dataset = xBD(opt.dataset_dir, opt.target_disaster, aug=False, sam=opt.sam)
    else:
        assert(0)
    
        
    logging.info('STARTING Dataloading')

    train_loader = torch.utils.data.DataLoader(opt.train_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.num_workers)
    val_loader = torch.utils.data.DataLoader(opt.val_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
        
    return train_loader, val_loader


def load_model(opt):
    """Load the model

    Parameters
    ----------
    opt : dict
        User specified flags/options
    device : string
        device on which to train model

    """
    model = SNUNet_ECAM(opt.num_channel, 2).to(opt.dev)
    return model