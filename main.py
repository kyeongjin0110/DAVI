VERSION_NAME = 'fr_v0000'

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

import datetime
import logging
import json
import copy

import argparse
import warnings

import matplotlib.pyplot as plt

from utils import transforms as tr
from sys import float_info

from utils.helpers import get_loaders, load_model
from utils.save_logs import get_param_log

from finetune import train
import option


warnings.filterwarnings(action='ignore')

"""
Initialize Parser and define arguments
"""
"""
Initialize experiments log
"""
def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--backbone', type=str, default='Siam-NestedUNet', help='save path for trained model')
    parser.add_argument('--run_name', type=str, default='test', help='save path for trained model')
    parser.add_argument('--pretrained', type=str, default='cnn_based_pretrained_xBD', help='path for pre-trained model')
    parser.add_argument('--seed', type=int, default=777, help='seed')
    parser.add_argument('--gpu', type=int, default=0, help='option for gpu')
    
    parser.add_argument('--patch_size', type=int, default=256, help='input patch size for training')
    parser.add_argument('--augmentation', type=int, default=1, help='input augmentation')
    
    parser.add_argument('--num_gpus', type=int, default=1, help='the number of gpus')
    parser.add_argument('--num_workers', type=int, default=8, help='the number of workers')
    parser.add_argument('--num_channel', type=int, default=3, help='the number of channel for input')
    parser.add_argument('--EF', type=int, default=0, help='input patch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=51, help='number of epochs') # 100
    parser.add_argument('--batch_size', type=int, default=8, help='batch size used in training')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate used in training') # 1e-6

    parser.add_argument('--dataset', type=str, default='xBD', help='type of dataset')
    parser.add_argument('--dataset_dir', type=str, default='./dataset', help='directory for dataset')
    parser.add_argument('--source_disaster', type=str, default='palu-tsunami', help='source disaster type')
    parser.add_argument('--target_disaster', type=str, default='palu-tsunami', help='target disaseter type') 
    
    parser.add_argument('--loss_opt', type=int, default=1, help='loss option') 
    parser.add_argument('--lambda_tta', type=float, default=0.1, help='lambda for change detection loss')
    parser.add_argument('--refine_threshold', type=float, default=0.001, help='threshold for refinement')
    
    parser.add_argument('--coarse_filter', type=bool, default=True, help='coarse-grained label filter')
    parser.add_argument('--gt_filter', type=bool, default=False, help='true label filter')
    
    parser.add_argument('--run_dir', type=str, default='./runs', help='directory for the checkpoints of trained model')
    
    # sam
    parser.add_argument('--sam', type=bool, default=True, help='activate sam')
    parser.add_argument('--soft', type=bool, default=False, help='soft label for sam')
    parser.add_argument('--hard', type=bool, default=True, help='hard label for sam')
    parser.add_argument('--add_source', type=bool, default=True, help='hard + source pred')
    parser.add_argument('--hard_threshold', type=float, default=0.1, help='threshold for hard label')

    return parser.parse_args()


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__=='__main__':

    opt = get_args()
    logging.basicConfig(level=logging.INFO)

    """
    Set up environment: define paths, download data, and set device
    """

    os.environ["CUDA_VISIBLE_DEVICES"]= f"{opt.gpu}"
    opt.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))

    seed_torch(seed=777)
    
    train_loader, val_loader = get_loaders(opt)

    opt.run_dir = os.path.join(opt.run_dir, VERSION_NAME, opt.run_name)
    if not os.path.exists(opt.run_dir):
        os.makedirs(opt.run_dir)

    """
    Load Model then define other aspects of the model
    """
    logging.info('LOADING Model')

    # target model (student model)
    model = load_model(opt)
    model_checkpoint = torch.load(f'./checkpoints/{opt.pretrained}/checkpoint_epoch_60.pt')
    model.load_state_dict(model_checkpoint.state_dict(), strict=False)
    
    # source model (teacher model)
    model_t = copy.deepcopy(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    # Get loss list
    loss_list = option.get_loss_opt(opt)
    
    # Save the log
    log_path = f'{opt.run_dir}/log'
    param_log = get_param_log(opt)
    
    print(f'\nLogging Path: {log_path}\n')
    print(param_log)
    
    flog = open(log_path, 'w')
    print(param_log, file=flog, flush=True)

    train(
        flog, 
        model,
        model_t,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_list,
        logging,
        opt,
    )
    
    flog.close()