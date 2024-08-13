import os
from datetime import datetime


def get_param_log(
    opt
):
    n_train = len(opt.train_dataset)
    n_val = len(opt.val_dataset)
    
    DATASET = ''
    if opt.dataset == 'xBD':
        DATASET = 'xBD'
    else:
        assert(0)
    
    log = ''
    log += f'\n***** {DATASET}: RSIs - {opt.source_disaster} ( Train ) *****\n\n'
    log += f'{opt.train_dataset}\n'
    log += f'\n***** {DATASET}: RSIs - {opt.target_disaster} ( Val ) *****\n\n'
    log += f'{opt.val_dataset}\n'
    
    log += f'\n***** {DATASET}: RSIs - {opt.source_disaster} *****\n\n'
    log += f'Num. of Train Imgs\t\t:{n_train}\n'
    log += f'Num. of Val. Imgs\t\t:{n_val}\n'

    log += f'\n***** Other Parameters *****\n\n'
    log += f'Epochs\t\t\t: {opt.epochs}\n'
    log += f'Batch Size\t\t: {opt.batch_size}\n'
    log += f'Optimizer\t\t: AdamW\n'
    log += f'Learning Rate\t\t: {opt.learning_rate}\n'

    log += f'Baseline Option\t\t: {opt.option}\n'
    log += f'Seed\t\t\t: {opt.seed}\n'
    
    # tta
    log += f'TTA Lambda\t\t: {opt.lambda_tta}\n'
    
    # filter
    log += f'Coarse Filter\t\t: {opt.coarse_filter}\n'
    log += f'GT Filter\t\t: {opt.gt_filter}\n'
    
    log += f'SAM Threshold\t\t: {opt.hard_threshold}\n'
    log += f'Flag for Adding Source\t: {opt.add_source}\n'
    
    log += f'Backbone\t\t: {opt.backbone}\n'
    log += f'Pretrained Dir.\t\t: {opt.pretrained}\n'
    
    return log