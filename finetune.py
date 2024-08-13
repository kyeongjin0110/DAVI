import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from collections import defaultdict
from torchvision.utils import save_image

from torchmetrics import AUROC
from sklearn.metrics import classification_report

from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def average(x):
    return sum(x) / len(x)

    
def train_epoch(
    model,
    model_t,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    loss_list,
    global_step,
    epoch,
    logging,
    opt,
):
    
    logging.info('SET model mode to train!')
    batch_iter = 0
    
    loss_val_dict = defaultdict(list)
    cnt = 0
    
    c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    acc_list = []
    auroc_list = []
    
    softmax = torch.nn.Softmax(dim=1)
    
    pbar = tqdm(range(len(train_loader)))
    for pre_imgs, post_imgs, pre_aug_imgs, post_aug_imgs, confidence_pre_imgs, confidence_post_imgs, labels, pre_labels, file_names in train_loader:
        
        labels = torch.where(labels != 0.0, 1.0, 0.0)
        pre_labels = torch.where(pre_labels != 0.0, 1.0, 0.0)

        # Set variables for training
        pre_imgs = pre_imgs.float().to(opt.dev)
        post_imgs = post_imgs.float().to(opt.dev)
        pre_aug_imgs = pre_aug_imgs.float().to(opt.dev)
        post_aug_imgs = post_aug_imgs.float().to(opt.dev)
        confidence_pre_imgs = confidence_pre_imgs.float().to(opt.dev)
        confidence_post_imgs = confidence_post_imgs.float().to(opt.dev)
        labels = labels.float().to(opt.dev)
        pre_labels = pre_labels.float().to(opt.dev)

        batch_size = labels.shape[0]
        w, h = labels.shape[1], labels.shape[2]
        class_labels = labels.view([batch_size, -1])
        class_labels = (torch.sum(class_labels, dim=1)!=0).long().to(opt.dev).detach()   
        
        batch_iter = batch_iter+batch_size
        opt.total_step += 1

        # Get source model predictions
        preds_t = model_t(pre_imgs, post_imgs)
        cd_preds_t = preds_t[0]
        source_fine_labels = torch.argmax(cd_preds_t, dim=1).to(opt.dev).detach()

        # Get target model predictions
        preds = model.forward_with_class(pre_aug_imgs, post_aug_imgs)
            
        cd_preds, cls_preds = preds[0], preds[1]
            
        # Get target model predictions (aug)
        aug1_preds = model.forward_with_class(pre_imgs, post_imgs)
        cd_aug1_preds, _ = aug1_preds[0], aug1_preds[1]

        # Get means, variances of target model
        mean_preds = torch.mean(torch.cat([softmax(cd_preds)[:,1,:,:].unsqueeze(0), softmax(cd_aug1_preds)[:,1,:,:].unsqueeze(0)]), dim=0)
        variance_preds = torch.var(torch.cat([softmax(cd_preds)[:,1,:,:].unsqueeze(0), softmax(cd_aug1_preds)[:,1,:,:].unsqueeze(0)]), dim=0)
        mean_preds = mean_preds.to(opt.dev).detach()
        variance_preds = variance_preds.to(opt.dev).detach()

        # Update the preds of source model
        mean_preds = torch.where(mean_preds>0.5, 1.0, 0.0)
        corrected_cd_preds_t = torch.where((variance_preds<opt.refine_threshold) & (mean_preds>=1.0), mean_preds.double(), source_fine_labels.double())
        corrected_cd_preds_t = corrected_cd_preds_t.to(opt.dev).detach()

        # Get fine-grained pseudo labels from source model
        source_fine_labels = corrected_cd_preds_t.to(opt.dev).detach() # use (source pred)

        # Get SAM preds
        mask_confidence_pre_imgs = torch.where(confidence_pre_imgs>0.0, 1.0, 0.0)
        confidence_post_imgs = confidence_post_imgs*mask_confidence_pre_imgs
        # Get fine-grained pseudo labels from SAM
        fine_labels = (confidence_pre_imgs-confidence_post_imgs).to(opt.dev).detach() # Diff confidence (diff = pre-post)
 
        # clip
        before_temp_fine_labels = fine_labels
        fine_labels = torch.clamp(fine_labels, min=0.0, max=1.0)
        soft_confidence_labels = fine_labels
        
        if opt.hard == True:
            fine_labels[fine_labels>opt.hard_threshold] = 1
            fine_labels[fine_labels<=opt.hard_threshold] = 0
            hard_confidence_labels = fine_labels
            if opt.add_source == True:
                
                before_corrects = (fine_labels+torch.argmax(cd_preds_t, dim=1))/2
                before_corrects[before_corrects>0] = 1

                after_corrects = (fine_labels+source_fine_labels)/2                
                after_corrects[after_corrects>0] = 1
                fine_labels = after_corrects
                
            fine_labels = fine_labels.long()     
        else:
            pass
            
        # Get coarse-grained pseudo labels from source model and sam
        coarse_labels = (torch.sum((torch.argmax(cd_preds_t, dim=1)).view([batch_size, -1]), dim=1)!=0) 
        coarse_labels = coarse_labels.long().to(opt.dev).detach()
        
        if opt.coarse_filter == True:
            for i, coarse_label in enumerate (coarse_labels): # coarse
                if coarse_label == 0:
                    fine_labels[i] = torch.zeros_like(fine_labels[i]).long().to(opt.dev).detach()

        if opt.gt_filter == True:
            for i, class_label in enumerate (class_labels): # gt
                if class_label == 0:
                    fine_labels[i] = torch.zeros_like(fine_labels[i]).long().to(opt.dev).detach()
          
        # Calculate loss
        loss = None
        for loss_name in loss_list: 
            if loss_name in 'cd_loss':
                if opt.hard == True:
                    # ce
                    reverse_labels = torch.ones_like(fine_labels).float().to(opt.dev)
                    temp_labels = torch.subtract(reverse_labels, fine_labels).unsqueeze(1)
                    fine_labels = torch.cat([temp_labels, fine_labels.unsqueeze(1)], dim=1)
                    ori_loss = torch.mean(-(F.log_softmax(cd_preds, dim=1)*fine_labels))
                val = ori_loss
                cd_loss = val
            elif loss_name in 'entropy_loss': # tent (lambda=1.0)
                ori_loss = opt.lambda_tta*torch.mean(-F.softmax(cd_preds, dim=1)*F.log_softmax(cd_preds, dim=1))
                val = ori_loss     
            else:
                assert(0)
                
            if loss == None:
                loss = val
            else:
                loss += val
                
            loss_val_dict[loss_name].append(val.detach().item())
            
        loss_val_dict['total'].append(loss.detach().item())
        
        # Zero the gradient
        optimizer.zero_grad()
        # Backprop
        loss.backward()
        optimizer.step()
        
        pbar_str = f'Total {average(loss_val_dict["total"]):.3f}'   
        
        loss_str_list = []
        for loss_name in loss_list:
            loss_str = f'{loss_name} {average(loss_val_dict[loss_name]):.5f}'
            loss_str_list.append(loss_str)
        loss_str = ', '.join(loss_str_list)
        pbar_str = f'{pbar_str} ({loss_str})'
        
        pbar.set_description(pbar_str)
        pbar.update(1)

        _, cd_preds_indices = torch.max(cd_preds, 1)
        
        ret = confusion_matrix(labels.data.cpu().numpy().flatten(), cd_preds_indices.data.cpu().numpy().flatten(), labels=[0, 1]).ravel()
            
        tn, fp, fn, tp = ret[0], ret[1], ret[2], ret[3]
        c_matrix['tn'] += tn
        c_matrix['fp'] += fp
        c_matrix['fn'] += fn
        c_matrix['tp'] += tp

        # Calculate and log other batch metrics
        cd_corrects = (100 *
                       (cd_preds_indices.squeeze().byte() == labels.squeeze().byte()).sum() /
                       (labels.size()[0] * (opt.patch_size**2)))
        
        acc_list.append(cd_corrects)

        # clear batch variables from memory
        del pre_imgs, post_imgs, pre_aug_imgs, post_aug_imgs, labels, pre_labels, file_names
    
    scheduler.step()
   
    acc_mean = np.array(torch.stack(acc_list, dim=0).reshape(-1).tolist()).mean()
    
    tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * P * R / (R + P)
    
    return acc_mean, P, R, F1


def train(
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
):
    
    """
     Set starting values
    """

    logging.info('STARTING training')
    opt.total_step = -1
    global_step = 0
    cnt = 0
    
    for epoch in range(opt.epochs):
       
        """
        Begin Training
        """
        
        log = ''
        
        epoch_str = '\n\n'
        epoch_str += f'Epoch {epoch + 1}/{opt.epochs}\n'
        epoch_str += ('-' * 10 + '\n')
        log += epoch_str
        
        print(epoch_str)
        
        model.train()
        model_t.eval()

        global_step += 1
        train_acc_mean, train_P, train_R, train_F1 = train_epoch(
            model,
            model_t,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            loss_list,
            global_step,
            epoch,
            logging,
            opt,
        )

        mean_train_metrics = f"(\'cd_losses\': -1, \'cd_corrects\': {train_acc_mean}, \'cd_precisions\': {train_P}, \'cd_recalls\': {train_R}, \'cd_f1scores\': {train_F1}, \'learning_rate\': -1)"

        logging.info("EPOCH {} TRAIN METRICS".format(epoch) + str(mean_train_metrics))
        log += f'EPOCH {epoch} TRAIN METRICS' + str(mean_train_metrics) + '\n'

        if epoch+1 == 50:
            torch.save(model, opt.run_dir+'/checkpoint_epoch_'+str(epoch)+'.pt')
        
        val_acc_mean, val_P, val_R, val_F1 = validation(
            model, 
            val_loader,
            scheduler,
            epoch,
            opt,
        )
        
        mean_val_metrics = f"(\'cd_losses\': -1, \'cd_corrects\': {val_acc_mean}, \'cd_precisions\': {val_P}, \'cd_recalls\': {val_R}, \'cd_f1scores\': {val_F1}, \'learning_rate\': -1\)"

        logging.info("EPOCH {} VALIDATION METRICS".format(epoch)+str(mean_val_metrics))
        log += f'EPOCH {epoch} VALIDATION METRICS' + str(mean_val_metrics) + '\n'

        print('An epoch finished.')
        log += f'An epoch finished.\n'
        
        print(log, file=flog, flush=True)
    
    print('Done!')
    
    
def validation(
    model, 
    val_loader, 
    scheduler,
    epoch,
    opt,
):
    
    """
    Begin Validation
    """
    eval_cnt = 0
    model.eval()

    c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    acc_list = []
    
    with torch.no_grad():
        for pre_imgs, post_imgs, _, _, _, _, labels, _, _ in tqdm(val_loader):
            
            labels = torch.where(labels != 0.0, 1.0, 0.0)
            
            # Set variables for training
            pre_imgs = pre_imgs.float().to(opt.dev)
            post_imgs = post_imgs.float().to(opt.dev)
            labels = labels.long().to(opt.dev)
            
            batch_size = labels.shape[0]

            # Get predictions and calculate loss
            preds = model.forward_with_class(pre_imgs, post_imgs)
            cd_preds, cls_preds = preds[0], preds[1]
            
            _, cd_preds_indices = torch.max(cd_preds, 1)
            
            ret = confusion_matrix(labels.data.cpu().numpy().flatten(), cd_preds_indices.data.cpu().numpy().flatten(), labels=[0, 1]).ravel()
            
            tn, fp, fn, tp = ret[0], ret[1], ret[2], ret[3]
            c_matrix['tn'] += tn
            c_matrix['fp'] += fp
            c_matrix['fn'] += fn
            c_matrix['tp'] += tp

            # Calculate and log other batch metrics
            cd_corrects = (100 *
                           (cd_preds_indices.squeeze().byte() == labels.squeeze().byte()).sum() /
                           (labels.size()[0] * (opt.patch_size**2)))
            
            acc_list.append(cd_corrects)
            
            # clear batch variables from memory
            del pre_imgs, post_imgs, labels
            
    acc_mean = np.array(torch.stack(acc_list, dim=0).reshape(-1).tolist()).mean()

    tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * P * R / (R + P)
            
    return acc_mean, P, R, F1