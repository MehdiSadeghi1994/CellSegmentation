import os
import time
import torch
import copy
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
from my_utils.metrics import dice_coefficients


def train_model(model, dataloaders, criterion, optimizer, device, num_data, val_step=2, scheduler=None, model_name='Best_Model', multi_task=False, max_epoch=50):
    if torch.cuda.is_available():
        print('\n-------training mode is on CUDA-------\n')
    else:
        print('\n-------training mode is on CPU-------\n')

    if os.path.exists(f'Checkpoint_Files/{model_name}_checkpoint.tar'):
        checkpoint = torch.load(f'Checkpoint_Files/{model_name}_checkpoint.tar', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_model_state = checkpoint['model_state_dict']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if torch.cuda.is_available():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        last_epoch = checkpoint['epoch']
        best_dice = checkpoint['best_dice']
        losses = checkpoint['losses']

    else:
        last_epoch = 0
        best_dice = 0.0
        losses= {'Train':[], 'Validation':[]}
    
    
    model.to(device)
    model.train()
    for epoch in range(last_epoch, max_epoch):

        running_loss = 0
        running_dice = 0
        with tqdm(total=num_data['Train'], desc=f'Epoch {epoch + 1}/{max_epoch}', unit='img') as pbar:
            for image, label in dataloaders['Train']:
                image = image.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(mode=True):
                    logit, logit_b = model(image)
                    _, predic = torch.max(logit, 1)
                    if multi_task:
                        loss = criterion(logit, logit_b, label)
                    else: 
                        loss = criterion(logit, label)

                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item() * image.shape[0]
                dice_temp = dice_coefficients(predic.cpu().numpy(), label.cpu().numpy())
                running_dice += ((dice_temp[0]+dice_temp[1])/2) * image.shape[0]
                pbar.update(image.shape[0])


        epoch_loss = running_loss / num_data['Train']
        epoch_dice = running_dice / num_data['Train']
        losses['Train'].append(epoch_loss)
        print(f'Train: Loss= {epoch_loss}, Dice= {epoch_dice}')
            
        if scheduler:
            scheduler.step()
        
        # Validate model
        if (epoch+1) % val_step == 0:
            val_loss, val_dice = validate_model(model,
                                                dataloaders['Validation'],
                                                criterion,
                                                device,
                                                num_data['Validation'],
                                                multi_task)
            losses['Validation'].append(val_loss)
            print(f'      Validation: Loss= {val_loss}, Dice= {val_dice}\n\n ')

        if val_dice > best_dice:
            best_dice = val_dice
            best_model_state = copy.deepcopy(model.state_dict())



        torch.save({'epoch': epoch+1,
                    'model_state_dict': best_model_state,
                    'best_dice': best_dice,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses': losses
        }, f'Checkpoint_Files/{model_name}_checkpoint.tar')

    print('Best Validation Dice: {:4f}'.format(best_dice))
    
    plt.plot(losses['Train'], label='Training loss')
    plt.plot(losses['Validation'], label='Validation loss')
    idx = np.argmin(losses['Validation'])
    plt.plot([idx,idx], [0,losses['Validation'][idx]],'--', color= 'coral' , label = 'Best' )
    plt.plot([0,idx], [losses['Validation'][idx],losses['Validation'][idx]],'--', color= 'coral' )
    plt.plot(idx, losses['Validation'][idx] , 'o' ,color = 'coral', markersize = 8, markerfacecolor = "None" )
    plt.legend()
    plt.ylabel('Loss', fontsize= 12)
    plt.xlabel('Epoch',fontsize= 12)
    plt.tight_layout()
    plt.title(f'{model_name}_Train-Validation loss ',fontsize= 12)
    plt.savefig(f'loss_{model_name}.png',bbox_inches='tight')
    plt.show()
    

    model.load_state_dict(best_model_state)
    return model




def validate_model(model, dataloader, criterion, device, num_data, multi_task):
    print('   Validation is in process....')
    model.to(device)
    model.eval()

    running_loss = 0
    running_dice = 0
    for image, label in dataloader:
        image = image.to(device)
        label = label.to(device)
        
        
        with torch.set_grad_enabled(mode=False):
            logit, logit_b = model(image)
            _, predic = torch.max(logit, 1)


            if multi_task:
                loss = criterion(logit, logit_b, label)
            else: 
                loss = criterion(logit, label)
            
            running_loss += loss.item() * image.shape[0]
            dice_temp = dice_coefficients(predic.cpu().numpy(), label.cpu().numpy())
            running_dice += ((dice_temp[0]+dice_temp[1])/2) * image.shape[0]


    val_loss = running_loss / num_data
    val_dice = running_dice / num_data

    return val_loss, val_dice


def test_model(model, dataloader, num_data, device, path_result=None, eval_metrics=None, model_name='Best'):
    print('\nTest is in process....\n')
    model.to(device)
    model.eval()
    metric_result = {metric.__name__: {} for metric in eval_metrics}
    for data_info in dataloader:
        image = data_info['image'].to(device)
        label = data_info['mask'].to(device)

        with torch.set_grad_enabled(mode=False):
            logit = model(image)[0]
            _, predic = torch.max(logit, 1)
            
        if eval_metrics:
            for func in eval_metrics:
                metric_result[func.__name__][data_info['name'][0]] = func(predic.cpu().numpy(), label.cpu().numpy())

        if path_result:
            pred_image = predic.cpu().numpy().squeeze()
            rescaled_image = (255.0 / pred_image.max() * (pred_image - pred_image.min())).astype(np.uint8)
            result = Image.fromarray(rescaled_image)
            add = os.path.join(path_result, data_info['name'][0].split('.')[0]+'_'+model_name+'_predicted.bmp')
            result.save(add)
    return metric_result
            