# from SRGAN_PyTorch.image_quality_assessment import PSNR, SSIM
from tqdm import tqdm
import torch
import torch.nn as nn
import sys
from evaluate import evaluate
from utils import parse_yaml_config, load_dataloaders_from_config
from models import ControlledSRResNet, ControlledSRResNetWithAttention, ControlledSRResNetWithFlow
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import OneCycleLR

from losses import charbonnier_loss_func, flow_loss_func, charbonnier_flow_loss
import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import os

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda: 
        torch.cuda.manual_seed_all(seed_value) # gpu 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


def save_checkpoint(model, optim,  filename, scheduler = None,):
    if scheduler is None:
        torch.save({'optim': optim.state_dict(),
                    'model':model.state_dict()
                   }, filename)
    else:
        torch.save({'optim': optim.state_dict(),
                    'model':model.state_dict(),
                    'scheduler': scheduler.state_dict()
                   }, filename)

def load_model_from_checkpoint(filepath, model, optim = None, scheduler = None):
    chkpnt = torch.load(filepath)
    model.load_state_dict(chkpnt['model'])
    if optim is not None:
        optim.load_state_dict(chkpnt['optim'])
    if scheduler is not None:
        scheduler.load_state_dict(chkpnt['scheduler'])
    
def train(model, criterion, optim, train_loader, val_loader, 
          writer, checkpoint_file,
          scheduler = None, epochs = 3, val_frac = 0.5):
    # import pdb; pdb.set_trace()
    n_batches = len(train_loader)
    total_updates = 0
    total_validations = 0
    best_ssim = 0.0
    
    # number of batches after which we run validation
    val_batch_num = int(val_frac*n_batches)
    for epoch in range(1, epochs+1):
        for i, batch_data in enumerate(train_loader):
            total_updates += 1
            model.train()
            x = batch_data['lq']
            y = batch_data['hq']

            x = x.to(device)
            y = y.to(device)

            upsampled_vids = model(x)
            loss = criterion(y, upsampled_vids)
            
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(),clip_value = 1.0)
            optim.step()
            
            if scheduler is not None:
                scheduler.step()
                print(f"LR: {scheduler.get_last_lr()}")
            
            writer.add_scalar("Train/loss", loss.item(), total_updates)
            print(f"---- EP [{epoch}/{epochs}] | BTCH [{i+1}/{n_batches}] ||| train_loss = {loss.item():.5f} ----")
            
            if (i+1)%val_batch_num == 0:
                val_loss, psnr, ssim = evaluate(model, criterion, val_loader, y_channel_only = True)
                writer.add_scalar('Val/loss', val_loss, total_updates)
                writer.add_scalar('Val/psnr', psnr, total_updates)
                writer.add_scalar('Val/ssim', ssim, total_updates)
                print(f"VAL ||| loss = {val_loss}, psnr = {psnr}, ssim = {ssim}")
                if ssim > best_ssim:
                    save_checkpoint(model, optim, checkpoint_file)
                    best_ssim = ssim
                    

    

if __name__ == '__main__':
    # log_file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.log'
    config_file = sys.argv[1]
    config = parse_yaml_config(config_file)
    timestamp = sys.argv[2]
    
    SUMMARY_FILE = os.path.join('logs/tensorboard', f'{timestamp}') 
    CHECKPOINT_FILE = os.path.join(config['MODEL_CHECKPOINT_PATH'], f'{timestamp}.pth') 
    
    writer = SummaryWriter(log_dir = SUMMARY_FILE)
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    random_seed(config['SEED'], device == 'cuda')
    
    if config['MODEL'] == 'SRNET':
        model = ControlledSRResNet(path_to_weights=config['TRAINED_SRSNET_PATH'])
    elif config['MODEL'] == 'SRNET_WITH_ATTN':
        model = ControlledSRResNetWithAttention(path_to_weights=config['TRAINED_SRSNET_PATH'])
    elif config['MODEL'] == 'SRNET_WITH_FLOW':
        model = ControlledSRResNetWithFlow(path_to_weights=config['TRAINED_SRSNET_PATH'])
    else:
        raise ValueError(f'Model {config["MODEL"]} is not supported')

    model.to(device)
    model.set_train_mode(control = config['TRAIN']['HYP']['CONTROL_GRAD'],
                         net = config['TRAIN']['HYP']['NET_GRAD'])
    
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    
    if config['TRAIN']['OPTIM']['TYPE'] == 'Adam':
        optimiser = Adam(trainable_params, 
                         lr = config['TRAIN']['OPTIM']['LR'])
        
    elif config['TRAIN']['OPTIM']['TYPE'] == 'SGD':
        optimiser =  SGD(trainable_params, 
                         lr = config['TRAIN']['OPTIM']['LR'],
                         momentum=config['TRAIN']['OPTIM']['MOMENTUM'], 
                         nesterov=config['TRAIN']['OPTIM']['NESTEROV'])
    else:
        raise ValueError(f"Optimiser type {config['OPTIM']['TYPE']} is not supported")

    train_loader, val_loader, test_loader = load_dataloaders_from_config(config, scale = 4)
    
    scheduler = None
    if config['TRAIN']['SCHEDULER']['USE']:
        if config['TRAIN']['SCHEDULER']['TYPE'] == 'OneCycleLR':
            scheduler = OneCycleLR(optimiser, 
                                   max_lr = config['TRAIN']['SCHEDULER']['MAX_LR'],  
                                   total_steps =  config['TRAIN']['HYP']['EPOCHS']*len(train_loader))
        else:
            raise ValueError(f"Scheduler type {config['TRAIN']['SCHEDULER']['TYPE']} is not suppported")
     
    if 'RESUME_FROM_CHECKPOINT' in config['TRAIN']:
        load_model_from_checkpoint(config['TRAIN']['RESUME_FROM_CHECKPOINT'], model)
        
    
    if config['TRAIN']['LOSS']['TYPE'] == 'Ch':
        criterion = charbonnier_loss_func(eps = 1e-8)
    elif config['TRAIN']['LOSS']['TYPE'] == 'F':
        criterion = flow_loss_func(eps = 1e-8)
    elif config['TRAIN']['LOSS']['TYPE'] == 'Ch+F':
        criterion = charbonnier_flow_loss(w_c = config['TRAIN']['LOSS']['C_WEIGHT'], w_f = config['TRAIN']['LOSS']['F_WEIGHT'])
    else:
        raise ValueError(f"Loss type {config['TRAIN']['LOSS']['TYPE']} is not supported")
    
    train(model, criterion, optimiser, 
          train_loader, val_loader,
          writer,
          CHECKPOINT_FILE,
          scheduler = scheduler,
          epochs = config['TRAIN']['HYP']['EPOCHS'],
          val_frac=config['TRAIN']['HYP']['VAL_FRAC'])
    
