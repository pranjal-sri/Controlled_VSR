# utility for plotting frames
import matplotlib.pyplot as plt
import torch
import os
from datasets import Vimeo90KDataset
from torch.utils.data import  DataLoader
import yaml

def plot_frames(frames):
  """
  Plots a sequence of 7 frames next to each other with time stamp.

  Args:
    frames: A tensor of shape (7, C, H, W) containing the 7 frames.
  """
  fig, axs = plt.subplots(3, 3, figsize = (12, 12))
  axs = axs.flatten()
  for i, ax in enumerate(axs):
    if i < 7:
      ax.imshow((frames[i]*255).permute(1, 2, 0).cpu().to(torch.uint8))
      ax.set_title(f"t{i+1}")
    ax.axis('off')

  plt.suptitle("Sequence of 7 Frames")
  plt.tight_layout()
  plt.show()

def parse_yaml_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Utility for loading dataloaders

def load_dataloaders_from_config(config, *args, **kwargs):
    with open(config['TRAIN_DATA_FILE']) as f:
        train_samples = f.read().splitlines()
        train_samples = [os.path.join(config['TRAIN_DIR'], sample) for sample in train_samples]
    
    with open(config['TEST_DATA_FILE']) as f:
        test_samples = f.read().splitlines()
        test_samples = [os.path.join(config['TEST_DIR'], sample) for sample in test_samples]
    
    VALIDATION_FLAG = False
    if 'VALIDATION_SPLIT' in config and config['VALIDATION_SPLIT'] is not None:
        VALIDATION_FLAG = True
        n_validation_samples = int(len(train_samples) * config['VALIDATION_SPLIT'])
        train_samples, validation_samples = train_samples[:-n_validation_samples], train_samples[-n_validation_samples:]
    
    train_dataset =  Vimeo90KDataset(config['TRAIN_DIR'], *args, sample_paths = train_samples, **kwargs)
    degenerated_train_dataloader = DataLoader(train_dataset,
                                              batch_size=config["TRAIN"]["HYP"]["SEQS_PER_BATCH"],
                                              shuffle=config["TRAIN"]["HYP"]["SHUFFLE"],
                                              num_workers=config["TRAIN"]["HYP"]["NUM_WORKERS"],
                                              pin_memory=config["TRAIN"]["HYP"]["PIN_MEMORY"],
                                              drop_last=True,
                                              persistent_workers=config["TRAIN"]["HYP"]["PERSISTENT_WORKERS"])
  
    test_dataset = Vimeo90KDataset(config['TEST_DIR'], *args, sample_paths = test_samples, **kwargs)
    degenerated_test_dataloader = DataLoader(test_dataset,
                                           batch_size=config["TRAIN"]["HYP"]["SEQS_PER_BATCH"],
                                          shuffle=config["TEST"]["HYP"]["SHUFFLE"],
                                              num_workers=config["TEST"]["HYP"]["NUM_WORKERS"],
                                              pin_memory=config["TEST"]["HYP"]["PIN_MEMORY"],
                                              drop_last=True,
                                              persistent_workers=config["TEST"]["HYP"]["PERSISTENT_WORKERS"])
    if VALIDATION_FLAG:
        valid_dataset = Vimeo90KDataset(config['TRAIN_DIR'], *args, sample_paths = validation_samples, **kwargs)
        degenerated_valid_dataloader = DataLoader(valid_dataset,
                                              batch_size=config["TRAIN"]["HYP"]["SEQS_PER_BATCH"],
                                              shuffle=config["TRAIN"]["HYP"]["SHUFFLE"],
                                              num_workers=config["TRAIN"]["HYP"]["NUM_WORKERS"],
                                              pin_memory=config["TRAIN"]["HYP"]["PIN_MEMORY"],
                                              drop_last=True,
                                              persistent_workers=config["TRAIN"]["HYP"]["PERSISTENT_WORKERS"])
        
        return degenerated_train_dataloader, degenerated_valid_dataloader, degenerated_test_dataloader
    return degenerated_train_dataloader, degenerated_test_dataloader
    
  
