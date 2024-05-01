# Defining dataset for videos
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
from SRGAN_PyTorch.imgproc import image_to_tensor, image_resize
import cv2


class Vimeo90KDataset(Dataset):
  def __init__(self, root_directory, scale,
               n_samples = None, n_batches = None,
               sample_paths = None, batch_paths = None):
    # import pdb; pdb.set_trace()
    self.root_directory = root_directory
    self.scale = scale
    if sample_paths is not None:
      self.sample_paths = sample_paths
      self.batch_paths = [os.path.join(*sample.split('/')[:-1]) for sample in self.sample_paths]
      self.batch_paths = list(set(self.batch_paths))
    elif batch_paths is not None:
      self.batch_paths = batch_paths
      self.sample_paths = [os.path.join(batch, sample) 
                            for batch in self.batch_paths 
                            for sample in os.listdir(batch)]
    
    elif n_samples is not None:
      self.batch_paths = os.listdir(root_directory)
      self.batch_paths = [os.path.join(root_directory, batch) for batch in self.batch_paths]
      random.shuffle(self.batch_paths)
      self.sample_paths = []
      
      remaining = n_samples
      n_batches = 0
      for batch in self.batch_paths:
        current = min(remaining, len(os.listdir(batch)))
        remaining -= current
        n_batches += 1

        sample_paths_from_dir = map(lambda x: os.path.join(batch, x), os.listdir(batch)[:current])
        self.sample_paths.extend(sample_paths_from_dir)

        if remaining == 0:
          break
      
      self.batch_paths = self.batch_paths[:n_batches]

    elif n_batches is not None:
      self.batch_paths = random.sample(os.listdir(root_directory), n_batches)
      self.batch_paths = [os.path.join(root_directory, batch) for batch in self.batch_paths]
      self.sample_paths = []

      for batch in self.batch_paths:
        sample_paths_from_dir = map(lambda x: os.path.join(batch, x), os.listdir(batch))
        self.sample_paths.extend(sample_paths_from_dir)

    else:
      raise ValueError('Specify one of the following: sample_paths, batch_paths, n_samples or n_batches')

  def __len__(self):
    return len(self.sample_paths)


  def open_image(self, path):
    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255.
    image = image_to_tensor(image, range_norm=False, half=False)
    return image

  def __getitem__(self, idx):
    sample_path = self.sample_paths[idx]
    hq_images = [self.open_image(os.path.join(sample_path, frame)) 
                          for frame in sorted(os.listdir(sample_path)) if frame[-3:] == 'png']
    lq_images = [image_resize(image, 1 / self.scale) for image in hq_images]
    hq_seq = torch.stack(hq_images)
    lq_sec = torch.stack(lq_images)
    return {'hq': hq_seq, 'lq':lq_sec}