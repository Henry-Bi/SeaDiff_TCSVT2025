"""
 If you find this code useful, please cite our paper:

 Mahmoud Afifi, Marcus A. Brubaker, and Michael S. Brown. "HistoGAN:
 Controlling Colors of GAN-Generated and Real Images via Color Histograms."
 In CVPR, 2021.

 @inproceedings{afifi2021histogan,
  title={Histo{GAN}: Controlling Colors of {GAN}-Generated and Real Images via
  Color Histograms},
  author={Afifi, Mahmoud and Brubaker, Marcus A. and Brown, Michael S.},
  booktitle={CVPR},
  year={2021}
}
"""
import os
from RGBuvHistBlock import RGBuvHistBlock
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from os.path import splitext, join, basename, exists


image_folder = ''
output_folder = ''
if exists(output_folder) is False:
  os.mkdir(output_folder)
  
torch.cuda.set_device(0)
histblock = RGBuvHistBlock(insz=336, h=336,
                           resizing='sampling',
                           method='inverse-quadratic',
                           sigma=0.02,
                           device=torch.cuda.current_device())
transform = transforms.Compose([transforms.Resize((336, 336)),
                                transforms.ToTensor()])

image_names = os.listdir(image_folder)
for filename in image_names:
  print(filename)
  img_hist = Image.open(os.path.join(image_folder, filename))
  img_hist = torch.unsqueeze(transform(img_hist), dim=0).to(
    device=torch.cuda.current_device())
  histogram = histblock(img_hist)
  # histogram = histogram.cpu().numpy()
  save_image(histogram * 255, os.path.join(output_folder, filename))
  # np.save(join(output_dir, basename(splitext(filename)[0]) + '.npy'), histogram)

