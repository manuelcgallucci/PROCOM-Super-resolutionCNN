import torch
import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

from model import MRUNet


pretrained = './test_v5'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load pretrained model
model_MRUnet = MRUNet(res_down=False, n_resblocks=1, bilinear=True).to(device)
model_MRUnet.load_state_dict(torch.load(pretrained)['model_state_dict'])
# Normalization factor and scale
max_val = 327.3800048828125
scale = 4


# Tiff process
data = np.load('final_database.npz')
lst = data['lst']
ndvi = data['ndvi']
print(np.shape(lst))
print(np.shape(ndvi))

# Test model
indx = 5000
lst_image = lst[indx,:,:,0]
ndvi_image = ndvi[indx,:,:]
lst_image_250 = cv2.resize(lst_image, (256, 256), cv2.INTER_CUBIC)


input = torch.tensor(np.reshape(lst_image_250/max_val, (1,1,256,256)), dtype=torch.float).to(device)
output = model_MRUnet(input).cpu().detach().numpy()*max_val

# plot and save fig   
fontsize = 15
clmap = 'jet'
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
im1 = ax[0,0].imshow(lst_image, vmin = lst_image.min(), vmax = lst_image.max(), cmap = clmap)
ax[0,0].set_title("Ground truth day", fontsize=fontsize)
ax[0,0].axis('off')

im2 = ax[0,1].imshow(ndvi_image, vmin = ndvi_image.min(), vmax = ndvi_image.max(), cmap = clmap)
ax[0,1].set_title("NDVI", fontsize=fontsize)
ax[0,1].axis('off')

im3 = ax[1,0].imshow(lst_image_250, vmin = lst_image_250.min(), vmax = lst_image_250.max(), cmap = clmap)
ax[1,0].set_title("Bicubic Interpolation", fontsize=fontsize)
ax[1,0].axis('off')

im4 = ax[1,1].imshow(output[0,0,:,:], vmin = output.min(), vmax = output.max(), cmap = clmap)
ax[1,1].set_title("Model Output", fontsize=fontsize)
ax[1,1].axis('off')

# cmap = plt.get_cmap('jet',20)
# fig.tight_layout()
# fig.subplots_adjust(right=0.7)
# cbar_ax = fig.add_axes([0.65, 0.15, 0.03, 0.7])
# cbar_ax.tick_params(labelsize=15)
# fig.colorbar(im1, cax=cbar_ax, cmap = clmap)

plt.savefig("./samples/interpolation_"+ str(indx)+ ".png",  bbox_inches='tight')