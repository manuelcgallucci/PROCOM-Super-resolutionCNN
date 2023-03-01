import cv2
from skimage import io
import matplotlib.pyplot as plt
from utility import *
import numpy as np
import os
from torchvision.transforms.functional import resize
import torchvision.transforms as T
import torch

"""
lst_path = './MODIS/MOD_2013_MOD11A1/hdfs_files/MOD11A1.A2013183.h18v04.061.2021305022634.hdf'
LST_K_day, LST_K_night, cols, rows, projection, geotransform = read_modis(lst_path)
plt.imshow(LST_K_day[256:256+64,64:64+64])
print(np.amin(LST_K_day),np.amax(LST_K_day))
plt.savefig('lst')
plt.close()

mask = np.where(LST_K_day == 0)
print(np.shape(mask))

print(mask)
#print(ndvi_mask)

ndvi_path = './MODIS/MOD_2013_MOD09GQ/hdfs_files/MOD09GQ.A2013183.h18v04.061.2021233024826.hdf'
qa, red, NIR, cols, rows, projection, geotransform = read_modis_MOD09GQ(ndvi_path)
plt.imshow(red)
plt.savefig('red')
plt.close()
print(np.nanmin(red),np.nanmax(red))

plt.imshow(NIR)
plt.savefig('NIR')
plt.close()
print(np.nanmin(NIR),np.nanmax(NIR))

ndvi = (NIR-red)/(NIR+red)
plt.imshow(ndvi[1024:1024+256,256:256+256])
plt.savefig('ndvi')
plt.close()
"""

"""
lst_root_dir      = 'MODIS/MOD_{}_{}'.format(2013,'MOD11A1')
lst_path = os.path.join(lst_root_dir, 'hdfs_files')
lst_hdfs = os.listdir(lst_path)
lst_hdfs.sort()

ndvi_root_dir      = 'MODIS/MOD_{}_{}'.format(2013,'MOD09GQ')
ndvi_path = os.path.join(ndvi_root_dir, 'hdfs_files')
ndvi_hdfs = os.listdir(ndvi_path)
ndvi_hdfs.sort()

for lst in lst_hdfs:
    path = os.path.join(lst_path,lst)
    read_value = read_modis(path)
    if read_value is not None :
         LST_K_day, LST_K_night, cols, rows, projection, geotransform = read_value
         plt.imsave('./samples/full_lst/'+lst+'.png',LST_K_day)

for ndvi in ndvi_hdfs:
    path = os.path.join(ndvi_path,ndvi)
    read_value = read_modis_MOD09GQ(path)
    if read_value is not None :
        qa, red, NIR, cols, rows, projection, geotransform = read_value
        NDVI = np.zeros((rows,cols))
        for i in range(rows):
            for j in range(cols):
                if NIR[i,j]+red[i,j] == 0 :
                    NDVI[i,j] = 0
                else:
                    NDVI[i,j] = (NIR[i,j]-red[i,j])/(NIR[i,j]+red[i,j])          
        plt.imsave('./samples/full_ndvi/'+ndvi+'.png',NDVI)
"""



"""
data = np.load('final_database.npz')
lst = data['lst']
ndvi = data['ndvi']
print(np.shape(lst))
print(np.shape(ndvi))

"""

"""
for i in range(9):
    lst_image = lst[i*100,:,:,1]
    ndvi_image = ndvi[i*100,:,:]
    plt.imshow(lst_image)
    plt.colorbar()
    plt.savefig('./samples/lst_'+str(i)+'.png')
    plt.close()
    plt.imshow(ndvi_image)
    plt.colorbar()
    plt.savefig('./samples/ndvi_'+str(i)+'.png')
    plt.close()

"""
"""

lst_image = lst[16,:,:,0]
ndvi_image = ndvi[15,:,:]
plt.imshow(ndvi_image)
plt.colorbar()
plt.savefig('./samples/ndvi_31.png')
plt.close()
plt.imshow(lst_image)
plt.colorbar()
plt.savefig('./samples/lst_31.png')
plt.close()
"""

"""
#modis/MODIS/MOD_2013_MOD09GQ/tifs_files/250m/MOD09GQ.A2013203.0270.tif
im = io.imread('./MODIS/MOD_2013_MOD11A1/tifs_files/1km/MOD11A1.A2013203.0270.tif')
im2 = io.imread('./MODIS/MOD_2013_MOD09GQ/tifs_files/250m/MOD09GQ.A2013203.0270.tif')
#im2 = im2/255
print("LST image size : " , im.shape)
print("NDVI image size : " , im2.shape)
print("NDVI image max value" , im2.max())
print(np.amin(im),np.amax(im))

plt.imshow(im2)
plt.savefig('NDVI_4')
plt.close()

plt.imshow(im[:,:,0])
plt.savefig('LST_4')
plt.close()


#img = cv2.imread('MOD11A1.A2011016.h18v04.061.2021187075057.hdf.0044.tif',cv2.IMREAD_ANYDEPTH)
#cv2.imshow('image',img)
"""

data = np.load('output_ep_5.npz')
originals = np.load('original_images.npz')
outputs = data['outputs']
lst = originals['lst']
ndvi = originals['ndvi']
original_lst = originals['lst_original']
original_ndvi = originals['ndvi_original']
print(np.shape(outputs))
print(np.shape(lst))
print(np.shape(ndvi))
print(np.shape(original_lst))
print(np.shape(original_ndvi))

for i in range(5):
    output_image = outputs[i,0,:,:][16:241,16:241] * 333.32000732421875
    plt.imshow(output_image)
    plt.colorbar()
    plt.savefig('./samples/output_'+str(i)+'.png')
    plt.close()
    plt.imshow(lst[i,:,:])
    plt.colorbar()
    plt.savefig('./samples/lst_'+str(i)+'.png')
    plt.close()
    plt.imshow(ndvi[i,:,:])
    plt.colorbar()
    plt.savefig('./samples/ndvi_'+str(i)+'.png')
    plt.close()
    plt.imshow(original_lst[i,:,:]* 333.32000732421875)
    plt.colorbar()
    plt.savefig('./samples/original_lst_'+str(i)+'.png')
    plt.close()
    plt.imshow(original_ndvi[i,:,:])
    plt.colorbar()
    plt.savefig('./samples/original_ndvi_'+str(i)+'.png')
    plt.close()
    image = original_lst[i,:,:]
    up = cv2.resize(image, (256, 256), cv2.INTER_CUBIC)
    up = torch.reshape(torch.Tensor(up),(1,256,256))
    down = resize(up,(64,64),T.InterpolationMode.BICUBIC)
    down = down.detach().numpy()
    print(np.linalg.norm(original_lst-down))


