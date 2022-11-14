import cv2
from skimage import io
import matplotlib.pyplot as plt

im = io.imread('./MODIS/MOD_2015_MOD11A1/tifs_files/1km/MOD11A1.A2015011.h18v04.061.2021348023226.hdf.0205.tif')
im2 = io.imread('./MODIS/MOD_2015_MOD13Q1/tifs_files/250m/MOD13Q1.A2015001.h18v04.061.2021317060846.hdf.0205.tif')
#im2 = im2/255
print("LST image size : " , im.shape)
print("NDVI image size : " , im2.shape)
print("NDVI image max value" , im2.max())
plt.imsave('NDVI image.png',im2[:,:,0])

#img = cv2.imread('MOD11A1.A2011016.h18v04.061.2021187075057.hdf.0044.tif',cv2.IMREAD_ANYDEPTH)
#cv2.imshow('image',img)