import cv2
from skimage import io
import matplotlib.pyplot as plt
from utility import *
import numpy as np
import os
from torchvision.transforms.functional import resize
import torchvision.transforms as T
import torch


from PIL import Image
import glob
import re 

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

def create_gif(idx, edge, paths, output_path, MAX_CONSTANT, frame_in_ms = 400):    
    file_list = glob.glob(paths) # Asumes order is correct
    
    frames = []
    for filename in file_list:
        # img = Image.open(filename)
        temp = re.findall(r'\d+', filename)
        epoch = list(map(int, temp))[0]

        data = np.load(filename)
        lst_img = data['outputs']
        lst_img = lst_img[idx,0,:,:][edge:256-edge,edge:256-edge] * MAX_CONSTANT
        
        plot_save(lst_img, 'Epoch: {:03d} \n edge: {:d}'.format(epoch, edge), './temp_{:03d}.png'.format(epoch))

        img = Image.open('temp_{:03d}.png'.format(epoch))
        frames.append(img)
        
    # gif_image = Image.new('RGB', (img.width, img.height))
    # gif_image.save(output_path, format='GIF', append_images=frames, save_all=True, duration=frame_in_ms, loop=0)

    gif_image = frames[0]
    gif_image.save(output_path, format='GIF', append_images=frames[1:], save_all=True, duration=frame_in_ms, loop=0)

    file_list = glob.glob('./temp_*.png')
    for filename in file_list:
        if 'temp' in filename:
            os.remove(filename)

# Histogram paramters
def save_histogram(x, out_path, title, n_bins=20):
    plt.figure()
    counts, bins = np.histogram(x, bins=n_bins)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.title(title)
    plt.savefig(out_path)
    plt.close()

def save_double_histograms(x, y, labelx, labely, out_path, title, n_bins=20):

    max_ = max([np.amax(x), np.amax(y)])
    min_ = min([np.amin(x), np.amin(y)])
    range_ = tuple((min_, max_))#np.linspace(, n_bins)
    plt.figure()
    counts, bins = np.histogram(x, bins=n_bins, range=range_)
    counts = counts / np.sum(counts)
    plt.hist(bins[:-1], bins, weights=counts, alpha=0.5, label=labelx)

    counts, bins = np.histogram(y, bins=n_bins, range=range_)
    counts = counts / np.sum(counts)
    plt.hist(bins[:-1], bins, weights=counts, alpha=0.5, label=labely)

    #plt.hist(y.flatten(), bins, alpha=0.5, label=labely)
    plt.legend(loc='best')
    plt.title(title)
    plt.savefig(out_path)
    plt.close()

def plot_save(x, title, path):
    plt.imshow(x)
    plt.colorbar()
    plt.title(title)
    plt.savefig(path)
    plt.close()

def save_losses(metrics_path, out_path, alpha=0.01, epochs=[x*5 for x in range(95)]):

    data = np.load(metrics_path)
    # Values 
    # mge_train_loss, mse_train_loss,train_loss,train_psnr,train_ssim,mge_val_loss,mse_val_loss,val_loss,val_psnr,val_ssim
    mge_train_loss = data[0,:] * alpha
    mse_train_loss = data[1,:]
    train_loss = data[2,:]

    mge_val_loss = data[5,:] * alpha
    mse_val_loss = data[6,:]
    val_loss = data[7,:]

    plt.plot(epochs, mge_train_loss, label="MGE Loss")
    plt.plot(epochs, mse_train_loss, label="MSE Loss")
    plt.plot(epochs, train_loss, label="Total Loss")
    plt.xlabel("Epochs (saved every 5)")
    plt.title("Train losses \n alpha:{:4.3f}".format(alpha))
    plt.legend(loc="best")
    plt.savefig(out_path + "loss_train.png")
    plt.close()

    plt.plot(epochs, mge_val_loss, label="MGE Loss")
    plt.plot(epochs, mse_val_loss, label="MSE Loss")
    plt.plot(epochs, val_loss, label="Total Loss")
    plt.xlabel("Epochs (saved every 5)")
    plt.title("Validation losses \n alpha:{:4.3f}".format(alpha))
    plt.legend(loc="best")
    plt.savefig(out_path + "loss_val.png")
    plt.close()

    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epochs (saved every 5)")
    plt.title("Loss comparison \n alpha:{:4.3f}".format(alpha))
    plt.legend(loc="best")
    plt.savefig(out_path + "loss_both.png")
    plt.close()


def main(MAX_CONSTANT):
    
    out_epoch = 295

    edge = 16
    n_imgs = 5
    model_name = "test_a0-01"
    samples_dir = './output/'+ model_name +'/samples/'
    training_data_dir = './output/'+ model_name +'/training_data/'
    metrics_dir = './output/'+ model_name +'/metrics/'
    #samples_dir = './samples/'
    #training_data_dir = './'
    #metrics_dir = './Metrics/'

    save_losses(metrics_dir + 'metrics.npy', metrics_dir)
     
    out_epoch_file = training_data_dir + 'output_ep_{:d}.npz'.format(out_epoch)
    original_imgs_file = training_data_dir + 'original_images.npz'
    data = np.load(out_epoch_file)
    originals = np.load(original_imgs_file)

    outputs = data['outputs'] # model outputs
    lst = originals['lst'] # lst outputs
    ndvi = originals['ndvi'] # ndvi gradients
    original_lst = originals['lst_original'] # original lst
    original_ndvi = originals['ndvi_original'] # original ndvi

    # print(np.shape(outputs))
    # print(np.shape(lst))
    # print(np.shape(ndvi))
    # print(np.shape(original_lst))
    # print(np.shape(original_ndvi))

    for i in range(n_imgs):
        output_image = outputs[i,0,:,:][edge:256-edge,edge:256-edge] * MAX_CONSTANT

        # save_histogram(output_image, './samples/hist_output_'+str(i)+'.png', "Histogram \n from: Output LST 256x256 250m", n_bins=20)
        # save_histogram(original_lst[i,:,:]* MAX_CONSTANT, './samples/hist_original_lst_'+str(i)+'.png', "Histogram \n from: Original LST 64x64 250m", n_bins=20)

        save_double_histograms(output_image, original_lst[i,:,:]* MAX_CONSTANT, "Output LST 256x256 1km", "Real LST 64x64 250m", './samples/hist_comp_'+str(i)+'.png', "Histogram comparison")

        plot_save(output_image, "Model Output LST \n 256x256 250m", samples_dir + 'output_'+str(i)+'.png')
        plot_save(lst[i,:,:], "Interpolated Bicubic LST \n 256x256 1km", samples_dir + 'lst_'+str(i)+'.png')
        plot_save(ndvi[i,:,:], "Normalized NVDI Gradient \n (from: 256x256 250m)", samples_dir + 'ndvi_'+str(i)+'.png')
        plot_save(original_lst[i,:,:]* MAX_CONSTANT, "Orignal LST \n 64x64 1km", samples_dir + 'original_lst_'+str(i)+'.png')
        plot_save(original_ndvi[i,:,:], "Orignal NDVI \n 256x256 250m", samples_dir + 'original_ndvi_'+str(i)+'.png')

        # create_gif(i, edge, training_data_dir + 'output_ep_*.npz', samples_dir + 'out_seq'+str(i)+'.gif', MAX_CONSTANT)

        image = original_lst[i,:,:]
        up = cv2.resize(image, (256, 256), cv2.INTER_CUBIC)
        up = torch.reshape(torch.Tensor(up),(1,256,256))
        down = resize(up,(64,64),T.InterpolationMode.BICUBIC)
        down = down.detach().numpy()
        print("Norm between original LST and upscale then downscale version:", np.linalg.norm(original_lst-down))

if __name__ == "__main__":
    MAX_CONSTANT = 333.32000732421875
    main(MAX_CONSTANT)
