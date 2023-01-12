from tiff_process import tiff_process
#from dataset import LOADDataset
import pymp
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
import os
from torch.utils.data import DataLoader 

import matplotlib.pyplot as plt

from model import MRUNet
from loss import MixedGradientLoss
from utility import *
from dataloader import DatasetCustom

def run_model(model, dataloader, optimizer, loss, batch_size, device, phase=None):
    if phase == "train":
        model.train()
    elif phase == "validation":
        model.eval()
    else:
        print("Error: Phase not defined for function run_model.\n") 
        return None
    
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0

    # TODO remove this line, it is to test nan values in grad
    torch.autograd.set_detect_anomaly(True)

    if phase == "train":
        for data_lst, data_nvdi in dataloader:
            #image_data = data[0].to(device)        
            # zero grad the optimizer
            optimizer.zero_grad()
            
            outputs = model(data_lst)
            
            loss_ = loss.get_loss(outputs, data_lst, data_nvdi)

            # backpropagation
            (loss_.sum()).backward()
            
            # update the parameters
            optimizer.step()
            
            # add loss of each item (total items in a batch = batch size)
            running_loss += loss_.sum().item()

            # calculate batch psnr (once every `batch_size` iterations)
            # batch_psnr =  psnr(label, outputs, max_val)
            # running_psnr += batch_psnr
            # batch_ssim =  ssim(label, outputs, max_val)
            # running_ssim += batch_ssim
            
            # for p in model.parameters():
            #     if p.requires_grad:
            #         print(p.name, p.data)

    elif phase == "validation":
        with torch.no_grad():
            for data_lst, data_nvdi in dataloader:
                outputs = model(data_lst)
                
                loss_ = loss.get_loss(outputs, data_lst, data_nvdi)
                running_loss += loss_.sum().item()

                # calculate batch psnr (once every `batch_size` iterations)
                # batch_psnr =  psnr(label, outputs, max_val)
                # running_psnr += batch_psnr
                # batch_ssim =  ssim(label, outputs, max_val)
                # running_ssim += batch_ssim
        
    # final_loss = running_loss/len(dataloader.dataset)
    # final_psnr = running_psnr/int(len(dataloader.dataset)/batch_size)
    # final_ssim = running_ssim/int(len(dataloader.dataset)/batch_size)
    # return final_loss, final_psnr, final_ssim
    return running_loss, running_loss, running_loss

def process_data(path, train_size=0.75, n_cores=3):
    # Images are saves in a .npz file.
    # LST are of size Nx2x64x64
    # ndvi are Nx256x256
    
    assert os.path.exists(path), "PathError: Path doesn't exist!"
        
    start = time.time()

    # Read path 
    npzfile = np.load(path)
    lst = npzfile['lst']   # Nx64x64x2
    ndvi = npzfile['ndvi'] # Nx256x256
    
    assert lst.shape[0] == ndvi.shape[0], "ImageError: The number of lst and nvdi images is not correct!"

    N_imgs = lst.shape[0]

    # Shuffle the images
    np.random.seed(42)
    randomize = np.arange(N_imgs)
    np.random.shuffle(randomize)
    lst = lst[randomize,:,:,:]
    ndvi = ndvi[randomize,:,:]

    lst = lst[:1500,:,:,:]
    ndvi = ndvi[:1500,:,:]
    
    # This puts the night and day images one after the other, thus the indexing in the ndvi corresponding image for both is idx/2 
    # ( Images with clouds / sea already taken care of )
    aux = np.zeros((2*lst.shape[0], lst.shape[1], lst.shape[2]))
    i = 0
    for i in range(0,lst.shape[0]*2,2):
        aux[i,:,:] = lst[int(i/2),:,:,0]
        aux[i+1,:,:] = lst[int(i/2),:,:,1]
    lst = aux
    del aux

    # LST max value (for normalization)
    max_val = np.max(lst)
    print('Max pixel value of training set is {},\nIMPORTANT: Please save it for later used as the normalization factor\n'.format(max_val))

    lst = lst / max_val
    
    # This takes about 5 seconds for 5000 images so its ok to do it each time the script is run
    Loss = MixedGradientLoss("cpu")
    aux = torch.zeros((ndvi.shape[0], ndvi.shape[1]-2, ndvi.shape[2]-2))
    for i in range(aux.shape[0]):
        aux[i,:,:] = Loss.get_gradient( torch.Tensor(ndvi[None,i,:,:]))
    
    upsampled_lst = np.zeros((lst.shape[0], 256, 256)) 

    for i in range(lst.shape[0]):
        upsampled_lst[i,:,:] = cv2.resize(lst[i,:,:], (256, 256), cv2.INTER_CUBIC)

    del lst

    lst = torch.Tensor(upsampled_lst)
    ndvi = torch.Tensor(aux)

    # Add none dimension due to batching in pytorch
    lst = lst[:,None,:,:]
    ndvi = ndvi[:,None,:,:]

    n_training_imgs = 2 * int(N_imgs * train_size)

    lst_train = lst[:n_training_imgs,:,:,:]
    ndvi_train = ndvi[:int(n_training_imgs/2),:,:,:]

    lst_val = lst[n_training_imgs:,:,:,:]
    ndvi_val = ndvi[int(n_training_imgs/2):,:,:,:]

    
    print("Total images used in each set:")
    print("\tLST (day and night) train:{}, validation:{}".format(lst_train.shape[0],  lst_val.shape[0]))
    print("\tndvi   (gradients)  train:{}, validation:{}".format(ndvi_train.shape[0], ndvi_val.shape[0]))
    
    # Plot ndvi and gradient for these images
    # L = [34, 267, 1845]
    # for im in L:
    #     plt.imsave('NDVI_{}.png'.format(im),ndvi[im,:,:])
    #     plt.imsave('NDVI_grad_{}.png'.format(im),aux[im,:,:])

    end = time.time()
    print(f"Finished processing data in {(end-start):.3f} seconds \n")
    
    return lst_train, ndvi_train, lst_val, ndvi_val

def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        print("Not executing on the GPU. Continue ? (y/n)")
        if input() != 'y':
            return 

    n_cores = 4

    lst_train, ndvi_train, lst_val, ndvi_val = process_data(args.datapath, n_cores=n_cores)
    
    # At this point all images are put in the GPU (which is faster but takes more memory, consider updating them in the dataloader loop)
    lst_train, lst_val = lst_train.to(device), lst_val.to(device)
    ndvi_train, ndvi_val = ndvi_train.to(device), ndvi_val.to(device) 
    
    # Load dataset and create data loader
    #transform = None
    #train_data = LOADDataset(lst_train, ndvi_train, transform=transform)
    #val_data = LOADDataset(lst_val, ndvi_val, transform=transform)
    
    batch_size = args.batch_size
    transform_augmentation_train = None
    train_dataset = DatasetCustom(lst_train, ndvi_train)
    val_dataset = DatasetCustom(lst_val, ndvi_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    #print('Length of training set: {}'.format(len(train_loader)))
    #print('Length of validating set: {}'.format(len(val_loader)))
    print('\tShape of LST input: ({},{})'.format(lst_train.shape[-2],lst_train.shape[-1]))
    print('\tShape of NVDI gradient input: ({},{})'.format(ndvi_train.shape[-2],ndvi_train.shape[-1]))
    
    epochs = args.epochs
    lr = args.lr
    model_name = args.model_name
    continue_train = args.continue_train == 'True'

    model = MRUNet(res_down=True, n_resblocks=1, bilinear=0).to(device)    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss = MixedGradientLoss(device)

    if os.path.exists("Metrics") == False:
        os.makedirs("Metrics")

    if not continue_train:
        # TRAINING CELL
        train_loss, val_loss = [], []
        train_psnr, val_psnr = [], []
        train_ssim, val_ssim = [], []
        start = time.time()

        last_epoch = -1
        best_validation_loss = np.inf

    else:
        # Load the lists of last time training metrics
        metrics = np.load(os.path.join("./Metrics",model_name + ".npy"))
        train_loss, val_loss = metrics[0].tolist(), metrics[3].tolist()
        train_psnr, val_psnr = metrics[1].tolist(), metrics[4].tolist()
        train_ssim, val_ssim = metrics[2].tolist(), metrics[5].tolist()
        start = time.time()

        # Model loading
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        losses = checkpoint['losses']
        vloss = losses[3]

    # TODO remove this test image thing 
    test_img_idx = 31
    plt.imsave('original_img.png',(lst_train[test_img_idx,0,:,:]).cpu().detach().numpy())
    plt.imsave('original_nvdi_img.png',(ndvi_train[test_img_idx,0,:,:]).cpu().detach().numpy())
    
    for epoch in range(last_epoch+1,epochs):
        
        print(f"Epoch {epoch + 1} of {epochs}")

        train_epoch_loss, train_epoch_psnr, train_epoch_ssim = run_model(model, train_loader, optimizer, loss, batch_size, device, phase="train")
        val_epoch_loss, val_epoch_psnr, val_epoch_ssim = run_model(model, train_loader, optimizer, loss, batch_size, device, phase="validation")
        
        if epoch % 5 == 0:
            output = model(lst_train[test_img_idx,:,:,:][None,:,:,:])
            plt.imsave('output_ep_{:d}.png'.format(epoch),output[0,0,:,:].cpu().detach().numpy())

        
        print(f"\tTrain loss: {train_epoch_loss:.6f}")
        print(f"\tVal loss: {val_epoch_loss:.6f}")
        train_loss.append(train_epoch_loss)
        train_psnr.append(train_epoch_psnr)
        train_ssim.append(train_epoch_ssim)
        val_loss.append(val_epoch_loss)
        val_psnr.append(val_epoch_psnr)
        val_ssim.append(val_epoch_ssim)
        
        if val_epoch_loss < best_validation_loss:
            print(10*"=")
            print("Saving model...")
            print(10*"=")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': [train_epoch_loss, train_epoch_psnr, train_epoch_ssim, val_epoch_loss, val_epoch_psnr, val_epoch_ssim],
                }, model_name)
            
            losses_path = os.path.join("./Metrics",model_name)
            metrics = [train_loss,train_psnr,train_ssim,val_loss,val_psnr,val_ssim]
            np.save(losses_path,metrics)
            best_validation_loss = val_epoch_loss
    
    end = time.time()
    print(f"Finished training in: {((end-start)/60):.3f} minutes")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch MR UNet training from tif files contained in a data folder",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datapath', help='path to directory containing training tif data')
    
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='size of batch')
    parser.add_argument('--model_name', type=str, help='name of the model')
    parser.add_argument('--continue_train', choices=['True', 'False'], default='False', type=str, 
                        help="flag for continue training, if True - continue training the 'model_name' model, else - training from scratch")
    args = parser.parse_args()

    main(args)



