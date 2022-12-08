from tiff_process import tiff_process
from dataset import LOADDataset
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

from model import MRUNet
from utils import *
from dataloader import DataLoaderCustom


def train(model, dataloader, optimizer, train_data, max_val):
    # Train model
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    for bi, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        image_data = data[0].to(device)
        label = data[1].to(device)
        
        # zero grad the optimizer
        optimizer.zero_grad()
        outputs = model(image_data)
        loss = get_loss(outputs*max_val, label)
        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step()
        # add loss of each item (total items in a batch = batch size)
        running_loss += loss.item()
        # calculate batch psnr (once every `batch_size` iterations)
        batch_psnr =  psnr(label, outputs, max_val)
        running_psnr += batch_psnr
        batch_ssim =  ssim(label, outputs, max_val)
        running_ssim += batch_ssim
    final_loss = running_loss/len(dataloader.dataset)
    final_psnr = running_psnr/int(len(train_data)/dataloader.batch_size)
    final_ssim = running_ssim/int(len(train_data)/dataloader.batch_size)
    return final_loss, final_psnr, final_ssim

def validate(model, dataloader, epoch, val_data, max_val):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            image_data = data[0].to(device)
            label = data[1].to(device)
            outputs = model(image_data)
            
            loss = get_loss(outputs*max_val, label)
            # add loss of each item (total items in a batch = batch size) 
            running_loss += loss.item()
            # calculate batch psnr (once every `batch_size` iterations)
            batch_psnr = psnr(label, outputs, max_val)
            running_psnr += batch_psnr
            batch_ssim =  ssim(label, outputs, max_val)
            running_ssim += batch_ssim
        outputs = outputs.cpu()
        # save_image(outputs, f"../outputs/val_sr{epoch}.png")
    final_loss = running_loss/len(dataloader.dataset)
    final_psnr = running_psnr/int(len(val_data)/dataloader.batch_size)
    final_ssim = running_ssim/int(len(val_data)/dataloader.batch_size)
    return final_loss, final_psnr, final_ssim


def process_data(path, train_size=0.75, n_cores=n_cores)
    # Images are saves in a .npz file.
    # LST are of size Nx2x64x64
    # NVDI are Nx256x256
    
    assert os.paths.exists(path), "PathError: Path doesn't exist!"
        
    start = time.time()

    # Read path 
    np.load(path)
    lst = npzfile['lst']   # Nx2x64x64
    nvdi = npzfile['nvdi'] # Nx256x256

    N_imgs = lst.shape[0]

    # Shuffle the images
    np.random.seed(42)
    randomize = np.arange(N_imgs)
    np.random.shuffle(randomize)
    lst = lst[randomize,:,:,:]
    nvdi = nvdi[randomize,:,:]

    print("Verify TODO this rehape is correct!")
    # This puts the night and day images one after the other, thus the indexing in the nvdi corresponding image for both is idx/2 
    # ( Images with clouds / sea already taken care of )
    lst = lst.reshape(N_imgs*2, lst.shape[2], lst.shape[3])

    # LST max value (for normalization)
    max_val = np.max(y_train)
    print('Max pixel value of training set is {},\nIMPORTANT: Please save it for later used as the normalization factor\n'.format(max_val))

    lst = lst / max_val
     
    #lst = lst.shape(lst.shape[0],1,lst.shape[1],lst.shape[2])
    #nvdi = nvdi.shape(nvdi.shape[0],1,nvdi.shape[1],nvdi.shape[2])
    
    n_training_imgs = 2 * int(N_imgs * train_size)

    lst_train = lst[:n_training_imgs,:,:]
    nvdi_train = nvdi[:n_training_imgs/2,:,:]

    lst_val = lst[n_training_imgs:,:,:]
    nvdi_val = nvdi[n_training_imgs/2:,:,:]

    end = time.time()
    print("Total images used in each set:")
    print("\tLST (day and night) train:{}, validation:{}".format(n_training_imgs,  N_imgs * 2 - n_training_imgs))
    print("\tNVDI   (gradients)  train:{}, validation:{}".format(n_training_imgs/2, N_imgs - n_training_imgs/2))
    
    print(f"Finished processing data in {((end-start)/60):.3f} minutes \n")
    
    return lst_train, nvdi_train, lst_val, nvdi_val

def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        print("Not executing on the GPU. Continue ? (y/n)")
        if input() != 'y':
            return 

    model = MRUNet(res_down=True, n_resblocks=1, bilinear=0).to(device)

    n_cores = 4

    lst_train, nvdi_train, lst_val, nvdi_val = process_data(args.datapath, n_cores=n_cores)

    # Load dataset and create data loader
    #transform = None
    #train_data = LOADDataset(lst_train, nvdi_train, transform=transform)
    #val_data = LOADDataset(lst_val, nvdi_val, transform=transform)
    
    batch_size = args.batch_size
    transform_augmentation_train = None
    train_loader = DataLoaderCustom(lst_train, nvdi_train, batch_size=batch_size, shuffle=False)
    val_loader = DataLoaderCustom(lst_val, nvdi_val, batch_size=batch_size)

    print('Length of training set: {} \n'.format(len(train_data)))
    print('Length of validating set: {} \n'.format(len(val_data)))
    print('Shape of input: ({},{}) \n'.format(lst_train.shape[-2],lst_train.shape[-1]))

    epochs = args.epochs
    lr = args.lr
    model_name = args.model_name
    continue_train = args.continue_train == 'True'

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if os.path.exists("Metrics") == False:
        os.makedirs("Metrics")

    if not continue_train:
        # TRAINING CELL
        train_loss, val_loss = [], []
        train_psnr, val_psnr = [], []
        train_ssim, val_ssim = [], []
        start = time.time()

        last_epoch = -1
        vloss = np.inf

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

    for epoch in range(last_epoch+1,epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_psnr, train_epoch_ssim = train(model, train_loader, optimizer, train_data, max_val)
        val_epoch_loss, val_epoch_psnr, val_epoch_ssim = validate(model, val_loader, epoch, val_data, max_val)
        print(f"Train loss: {train_epoch_loss:.6f}")
        print(f"Val loss: {val_epoch_loss:.6f}")
        train_loss.append(train_epoch_loss)
        train_psnr.append(train_epoch_psnr)
        train_ssim.append(train_epoch_ssim)
        val_loss.append(val_epoch_loss)
        val_psnr.append(val_epoch_psnr)
        val_ssim.append(val_epoch_ssim)
        if val_epoch_loss < vloss:
            print("Save model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': [train_epoch_loss, train_epoch_psnr, train_epoch_ssim,val_epoch_loss, val_epoch_psnr, val_epoch_ssim],
                }, model_name)
            losses_path = os.path.join("./Metrics",model_name)
            metrics = [train_loss,train_psnr,train_ssim,val_loss,val_psnr,val_ssim]
            np.save(losses_path,metrics)
            vloss = val_epoch_loss
    end = time.time()
    print(f"Finished training in: {((end-start)/60):.3f} minutes")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch MR UNet training from tif files contained in a data folder",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datapath', help='path to directory containing training tif data')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=24, type=int, help='size of batch')
    parser.add_argument('--model_name', type=str, help='name of the model')
    parser.add_argument('--continue_train', choices=['True', 'False'], default='False', type=str, 
                        help="flag for continue training, if True - continue training the 'model_name' model, else - training from scratch")
    args = parser.parse_args()

    main(args)






