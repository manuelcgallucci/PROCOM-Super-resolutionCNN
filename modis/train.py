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

# nohup python3 train.py --datapath ./data/final_database.npz --model_name test_v1 &

def run_model(model, dataloader, optimizer, loss, batch_size, device, phase=None):
    if phase == "train":
        model.train()
    elif phase == "validation":
        model.eval()
    else:
        print("Error: Phase not defined for function run_model.\n") 
        return None
    
    mse_loss = 0.0
    mge_loss = 0.0
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0

    # TODO remove this line, it is to test nan values in grad
    torch.autograd.set_detect_anomaly(True)

    if phase == "train":
        for data_lst, data_nvdi, orginal_lst in dataloader:
            #image_data = data[0].to(device)        
            # zero grad the optimizer
            optimizer.zero_grad()
            
            outputs = model(data_lst)
            
            MGE, MSE, loss_ = loss.get_loss(outputs, orginal_lst, data_nvdi)

            # backpropagation
            (loss_.sum()).backward()
            
            # update the parameters
            optimizer.step()
            
            # add loss of each item (total items in a batch = batch size)
            running_loss += loss_.sum().item()
            mse_loss += MSE.sum().item()
            mge_loss += MGE.sum().item()

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
            for data_lst, data_nvdi, original_lst in dataloader:
                outputs = model(data_lst)
                
                MGE, MSE, loss_ = loss.get_loss(outputs, original_lst, data_nvdi)
                running_loss += loss_.sum().item()
                mse_loss += MSE.sum().item()
                mge_loss += MGE.sum().item()

                # calculate batch psnr (once every `batch_size` iterations)
                # batch_psnr =  psnr(label, outputs, max_val)
                # running_psnr += batch_psnr
                # batch_ssim =  ssim(label, outputs, max_val)
                # running_ssim += batch_ssim
        
    # final_loss = running_loss/len(dataloader.dataset)
    # final_psnr = running_psnr/int(len(dataloader.dataset)/batch_size)
    # final_ssim = running_ssim/int(len(dataloader.dataset)/batch_size)
    # return final_loss, final_psnr, final_ssim
    return mge_loss, mse_loss, running_loss, running_loss, running_loss

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
    
    # This puts the night and day images one after the other, thus the indexing in the ndvi corresponding image for both is idx/2 
    # ( Images with clouds / sea already taken care of )
    #aux = np.zeros((2*lst.shape[0], int(lst.shape[1]/2), int(lst.shape[2]/2)))
    aux = np.zeros((2*lst.shape[0], lst.shape[1], lst.shape[2]))
    i = 0
    for i in range(0,lst.shape[0]*2,2):
        #aux[i,:,:] = lst[int(i/2),:,:,0][0:32,0:32]
        #aux[i+1,:,:] = lst[int(i/2),:,:,1][0:32,0:32]
        aux[i,:,:] = lst[int(i/2),:,:,0]
        aux[i+1,:,:] = lst[int(i/2),:,:,1]
    lst = aux
    del aux

    #aux_ndvi = np.zeros((ndvi.shape[0], int(ndvi.shape[1]/2), int(ndvi.shape[2]/2)))
    #for i in  range(0,ndvi.shape[0]) :
    #    aux_ndvi[i,:,:] = ndvi[i,:,:][0:128,0:128]
    #ndvi = aux_ndvi
    #del aux_ndvi

    # LST max value (for normalization)
    max_val = np.max(lst)
    print('Max pixel value of training set is {},\nIMPORTANT: Please save it for later used as the normalization factor\n'.format(max_val))

    lst = lst / max_val
    
    # This takes about 5 seconds for 5000 images so its ok to do it each time the script is run
    Loss = MixedGradientLoss("cpu")
    aux = torch.zeros((ndvi.shape[0], ndvi.shape[1]-2, ndvi.shape[2]-2))
    #aux = torch.zeros((ndvi.shape[0], ndvi.shape[1], ndvi.shape[2]))
    for i in range(aux.shape[0]):
        aux[i,:,:] = Loss.get_gradient( torch.Tensor(ndvi[None,i,:,:]))
    
    upsampled_lst = np.zeros((lst.shape[0], 256, 256)) 

    for i in range(lst.shape[0]):
        upsampled_lst[i,:,:] = cv2.resize(lst[i,:,:], (256, 256), cv2.INTER_CUBIC)

    original_lst = lst
    original_ndvi = ndvi

    lst = torch.Tensor(upsampled_lst)
    ndvi = torch.Tensor(aux)
    original_lst_tensor = torch.Tensor(original_lst)

    # Add none dimension due to batching in pytorch
    lst = lst[0:2000,None,:,:]
    ndvi = ndvi[0:1000,None,:,:]
    original_lst_tensor = original_lst_tensor[0:2000,None,:,:]

    N_imgs = lst.shape[0]
    n_training_imgs = int(N_imgs * train_size)

    lst_train = lst[:n_training_imgs,:,:,:]
    ndvi_train = ndvi[:int(n_training_imgs/2),:,:,:]
    original_lst_train = original_lst_tensor[:n_training_imgs,:,:,:]

    lst_val = lst[n_training_imgs:,:,:,:]
    ndvi_val = ndvi[int(n_training_imgs/2):,:,:,:]
    original_lst_val = original_lst_tensor[n_training_imgs:,:,:,:]
    
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
    
    return lst_train, ndvi_train, lst_val, ndvi_val, original_lst_train, original_lst_val, original_lst, original_ndvi

# Creates metadata txt file
def create_meta(dir, args):
    writable_dict = {
        "alpha": args.alpha,
        "beta": args.beta,
        "epochs": args.epochs, 
        "lr": args.lr,
        "batch_size": args.batch_size,
        "model_name": args.model_name
    }
    with open(os.path.join(dir, "meta.txt"), 'w') as f:
        for k,v in writable_dict.items():
            f.write("{:s}: ".format(k) + str(v) + "\n")

def main(args):
    base_dir = './output/' + args.model_name + "/"
    metrics_dir = base_dir + "metrics/"
    samples_dir = base_dir + "samples/"
    training_data_dir = base_dir + "training_data/"
    
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
        os.mkdir(metrics_dir)
        os.mkdir(samples_dir)
        os.mkdir(training_data_dir)
        create_meta(base_dir, args)
    else:
        print("Folder already exists. Stopped training to prevent overwrite.")
        return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        print("Not executing on the GPU. Continue ? (y/n)")
        if input() != 'y':
            return 

    n_cores = 4

    lst_train, ndvi_train, lst_val, ndvi_val, original_lst_train, original_lst_val, original_lst, original_ndvi = process_data(args.datapath, n_cores=n_cores)
    
    # At this point all images are put in the GPU (which is faster but takes more memory, consider updating them in the dataloader loop)
    lst_train, lst_val, original_lst_train = lst_train.to(device), lst_val.to(device), original_lst_train.to(device)
    ndvi_train, ndvi_val, original_lst_val = ndvi_train.to(device), ndvi_val.to(device), original_lst_val.to(device) 
    
    # Load dataset and create data loader
    #transform = None
    #train_data = LOADDataset(lst_train, ndvi_train, transform=transform)
    #val_data = LOADDataset(lst_val, ndvi_val, transform=transform)
    
    batch_size = args.batch_size
    transform_augmentation_train = None
    train_dataset = DatasetCustom(lst_train, ndvi_train,original_lst_train)
    val_dataset = DatasetCustom(lst_val, ndvi_val, original_lst_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    #print('Length of training set: {}'.format(len(train_loader)))
    #print('Length of validating set: {}'.format(len(val_loader)))
    print('\tShape of LST input: ({},{})'.format(lst_train.shape[-2],lst_train.shape[-1]))
    print('\tShape of NVDI gradient input: ({},{})'.format(ndvi_train.shape[-2],ndvi_train.shape[-1]))

    alpha = args.alpha 
    beta = args.beta
    epochs = args.epochs
    lr = args.lr
    model_name = args.model_name
    continue_train = args.continue_train == 'True'

    model = MRUNet(res_down=False, n_resblocks=1, bilinear=True).to(device)    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss = MixedGradientLoss(device, alpha=alpha, beta=beta)

    if not continue_train:
        # TRAINING CELL
        mge_train_loss, mge_val_loss = [],[]
        mse_train_loss, mse_val_loss = [],[]
        train_loss, val_loss = [], []
        train_psnr, val_psnr = [], []
        train_ssim, val_ssim = [], []
        start = time.time()

        last_epoch = -1
        best_validation_loss = np.inf

    else:
        # Load the lists of last time training metrics
        metrics = np.load(os.path.join(metrics_dir, "metrics" + ".npy"))
        mge_train_loss, mge_val_loss = metrics[0].tolist(), metrics[5].tolist()
        mse_train_loss, mse_val_loss = metrics[1].tolist(), metrics[6].tolist()
        train_loss, val_loss = metrics[2].tolist(), metrics[7].tolist()
        train_psnr, val_psnr = metrics[3].tolist(), metrics[8].tolist()
        train_ssim, val_ssim = metrics[4].tolist(), metrics[9].tolist()
        start = time.time()

        # Model loading
        checkpoint = torch.load(base_dir + args.model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        losses = checkpoint['losses']
        vloss = losses[3]

    # TODO remove this test image thing 
    test_img_idx = 31
    #plt.imsave('original_img.png',(lst_train[test_img_idx,0,:,:]).cpu().detach().numpy())
    #plt.imsave('original_nvdi_img.png',(ndvi_train[test_img_idx,0,:,:]).cpu().detach().numpy())
    #plt.imshow((lst_train[test_img_idx,0,:,:]).cpu().detach().numpy())
    #plt.colorbar()
    #plt.savefig('original_img')
    #plt.close()
    #plt.imshow((ndvi_train[test_img_idx,0,:,:]).cpu().detach().numpy())
    #plt.colorbar()
    #plt.savefig('original_ndvi_img')
    #plt.close()
    lst_list = (lst_train[test_img_idx:test_img_idx+5,0,:,:]).cpu().detach().numpy()
    ndvi_list = (ndvi_train[test_img_idx//2 : test_img_idx//2 +5,0,:,:]).cpu().detach().numpy()
    np.savez_compressed(training_data_dir + 'original_images',lst=lst_list, ndvi=ndvi_list,  lst_original = original_lst[test_img_idx:test_img_idx+5,:,:], ndvi_original = original_ndvi[test_img_idx//2 : test_img_idx//2 +5,:,:])
    
    for epoch in range(last_epoch+1,epochs):
        
        print(f"Epoch {epoch + 1} of {epochs}")

        train_epoch_mge, train_epoch_mse, train_epoch_loss, train_epoch_psnr, train_epoch_ssim = run_model(model, train_loader, optimizer, loss, batch_size, device, phase="train")
        val_epoch_mge, val_epoch_mse, val_epoch_loss, val_epoch_psnr, val_epoch_ssim = run_model(model, val_loader, optimizer, loss, batch_size, device, phase="validation")
        
        if epoch % 5 == 0:
            outputs = np.zeros((5,lst_train.shape[1],lst_train.shape[2],lst_train.shape[3]))
            for i in range(5):
                output = model(lst_train[test_img_idx+i,:,:,:][None,:,:,:])
                outputs[i,:,:,:] = output.cpu().detach().numpy()
            #plt.imsave('output_ep_{:d}.png'.format(epoch),output[0,0,:,:].cpu().detach().numpy())
            #plt.imshow(output[0,0,:,:].cpu().detach().numpy())
            #plt.colorbar()
            #plt.savefig('output_ep_{:d}.png'.format(epoch))
            #plt.close()
            print(np.shape(outputs))
            np.savez_compressed(training_data_dir + 'output_ep_{:d}'.format(epoch),outputs=outputs)

            # Save metrics every nth iter
            losses_path = os.path.join(metrics_dir,"metrics")
            metrics = [mge_train_loss, mse_train_loss,train_loss,train_psnr,train_ssim,mge_val_loss,mse_val_loss,val_loss,val_psnr,val_ssim]
            np.save(losses_path,metrics)

        print(f"\tMGE train loss: {train_epoch_mge:.6f}")
        print(f"\tMSE train loss: {train_epoch_mse:.6f}")
        print(f"\tTrain loss: {train_epoch_loss:.6f}")
        print(f"\tVal loss: {val_epoch_loss:.6f}")
        mge_train_loss.append(train_epoch_mge)
        mse_train_loss.append(train_epoch_mse)
        train_loss.append(train_epoch_loss)
        train_psnr.append(train_epoch_psnr)
        train_ssim.append(train_epoch_ssim)
        mge_val_loss.append(val_epoch_mge)
        mse_val_loss.append(val_epoch_mse)
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
                }, base_dir + model_name)
            
            best_validation_loss = val_epoch_loss
    
    # Save metrics once again in the end 
    losses_path = os.path.join(metrics_dir,"metrics")
    metrics = [mge_train_loss, mse_train_loss,train_loss,train_psnr,train_ssim,mge_val_loss,mse_val_loss,val_loss,val_psnr,val_ssim]
    np.save(losses_path,metrics)

    end = time.time()
    print(f"Finished training in: {((end-start)/60):.3f} minutes")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch MR UNet training from tif files contained in a data folder",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datapath', help='path to directory containing training tif data')
    
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=150, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=4, type=int, help='size of batch')
    parser.add_argument('--model_name', type=str, help='name of the model')
    
    parser.add_argument('--alpha', default=0.0001, type=float, help='Weight of MGE in loss')
    parser.add_argument('--beta', default=0.9999, type=float, help='Weight of MSE in loss')
    
    parser.add_argument('--continue_train', choices=['True', 'False'], default='False', type=str, 
                        help="flag for continue training, if True - continue training the 'model_name' model, else - training from scratch")
    args = parser.parse_args()

    main(args)



