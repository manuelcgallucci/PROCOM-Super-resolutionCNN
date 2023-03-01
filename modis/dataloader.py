from torch.utils.data import Dataset
import numpy as np

# Define Dataset
class DatasetCustom(Dataset):
    def __init__(self, data_lst, data_nvdi, original_lst):
        # Data is an array of 2d images of size N x w x h 
        self.data_lst = data_lst
        self.data_nvdi = data_nvdi
        self.original_lst = original_lst
        # assert self.data_lst.shape[0] == self.data_nvdi.shape[0], "NVDI and LST number of images is different. Indexing will break"
        
    def __len__(self):
        return self.data_lst.shape[0]
    
    def __getitem__(self, index):
        return self.data_lst[index,:,:], self.data_nvdi[int(index/2),:,:], self.original_lst[index,:,:]