import os
# from pymodis import downmodis
import numpy as np
import time
import pymp
from utility import *
from argparse import ArgumentParser
import shutil


def MODIS_Data_Preprocessing(year, product,delete_files, num_threads):
    sensor        = product.split(".")[0]
    root_dir      = 'MODIS/MOD_{}_{}'.format(year,sensor)
    hdfs_path     = os.path.join(root_dir, 'hdfs_files')
    tifs_1km_path = os.path.join(root_dir, 'tifs_files/1km')
    tifs_250m_path = os.path.join(root_dir, 'tifs_files/250m')
    
    os.makedirs(hdfs_path,exist_ok=1)

    if sensor == 'MOD11A1':
        os.makedirs(tifs_1km_path,exist_ok=1)
    elif sensor == 'MOD13A2':
        os.makedirs(tifs_1km_path,exist_ok=1)
    elif sensor == "MOD13Q1":
        os.makedirs(tifs_250m_path,exist_ok=1)

    print("start to processing {}".format(hdfs_path))
    hdfs = os.listdir(hdfs_path)
    hdfs.sort()
    start_time = time.time()
    # Core images with multi-core
    with pymp.Parallel(num_threads) as p:
        for index in p.range(0, len(hdfs)):

    #for index in range(0, len(hdfs)):
            hdf = hdfs[index]
            if not hdf.endswith('hdf'): continue
            hdf_path = os.path.join(hdfs_path,hdf)
            ndvi_folder = 'MODIS/MOD_{}_MOD13Q1'.format(year)
            ndvi_save_path = os.path.join(ndvi_folder, 'tifs_files/250m')
            ndvi_dir     = os.path.join(ndvi_folder, 'hdfs_files')
            os.makedirs(ndvi_save_path,exist_ok=1)

            # LST images
            if sensor=='MOD11A1':
                crop_modis(hdf_path, hdf,tifs_1km_path,ndvi_save_path,ndvi_dir, 64, (64,64))
            # NVDI 1k images
            # elif sensor=='MOD13A2':
            #     crop_modis_MOD13A2(hdf_path, hdf,tifs_1km_path, 64, (64,64))
            # # NVDI 250m images
            # elif sensor == "MOD13Q1":
            #     crop_modis_MOD13Q1(hdf_path, hdf,tifs_250m_path, 256, (256,256))
            # elif sensor == "MOD09GQ":
            #     # crop_modis_MOD09GQ()
            # elif sensor == "MOD09GQ":

    if(delete_files):
        shutil.rmtree(ndvi_dir, ignore_errors=False, onerror=None)
        shutil.rmtree(hdfs_path, ignore_errors=False, onerror=None)

    print("Using {:.4f}s to process product = {}".format(time.time()-start_time, product))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--year_begin', type=int, default=2000)
    parser.add_argument('--year_end', type=int, default=2021)
    parser.add_argument('--delete_files', type=bool, default=False)
    args = parser.parse_args()

    years = list(np.arange(args.year_begin, args.year_end))
    # Terra Surface reflectance "MOD09GQ" 
    # Aqua Suarface reflectance "MYD09GQ" 
    # LST: "MOD11A1.061"
    # NDVI_1km: "MOD13A2.061", NDVI_250m: "MOD13Q1.061"
    products = ["MOD11A1.061"] 
    # tiles to download, France is in h17v04 and h18v04 , string of tiles separated by comma
    tiles = "h18v04"
    # Cores number to use     
    num_threads = 6
    for year in years:
        for product in products:
            MODIS_Data_Preprocessing(year, product, args.delete_files, num_threads)