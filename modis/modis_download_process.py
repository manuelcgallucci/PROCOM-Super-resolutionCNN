import os
from pymodis import downmodis
import numpy as np
import pymp
import time
from utility import *
from argparse import ArgumentParser

import shutil
import multiprocessing as mp

def calculate_dates_daily(year, month):
    startdates = ["{}-{}-01".format(str(year),str(month).zfill(2))]
    enddates = []

    if month == 2:
        if year % 4 == 0:
            total_days = 29
        else:
            total_days = 28
    else:
        days_months_31 = [1,3,5,7,8,10,12]
        if month in days_months_31:
            total_days = 31
        else:
            total_days = 30

    for day in range(2, total_days+1):
        enddates.append("{}-{}-{}".format(str(year),str(month).zfill(2),str(day).zfill(2)))
        startdates.append("{}-{}-{}".format(str(year),str(month).zfill(2),str(day).zfill(2)))

    if month != 12:
        enddates.append("{}-{}-01".format(str(year),str(month+1).zfill(2)))
    else:
        enddates.append("{}-01-01".format(str(year+1)))
    
    return startdates, enddates

def MODIS_Downloader(startdate, enddate, year, product, num_threads, tiles, user="projet3a", password="Projet3AIMT"):
    sensor        = product.split(".")[0]
    hdfs_path     = 'MODIS/MOD_{}_{}/hdfs_files'.format(year,sensor)
    
    start_time = time.time()
    print("Start to download {} From {} to {}".format(hdfs_path, startdate,enddate))
    
    try:
        modisDown = downmodis.downModis(user=user,password=password,product=product,destinationFolder=hdfs_path, tiles=tiles, today=startdate, enddate=enddate)
        modisDown.connect()
        modisDown.downloadsAllDay()
        # modisDown.downloadsAllDay(clean=True, allDays=True)
    except:    
        print("Download Error {} From {} to {}".format(hdfs_path, startdate,enddate))
    print("Finish download {} From {} to {}, time cost: {:.4f}".format(hdfs_path, startdate,enddate,time.time()-start_time))

def MODIS_Downloader_DAILY(startdate, enddate, year, product, tiles, user="projet3a", password="Projet3AIMT"):
    sensor = product.split(".")[0]
    hdfs_path = 'MODIS/MOD_{}_{}/hdfs_files'.format(year,sensor)
    
    start_time = time.time()
    print("Start to download {} From {} to {}".format(hdfs_path, startdate,enddate))
    
    try:
        modisDown = downmodis.downModis(user=user,password=password,product=product,destinationFolder=hdfs_path, tiles=tiles, today=startdate, enddate=enddate)
        modisDown.connect()
        modisDown.downloadsAllDay()
    # modisDown.downloadsAllDay(clean=True, allDays=True)
    except:
        print("Download Error {} From {} to {}".format(hdfs_path, startdate,enddate))
    
    print("Finish download {} From {} to {}, time cost: {:.4f}".format(hdfs_path, startdate,enddate,time.time()-start_time))

def MODIS_Data_Preprocessing(year, product, num_threads, delete_files=False):
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

    ndvi_folder = 'MODIS/MOD_{}_MOD09GQ'.format(year)
    ndvi_save_path = os.path.join(ndvi_folder, 'tifs_files/250m')
    ndvi_dir     = os.path.join(ndvi_folder, 'hdfs_files')
    os.makedirs(ndvi_save_path,exist_ok=1)

    list_ndvi =  os.listdir(ndvi_dir)
    indexes_to_delete=[]
    for index in range(len(list_ndvi)):
        if not list_ndvi[index].endswith('hdf'):
            indexes_to_delete.append(index)
    for j in sorted(indexes_to_delete,reverse=True):
        del list_ndvi[j]
    list_ndvi.sort()

    print("start to processing {}".format(hdfs_path))
    hdfs = os.listdir(hdfs_path)
    hdfs.sort()
    start_time = time.time()
    # Core images with multi-core
    with pymp.Parallel(num_threads) as p:
        for index in p.range(0, len(hdfs)):

            hdf = hdfs[index]
            if not hdf.endswith('hdf'): continue
            hdf_path = os.path.join(hdfs_path,hdf)

            # LST images
            if sensor=='MOD11A1':
                crop_modis(hdf_path, hdf,tifs_1km_path,ndvi_save_path,list_ndvi,ndvi_dir, 64, (64,64))

    if delete_files:
        shutil.rmtree(ndvi_dir, ignore_errors=False, onerror=None)
        shutil.rmtree(hdfs_path, ignore_errors=False, onerror=None)

    print("Using {:.4f}s to process product = {}".format(time.time()-start_time, product))
   
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--year_begin', type=int, default=2015)
    parser.add_argument('--year_end', type=int, default=2016)
    args = parser.parse_args()

    years = list(np.arange(args.year_begin, args.year_end))
    # Terra Surface reflectance MOD09GQ 
    # Aqua Suarface reflectance MYD09GQ 
    # LST: "MOD11A1.061"
    products = ["MOD11A1.061","MOD09GQ.061"]
    
    tiles = "h18v04" # tiles to download, France is in h17v04 and h18v04 , string of tiles separated by comma
    num_threads = 6 # Cores number to use 
    for year in years:
        for month in range(1,13):
            
            # Download data
            startdates, enddates = calculate_dates_daily(year, month)
            total_days = len(startdates)

            n_processes = 18
            for product in products:
                
                pool = mp.Pool(n_processes)
                results = []

                for i in range(total_days):
                    res = pool.apply_async(MODIS_Downloader_DAILY, (startdates[i], enddates[i], year, product, tiles))
                    results.append(res)
                
                for i in range(total_days):
                    results[i].get()
                    
                pool.close()
                pool.join()

            # Process both products and delete all hdf files 
            MODIS_Data_Preprocessing(year, products[0], num_threads, delete_files=True)


