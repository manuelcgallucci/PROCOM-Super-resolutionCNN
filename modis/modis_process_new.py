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
                process_hdf(hdf_path, hdf,tifs_1km_path,ndvi_save_path,list_ndvi,ndvi_dir, 64, (64,64))

    if delete_files:
        shutil.rmtree(ndvi_dir, ignore_errors=False, onerror=None)
        shutil.rmtree(hdfs_path, ignore_errors=False, onerror=None)

    print("Using {:.4f}s to process product = {}".format(time.time()-start_time, product))

def process_hdf(hdf_path, hdf_name, save_dir,ndvi_save_path,list_ndvi,ndvi_dir,step=64,size=(64,64)):
    """
    INPUT:
    hdf_path = input image path to be cropped | or hdf file path ("/a/b/c.hdf")
    save_dir = directory for saving cropped images
    step, size: parameters of "sliding_window()"
    OUTPUT: images cropped from the image in hdf_path, saved to save_dir
    """
    if not hdf_path.endswith('hdf'): 
        print("Not hdf file Sorry!")
        return 

    read_val = read_modis(hdf_path)
    if read_val is None:
        print("Cannot handle this MODIS file: ", hdf_path, ". Please check it again")
        return

    img_day, img_night, cols, rows, projection, geotransform = read_val
    
    img_days = []
    img_nights = []
    img_cropped_names = []
    ndvis = []
    ndvi_names = []
    ndvi_geotransforms = []
    geotransform2s = []
    cols2, rows2 = size

    if img_day is None or img_night is None:
        print("Cannot handle this MODIS file: ", hdf_path, ". Please check it again")
        return

    hdf_name_list = hdf_name.split(".")
    # For day image
    win_count = 0
    for (x,y,window) in sliding_window(img_day, step, size):
            if window.shape[0] != size[0] or window.shape[1] != size[1]:
                    continue

            img_cropped_name = hdf_name_list[0] + "." + hdf_name_list[1] + ".{}.tif".format(str(win_count).zfill(4))
            img_cropped = window
            geotransform2 = np.asarray(geotransform)
            geotransform2[0] = geotransform[0]+x*geotransform[1] # 1st coordinate of top left pixel of the image 
            geotransform2[3] = geotransform[3]+y*geotransform[5] # 2nd coordinate of top left pixel of the image
            geotransform2=tuple(geotransform2)

            img_cropped_names.append(img_cropped_name)
            img_days.append(img_cropped)
            geotransform2s.append(geotransform2)
            
            win_count += 1
    #print("Number of cropped day images", win_count)

    # For night image
    win_count = 0
    for (x,y,window) in sliding_window(img_night, step, size):
        if window.shape[0] != size[0] or window.shape[1] != size[1]:
                continue
        # save_path = os.path.join(save_dir,img_cropped_name)
        img_cropped = window
        # np.save(save_path,img_cropped)
        img_nights.append(img_cropped)
        win_count += 1
    #print("Number of cropped night images night", win_count)

    ndvi_read_value = get_corresponding_ndvi(list_ndvi,ndvi_dir,hdf_name)
    if ndvi_read_value is None :
        return 
    red, NIR, NDVI, ndvi_projection, ndvi_geotransform = ndvi_read_value
    reds = []
    NIRs = []

    win_count = 0
    for (x,y,window) in sliding_window(NDVI, 256, (256,256)):
            if window.shape[0] != 256 or window.shape[1] != 256:
                    continue

            img_cropped_name = img_cropped_names[win_count]
            image_name_list = img_cropped_name.split(".")
            image_name_list[0] = "MOD09GQ"
            save_name = '.'.join(image_name_list)
            ndvi_names.append(save_name)

            img_cropped = window
            ndvis.append(img_cropped)     

            geotransform2 = np.asarray(ndvi_geotransform)
            geotransform2[0] = ndvi_geotransform[0]+x*ndvi_geotransform[1] # 1st coordinate of top left pixel of the image 
            geotransform2[3] = ndvi_geotransform[3]+y*ndvi_geotransform[5] # 2nd coordinate of top left pixel of the image
            geotransform2=tuple(geotransform2) 
            ndvi_geotransforms.append(geotransform2)      
            win_count += 1
    
    """ for (x,y,window) in sliding_window(red, 256, (256,256)):
            if window.shape[0] != 256 or window.shape[1] != 256:
                    continue
            reds.append(window)
    
    for (x,y,window) in sliding_window(NIR, 256, (256,256)):
            if window.shape[0] != 256 or window.shape[1] != 256:
                    continue
            NIRs.append(window) """
    
    # Save images and metadata into .tif file
    for i in range(len(img_cropped_names)):
        save_path = os.path.join(save_dir,img_cropped_names[i])
        succes = save_tif(save_path, img_days[i], img_nights[i], cols2, rows2, projection, geotransform2s[i])
        if(succes):
            if(len(ndvis[i][np.isnan(ndvis[i])])==0):
                save_path_ndvi = os.path.join(ndvi_save_path,ndvi_names[i])
                save_tif_MOD09GQ(save_path_ndvi, ndvis[i], 256, 256, ndvi_projection, ndvi_geotransforms[i])
            else:
                os.remove(save_path) 
                 
            """ downscaled_red = skimage.transform.downscale_local_mean(reds[i],(4,4))
            downscaled_nir = skimage.transform.downscale_local_mean(NIRs[i],(4,4))
            print(np.shape(downscaled_nir))
            downscaled_ndvi =  (downscaled_nir-downscaled_red)/(downscaled_nir+downscaled_red)
            plt.imshow(downscaled_ndvi)
            plt.colorbar()
            plt.savefig('./samples/'+ndvi_names[i]+'.png')  
            plt.close()   
            plt.imshow(img_days[i])
            plt.colorbar()
            plt.savefig('./samples/'+img_cropped_names[i]+'.png')
            plt.close() """

def get_corresponding_ndvi(list_ndvi,ndvi_dir,image_name):
    image_name_string = image_name.split(".")
    image_day = image_name_string[1]
    
    for ndvi in list_ndvi :
        ndvi_strings = ndvi.split(".")
        if(image_day == ndvi_strings[1]):
            ndvi_path = os.path.join(ndvi_dir, ndvi)
            if os.path.exists(ndvi_path):
                read_value = read_modis_MOD09GQ(ndvi_path)
                if read_value is None :
                    print("Cannot handle this MODIS file: ", ndvi_path, ". Please check it again")
                    return None
                qa, red, NIR, cols, rows, projection, geotransform = read_value

                if qa is None or red is None or NIR is None:
                    print("Cannot handle this MODIS file: ", ndvi_path, ". Please check it again")
                    return None

                ndvi = (NIR-red)/(NIR+red)
                return red, NIR, ndvi, projection, geotransform

    return None
   
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
    num_threads = 6
    n_processes = 18
    for year in years:
        for month in range(1,13):
            
            # Download data
            startdates, enddates = calculate_dates_daily(year, month)
            total_days = len(startdates)

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


