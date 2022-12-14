import numpy as np
import os
from skimage import io


years=[2021]

lst=None
ndvi=None

for k in range(len(years)) :
    year=years[k]

    lst_root_dir      = 'MODIS/MOD_{}_{}'.format(year,'MOD11A1')
    lst_path = os.path.join(lst_root_dir, 'tifs_files/1km')
    lst_tifs = os.listdir(lst_path)
    lst_tifs.sort()

    ndvi_root_dir      = 'MODIS/MOD_{}_{}'.format(year,'MOD09GQ')
    ndvi_path = os.path.join(ndvi_root_dir, 'tifs_files/250m')
    ndvi_tifs = os.listdir(ndvi_path)
    ndvi_tifs.sort()

    lst_tif_indexes=[]
    ndvi_tif_indexes=[]
    for i in range(len(lst_tifs)):
        lst_tif = lst_tifs[i]
        lst_tif_name = lst_tif.split('.')
        for j in range(len(ndvi_tifs)):
            ndvi_tif=ndvi_tifs[j]
            ndvi_tif_name = ndvi_tif.split('.')
            if lst_tif_name[1:-1] == ndvi_tif_name[1:-1]:
                lst_tif_indexes.append(i)
                ndvi_tif_indexes.append(j)

    lst_tifs=[lst_tifs[i] for i in lst_tif_indexes]
    ndvi_tifs=[ndvi_tifs[i] for i in ndvi_tif_indexes]

    lst_tifs.sort()
    ndvi_tifs.sort()

    i=0
    ndvi_indexes_to_delete = []
    for index in range(0, len(lst_tifs)):
            tif = lst_tifs[index]
            tif_path = os.path.join(lst_path,tif)
            current_array = io.imread(tif_path)
            if(len(current_array[:,:,0][current_array[:,:,0] == 0]) > 0 or len(current_array[:,:,1][current_array[:,:,1] == 0]) > 0) :
                ndvi_indexes_to_delete.append(index)
                continue

            if i == 0 and k==0:                
                lst =  current_array
            elif i == 1 and k==0:
                lst = np.stack([lst, current_array], axis=0)
            else :
                current_array = np.reshape(current_array,(1,64,64,2))
                lst = np.concatenate([lst, current_array], axis=0)
            i += 1 
            #if(i==10):
            #    break
    

    for j in sorted(ndvi_indexes_to_delete,reverse=True):
        del ndvi_tifs[j]

    i=0
    for index in range(0, len(ndvi_tifs)):
            tif = ndvi_tifs[index]
            tif_path = os.path.join(ndvi_path,tif)
            if i == 0 and k==0:
                ndvi =  io.imread(tif_path)
            elif i == 1 and k==0:
                ndvi = np.stack([ndvi, io.imread(tif_path)], axis=0)
            else :
                current_array = np.reshape(io.imread(tif_path),(1,256,256))
                ndvi = np.concatenate([ndvi, current_array], axis=0)
            i += 1 
            #if(i==10):
            #    break

print(np.shape(lst))
print(np.shape(ndvi))
np.savez_compressed('data_stats',lst=lst, ndvi=ndvi)



#file = open('lst', 'w')


    


#file = open('lst', 'w')



            

