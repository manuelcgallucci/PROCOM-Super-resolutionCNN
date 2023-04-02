import numpy as np
import matplotlib.pyplot as plt 
import utility as ut
from model import MRUNet
import torch
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim_sk
import cv2
from loss import MixedGradientLoss

def plot_save(x, title, path, limits=None):
    if limits is None:
        plt.imshow(x)
    else:
        plt.imshow(x, vmin=limits[0], vmax=limits[1])
    plt.colorbar()
    plt.title(title)
    plt.savefig(path)
    plt.close()

def analyze_arr(x, name):
    print("Array: ", str(name), ":", str(x.shape))

    print("\tMax:", np.max(x))
    print("\tMin:", np.min(x))
    print("\tNan values:", np.sum(np.isnan(x)))

def interpolate_nan_and_zeros(arr, verbose=True):
    mask = np.logical_or( np.isnan(arr), arr == 0)
    k = 0
    if verbose: print("Iteration {:8d}".format(k), np.sum(mask))
    while np.sum(mask) > 0:
        # Define a mask for the 0 values in the array
        # Find the indices of the 0 values in the array
        nan_indices = np.argwhere(mask)

        # Loop over the 0 indices and replace each value with the average of its neighboring pixels
        # Disgusting way to do it but it works for now
        for i, j in nan_indices:
            neighbors = []
            if i > 0:
                if not np.isnan(arr[i-1, j]) and arr[i-1, j] != 0:
                    neighbors.append(arr[i-1, j])
            if i < arr.shape[0] - 1:
                if not np.isnan(arr[i+1, j]) and arr[i+1, j] != 0:
                    neighbors.append(arr[i+1, j])
            if j > 0:
                if not np.isnan(arr[i, j-1]) and arr[i, j-1] != 0:
                    neighbors.append(arr[i, j-1])
            if j < arr.shape[1] - 1:
                if not np.isnan(arr[i, j+1]) and arr[i, j+1] != 0:
                    neighbors.append(arr[i, j+1])
                    
            if j < arr.shape[1] - 1 and i < arr.shape[0] - 1:
                if not np.isnan(arr[i+1, j+1]) and arr[i+1, j+1] != 0:
                    neighbors.append(arr[i+1, j+1])  
            if i > 0 and j > 0:
                if not np.isnan(arr[i-1, j-1]) and arr[i-1, j-1] != 0:
                    neighbors.append(arr[i-1, j-1])
            if j < arr.shape[1] - 1 and i > 0 :
                if not np.isnan(arr[i-1, j+1]) and arr[i-1, j+1] != 0:
                    neighbors.append(arr[i-1, j+1])
            if i < arr.shape[1] - 1 and j > 0 :
                if not np.isnan(arr[i+1, j-1]) and arr[i+1, j-1] != 0:
                    neighbors.append(arr[i+1, j-1])
            if len(neighbors) > 0:
                arr[i, j] = np.mean(neighbors)
                
        mask = np.logical_or( np.isnan(arr), arr == 0)
        
        k = k + 1
        if verbose: print("Iteration {:8d}".format(k), np.sum(mask))
        

    return arr

def plot_other_methods(limits=None):
    path = "./data/"
    out_path = "./validation/" 
    names = ["AATPRK_final.npy", "ATPRK_final.npy", "TSHARP_final.npy"]
    out_names = ["final_"+str(x)+".png" for x in ["aatprk", "atprk", "tsharp"]]
    titles = [str(x)+" cropped LST image" for x in ["aatprk", "atprk", "tsharp"]]

    imgs = []
    for k, name in enumerate(names):
        img = np.load(path+name)
        plot_save(img, titles[k], out_path+out_names[k], limits=limits)
        imgs.append(img)

    return imgs

# Stolen from utility.py
def psnr_notorch(label, outputs):
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    First we need to convert torch tensors to NumPy operable.
    """
    img_diff = outputs - label
    rmse = np.sqrt(np.mean((img_diff) ** 2))
    if rmse == 0 or rmse==np.inf:
        return 0, np.inf
    else:
        PSNR = 20 * np.log10((np.max(label) - np.min(label)) / rmse)
        #PSNR = 20 * np.log10(np.max(label) / rmse)
        return PSNR, rmse

# Stolen from utility.py
def ssim_notorch(label, outputs):
    ssim_map = ssim_sk(label, outputs, data_range=label.max() - label.min())
    return ssim_map


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

def calculate_metrics(cropped_aster, cropped_output_lst, cropped_aatprk, cropped_atprk, cropped_tsharp, cropped_bilinear=None, plot=True):
	# This done in this way give the same result as psnr_notorch(cropped_aster, cropped_output_lst)
	if plot:
		print("Model PSNR:", peak_signal_noise_ratio(cropped_aster, cropped_output_lst, data_range=cropped_aster.max()-cropped_aster.min()))
		print("Aatprk PSNR:", peak_signal_noise_ratio(cropped_aster, cropped_aatprk, data_range=cropped_aster.max()-cropped_aster.min()))
		print("Atprk PSNR:", peak_signal_noise_ratio(cropped_aster, cropped_atprk, data_range=cropped_aster.max()-cropped_aster.min()))
		print("Tsharp PSNR:", peak_signal_noise_ratio(cropped_aster, cropped_tsharp, data_range=cropped_aster.max()-cropped_aster.min()))
		print("")
		print("Model SSIM:", ssim_notorch(cropped_aster, cropped_output_lst))
		print("Aatprk SSIM:", ssim_notorch(cropped_aster, cropped_aatprk))
		print("Atprk SSIM:", ssim_notorch(cropped_aster, cropped_atprk))
		print("Tsharp SSIM:", ssim_notorch(cropped_aster, cropped_tsharp))
	else:
		psnrs = np.array([peak_signal_noise_ratio(cropped_aster, cropped_output_lst, data_range=cropped_aster.max()-cropped_aster.min()),
			peak_signal_noise_ratio(cropped_aster, cropped_aatprk, data_range=cropped_aster.max()-cropped_aster.min()),
			peak_signal_noise_ratio(cropped_aster, cropped_atprk, data_range=cropped_aster.max()-cropped_aster.min()),
			peak_signal_noise_ratio(cropped_aster, cropped_tsharp, data_range=cropped_aster.max()-cropped_aster.min()),
			peak_signal_noise_ratio(cropped_aster, cropped_bilinear, data_range=cropped_aster.max()-cropped_aster.min())])		
		ssims = np.array([ssim_notorch(cropped_aster, cropped_output_lst),
			ssim_notorch(cropped_aster, cropped_aatprk), 
			ssim_notorch(cropped_aster, cropped_atprk),
			ssim_notorch(cropped_aster, cropped_tsharp),
			ssim_notorch(cropped_aster, cropped_bilinear)])
		return psnrs, ssims
	
def calculate_metrics_devided(cropped_aster, cropped_output_lst, cropped_aatprk, cropped_atprk, cropped_tsharp, cropped_bilinear, division=2):
	size_ = cropped_output_lst.shape[0]
	
	psnr_lst = np.zeros((division**2,1))
	psnr_aatprk = np.zeros((division**2,1))
	psnr_atrprk = np.zeros((division**2,1))
	psnr_tsharp = np.zeros((division**2,1))
	psnr_bilinear = np.zeros((division**2,1))
	
	ssim_lst = np.zeros((division**2,1))
	ssim_aatprk = np.zeros((division**2,1))
	ssim_atrprk = np.zeros((division**2,1))
	ssim_tsharp = np.zeros((division**2,1))
	ssim_bilinear = np.zeros((division**2,1))
	k = 0
	for i in range(division):
		cut_size_ = size_ // division
		for j in range(division):
			
			aster = cropped_aster[i*cut_size_:(i+1)*cut_size_,j*cut_size_:(j+1)*cut_size_] 
			lst = cropped_output_lst[i*cut_size_:(i+1)*cut_size_,j*cut_size_:(j+1)*cut_size_]
			aatprk = cropped_aatprk[i*cut_size_:(i+1)*cut_size_,j*cut_size_:(j+1)*cut_size_]
			atprk = cropped_atprk[i*cut_size_:(i+1)*cut_size_,j*cut_size_:(j+1)*cut_size_]
			tsharp = cropped_tsharp[i*cut_size_:(i+1)*cut_size_,j*cut_size_:(j+1)*cut_size_]
			bilinear = cropped_bilinear[i*cut_size_:(i+1)*cut_size_,j*cut_size_:(j+1)*cut_size_]

			psnrs, ssims = calculate_metrics(aster, lst, aatprk, atprk, tsharp, cropped_bilinear=bilinear, plot=False)
			psnr_lst[k] = psnrs[0]
			psnr_aatprk[k] = psnrs[1] 
			psnr_atrprk[k] = psnrs[2]
			psnr_tsharp[k] = psnrs[3]
			psnr_bilinear[k] = psnrs[4]
			
			ssim_lst[k] = ssims[0]
			ssim_aatprk[k] = ssims[1]
			ssim_atrprk[k] = ssims[2] 
			ssim_tsharp[k] = ssims[3] 
			ssim_bilinear[k] = ssims[4] 
			k = k + 1
		
	names = ["Model", "Aatprk", "Atprk", "Tsharp", "Bilinear"]
	metrics = [psnr_lst, psnr_aatprk, psnr_atrprk, psnr_tsharp, psnr_bilinear]
	
	for m, name in enumerate(names):
		print("\t"+name+" PSNR", np.mean(metrics[m]), np.max(metrics[m]), np.min(metrics[m]))

	metrics = [ssim_lst, ssim_aatprk, ssim_atrprk, ssim_tsharp, ssim_bilinear]
	print("")
	for m, name in enumerate(names):
		print("\t"+name+" SSIM", np.mean(metrics[m]), np.max(metrics[m]), np.min(metrics[m]))

	
def main(model_name="test_loss3", max_val=333.3200048828125 ,data_dir="./data/", base_dir="./validation/"):
    
	# Limit for the color bars in the plot. To keep them all the same range
	colorbar_limits = (260, 280)

	aster_img = np.load(data_dir+"aster_final.npy") # Nothing changed
	lst_img = np.load(data_dir+"lst_final.npy") # 0 interpolated
	ndvi_img = np.load(data_dir+"ndvi250final.npy") # 0 replaced by 260

	# Plot the originial images
	#plot_save(aster_img, "Aster image", base_dir+"crop_aster.png") # Has 0 and NaN in the border 
	#plot_save(lst_img, "LST image", base_dir+"crop_lst.png") # Has 0 in the border and inside 
	#plot_save(ndvi_img, "NDVI image", base_dir+"crop_ndvi.png") # Has 0 in the border

	# See the 0 and NaNs in the arrays 
	#analyze_arr(aster_img, "Aster image")
	#analyze_arr(lst_img, "LST image")
	#analyze_arr(ndvi_img, "NDVI image")

	# Interpolate the values, this is only useful for some 0 and NaNs found inside the image, not for the big border around the image
	interpolated_aster = interpolate_nan_and_zeros(aster_img, verbose=False)
	interpolated_lst = interpolate_nan_and_zeros(lst_img, verbose=False)
	interpolated_ndvi = interpolate_nan_and_zeros(ndvi_img, verbose=False)

	# Generate the bilinear LST as a baseline
	biliear_lst = cv2.resize(interpolated_lst, (256, 256), cv2.INTER_CUBIC)

	print("PSNR bilinear:", psnr_notorch(interpolated_aster, biliear_lst))
	print("SSIM bilinear:", ssim_notorch(interpolated_aster, biliear_lst))
	print("")
	print("PSNR bilinear edge:", psnr_notorch(interpolated_aster[16:-16,16:-16], biliear_lst[16:-16,16:-16]))
	print("SSIM bilinear edge:", ssim_notorch(interpolated_aster[16:-16,16:-16], biliear_lst[16:-16,16:-16]))

	cropped_bilinear = biliear_lst[16:-16,16:-16]
	plot_save(biliear_lst, "Bilinear image", base_dir+"final_binlinear.png", limits=colorbar_limits) # Has 0 and NaN in the border 
	plot_save(cropped_bilinear, "Bilinear image without edges", base_dir+"final_binlinear_edge.png", limits=colorbar_limits) # Has 0 and NaN in the border 

	plot_save(interpolated_aster, "Interpolated Aster image", base_dir+"interpolated_aster.png") 
	plot_save(interpolated_lst, "Interpolated LST image", base_dir+"interpolated_lst.png") 
	plot_save(interpolated_ndvi, "Interpolated NDVI image", base_dir+"interpolated_ndvi.png") 

	# Use the interpolated LST and NDVI to generate the output of the image
	### ================================== ###
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	if device != "cuda":
		print("No device available. Stopping the program. device:", device)
		return 
	else:
		print("GPU not accesible not running the model output is not correct")

		loss = MixedGradientLoss(device=device)

		# Load pretrained model
		model_MRUnet = MRUNet(res_down=False, n_resblocks=1, bilinear=True).to(device)
		model_MRUnet.load_state_dict(torch.load("./output/"+model_name+"/"+model_name)['model_state_dict'])

		# Normalization factor and scale dividing by max_val after or before is the same 
		input_ = cv2.resize(interpolated_lst, (256, 256), cv2.INTER_CUBIC)
		input_ = torch.tensor(np.reshape(input_ / max_val , (1,1,256,256)), dtype=torch.float).to(device)

		#input_ = input_[None,None,:,:]
		output_lst = model_MRUnet(input_).cpu().detach().numpy() * max_val

		output_lst = output_lst[0,0,:,:]
		np.save("model_output", output_lst)

	# Crop the images to calculate the PSNR since because of the edge effects 
	cropped_output_lst = output_lst[16:-16,16:-16]
	cropped_aster = aster_img[16:-16,16:-16]

	# plot_save(cropped_output_lst, "Output cropped LST image", base_dir+"final_lst.png", limits=colorbar_limits)
	# plot_save(cropped_aster, "Aster cropped LST image", base_dir+"final_aster.png", limits=colorbar_limits)
	cropped_aatprk, cropped_atprk, cropped_tsharp = plot_other_methods(limits=colorbar_limits)

	# crop the original method images to compare them with the model results
	cropped_aatprk = cropped_aatprk[16:-16,16:-16]
	cropped_atprk = cropped_atprk[16:-16,16:-16]
	cropped_tsharp = cropped_tsharp[16:-16,16:-16]

	# Save the histograms
	# save_double_histograms(cropped_aster, cropped_output_lst, "ASTER", "LST output", "./", "hist_lst.png")
	# save_double_histograms(cropped_aster, cropped_aatprk, "ASTER", "AATPRK", "./", "hist_aatprk.png")
	# save_double_histograms(cropped_aster, cropped_atprk, "ASTER", "ATPRK", "./", "hist_atprk.png")
	# save_double_histograms(cropped_aster, cropped_tsharp, "ASTER", "TSharp", "./", "hist_tsharp.png")

	# Print the metrics (psnr and ssmi for the images)
	calculate_metrics(cropped_aster, cropped_output_lst, cropped_aatprk, cropped_atprk, cropped_tsharp)

	# Print the metrics but subdivide the images into smaller sections
	# for division in range(2,5):
	# 	print("Division: ", division)
	# 	calculate_metrics_devided(cropped_aster, cropped_output_lst, cropped_aatprk, cropped_atprk, cropped_tsharp, cropped_bilinear, division=division)


if __name__ == "__main__":
    main()
