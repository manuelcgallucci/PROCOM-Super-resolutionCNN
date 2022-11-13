# File for defining the loss function
impo





def get_loss(prediction, t_img):
    '''
    prediction: Predicted image at 250 m
    t_img: Temperature images at 1km
    nvdi_img: NVDI images at 250m
    '''
    # Original loss function 
    # mse_img = ((disp - img)**2).mean()
    return mse_img