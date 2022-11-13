# File for defining the loss function
import torch.nn.functional as F
from torchmetrics.functional import mean_squared_error
import torch

class GradientLoss():
    def __init__(self, device, alpha=1)
        self.kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        self.kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)
        
        self.kernel_y = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        self.kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)

        self.alpha = alpha

    def get_gradient(self, img):
        # Compute the gradient for an image using the sobel operator  
        return torch.sqrt(torch.square(F.conv2d(img, self.kernel_x)) + torch.square(F.conv2d(img, self.kernel_y)))

    def apply_loss(self, prediction, t_img, nvdi_img):
        '''
        prediction: Predicted image at 250 m  256x256
        t_img: Temperature images at 1km      64x64
        nvdi_img: NVDI images at 250m         256x256
        '''
        # Original loss function 
        # mse_img = ((disp - img)**2).mean()

        # Mean gradient error
        MGE = mean_squared_error(get_gradient(prediction), get_gradient(nvdi_img))
        # Mean squared error
        MSE = mean_squared_error(t_img, upscale()



        return MGE + self.alpha * MSE