# File for defining the loss function
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize
import torchvision.transforms as T
from utility import downsampling

class MixedGradientLoss():
    def __init__(self, device, alpha=1, beta=1):
        # Kernels defined for the gradient
        self.kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        self.kernel_x = torch.FloatTensor(self.kernel_x).unsqueeze(0).unsqueeze(0).to(device)
        self.kernel_y = [[-1., -2., 1.], [0., 0., 0.], [1., 2., 1.]]
        self.kernel_y = torch.FloatTensor(self.kernel_y).unsqueeze(0).unsqueeze(0).to(device)

        # Parameters of the loss function
        self.alpha = alpha
        self.beta = beta


    def get_gradient(self, img):
        epsilon = 10e-5
        # Compute the gradient for an image using the sobel operator  
        # Orginially: torch.sqrt(torch.square(F.conv2d(img, self.kernel_x, padding=0)) + torch.square(F.conv2d(img, self.kernel_y, padding=0)))
        # Removed sqrt as the backwards loss call gave an error 
        # return torch.square(F.conv2d(img, self.kernel_x, padding=0)) + torch.square(F.conv2d(img, self.kernel_y, padding=0))
        gradient = torch.sqrt(torch.square(F.conv2d(img, self.kernel_x, padding=0)) + torch.square(F.conv2d(img, self.kernel_y, padding=0)) +epsilon)
        return gradient/torch.max(gradient)


    def get_loss(self, prediction, t_img, nvdi_img):
        '''
        prediction: Predicted image at 250 m  batchx1x256x256
        t_img: Temperature images at 1km      batchx1x256x256
        nvdi_img: NVDI gradient images at 250m         batchx1x254x254
        '''
        # Mean gradient error
        MGE = torch.square(self.get_gradient(prediction)[:,:,16:241,16:241] - nvdi_img[:,:,16:241,16:241]).mean(dim=[1,2,3])
        # Mean squared error [ 256 -> 64 (x4) ]
        MSE = torch.square(t_img[:,:,8:57,8:57] - resize(prediction,(64,64),T.InterpolationMode.BICUBIC)[:,:,8:57,8:57]).mean(dim=[1,2,3])
        # F.interpolate()

        return MGE, MSE, self.alpha * MGE + self.beta * MSE


'''
# Main to test the MixedGradientLoss 
if __name__ == "__main__":
    loss = MixedGradientLoss("cpu", alpha=1)

    img = torch.ones((1,1,4,4))
    prediction = torch.ones((1,1,16,16))
    nvdi_ = torch.ones((1,1,16,16))

    print("loss", loss.get_loss(prediction, img, nvdi_))
'''