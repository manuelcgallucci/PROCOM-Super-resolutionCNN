# File for defining the loss function
import torch
import torch.nn.functional as F
from utility import downsampling

class MixedGradientLoss():
    def __init__(self, device, alpha=1):
        self.kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        self.kernel_x = torch.FloatTensor(self.kernel_x).unsqueeze(0).unsqueeze(0).to(device)
        
        self.kernel_y = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        self.kernel_y = torch.FloatTensor(self.kernel_y).unsqueeze(0).unsqueeze(0).to(device)

        self.alpha = alpha

    def get_gradient(self, img):
        # Compute the gradient for an image using the sobel operator  
        return torch.sqrt(torch.square(F.conv2d(img, self.kernel_x)) + torch.square(F.conv2d(img, self.kernel_y)))

    def get_loss(self, prediction, t_img, nvdi_img):
        '''
        prediction: Predicted image at 250 m  batchx1x256x256
        t_img: Temperature images at 1km      batchx1x64x64
        nvdi_img: NVDI images at 250m         batchx1x256x256
        '''
        # Mean gradient error
        MGE = torch.square(self.get_gradient(prediction) - self.get_gradient(nvdi_img)).mean()
        # Mean squared error [ 256 -> 64 (x4) ]
        MSE = torch.square(t_img- F.interpolate(prediction, scale_factor=0.25, mode="bilinear")).mean()
        # F.interpolate()
        
        return MGE + self.alpha * MSE


'''
if __name__ == "__main__":
    loss = MixedGradientLoss("cpu", alpha=1)

    img = torch.ones((1,1,4,4))
    prediction = torch.ones((1,1,16,16))
    nvdi_ = torch.ones((1,1,16,16))

    print("loss", loss.get_loss(prediction, img, nvdi_))
'''