import torch
from torch.nn import functional as F

class MixedGradientLoss():
    def __init__(self, device, alpha=1):
        # Definig a sobel kernel for the gradient optimized for edge detection
        self.kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        self.kernel_x = torch.FloatTensor(self.kernel_x).unsqueeze(0).unsqueeze(0).to(device)

        self.kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        self.kernel_y = torch.FloatTensor(self.kernel_y).unsqueeze(0).unsqueeze(0).to(device)

        # Parameters of the loss function
        self.alpha = alpha
        

    def get_gradient(self, img):
        # Perform convolution of the image with the Sobel kernels
        grad_x = F.conv2d(img, self.kernel_x, padding=1)
        grad_y = F.conv2d(img, self.kernel_y, padding=1)

        # Calculate the gradient magnitude
        grad_mag = torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2))
        # grad_dir = torch.atan2(grad_y, grad_x)
        return grad_mag

    def get_loss(self, prediction, t_img, nvdi_img):
        # Calculate the mean squared error between the prediction and temperature image
        mse = F.mse_loss(prediction, t_img)

        # Calculate the gradient of the prediction image
        pred_grad_mag = self.get_gradient(prediction)

        # Calculate the mean squared error between the prediction gradient and NVDI gradient
        grad_mse = F.mse_loss(pred_grad_mag, nvdi_img)

        # Weight the two losses and return the total loss
        total_loss = self.alpha * mse + (1 - self.alpha) * grad_mse
        return total_loss

# Main to test the MixedGradientLoss
if __name__ == "__main__":
    loss = MixedGradientLoss("cpu", alpha=1)

    img = torch.ones((1,1,4,4))
    # making img and prediction the same size
    img = F.interpolate(img, size=(16, 16), mode='bilinear', align_corners=True)

    prediction = torch.ones((1,1,16,16))

    nvdi_ = torch.ones((1,1,16,16))


    print("loss", loss.get_loss(prediction, img, nvdi_))


