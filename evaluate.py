from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch
from tqdm import tqdm
def y_channel_from_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to Y from YCbCr.

    Args:
        image (torch.Tensor): RGB Image to be converted to Y.

    Returns:
        torch.Tensor: Y channel of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta = .5
    y: torch.Tensor = .299 * r + .587 * g + .114 * b
    # cb: torch.Tensor = (b - y) * .564 + delta
    # cr: torch.Tensor = (r - y) * .713 + delta
    return y.unsqueeze(1)


def evaluate(model, criterion, dataloader, y_channel_only = False, device = None):
    running_loss = 0.0
    if device is None:
        device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    
    psnr_model = PeakSignalNoiseRatio().to(device)
    ssim_model = StructuralSimilarityIndexMeasure().to(device)
    
    n = 0
    for batch in dataloader:
        # import pdb; pdb.set_trace()
        x = batch['lq'].to(device)
        y = batch['hq'].to(device)
        b, t, c, h, w = y.shape
        
        n+= 1
        model.eval()
        with torch.no_grad():
            upsampled_vids = model(x)
            
            loss = criterion(y, upsampled_vids)
            

        
        upsampled_vids = upsampled_vids.view(b*t, c, h, w)
        y = y.view(b*t, c, h, w)
        
        if y_channel_only:
            upsampled_vids = y_channel_from_rgb(upsampled_vids)
            y = y_channel_from_rgb(y)
        
        
        running_loss += loss.item()
        psnr_model( upsampled_vids, y)
        ssim_model(upsampled_vids, y)
    return running_loss/n, psnr_model.compute().item(), ssim_model.compute().item()