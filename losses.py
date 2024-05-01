import torch
from torch import nn
from torchvision.models.optical_flow import raft_small
from torchvision.models.optical_flow import Raft_Small_Weights
import torch.utils.checkpoint as checkpoint

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt( diff**2 + self.eps**2))
        return loss



class RAFT_LOSS(nn.Module):
    def __init__(self, eps = 1e-8, stride = 2, device = None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        weights = Raft_Small_Weights.DEFAULT
        self.net = raft_small(weights=weights.DEFAULT, progress=False)
        self.t = weights.transforms()
        self.stride = stride
        self.eps = eps
        
    def chkpoint(self, ten1, ten2):
        return self.net(*self.t(ten1, ten2), num_flow_updates = 1)[-1]
        
    def forward(self, upsampled_vids, hq_vids):
        b,t, c, h, w = upsampled_vids.shape

        ten1 = upsampled_vids[:, :-1:self.stride, :, :, :].reshape(-1, c, h, w)
        ten2 = upsampled_vids[:, 1::self.stride, :, :, :].reshape(-1, c, h, w)
        
        flow_upsampled = checkpoint.checkpoint(self.chkpoint, ten1, ten2)
        max_norm = torch.sum(flow_upsampled**2, dim=1).sqrt().max()
        
        epsilon = torch.finfo((flow_upsampled).dtype).eps
        normalized_flow_upsampled = flow_upsampled / (max_norm + epsilon)
        
        ten1 = hq_vids[:, :-1:self.stride, :, :, :].reshape(-1, c, h, w)
        ten2 = hq_vids[:, 1::self.stride, :, :, :].reshape(-1, c, h, w)      
        
        flow_hq = checkpoint.checkpoint(self.chkpoint, ten1, ten2)
        max_norm = torch.sum(flow_hq**2, dim=1).sqrt().max()
        epsilon = torch.finfo((flow_hq).dtype).eps
        normalized_flow_hq = flow_hq / (max_norm + epsilon)

        l = (normalized_flow_upsampled - normalized_flow_hq)**2 + self.eps**2
        l[torch.isnan(l)] = 0.0
        return torch.mean(l)
    
def charbonnier_loss_func(eps = 1e-8):
    closs = CharbonnierLoss(eps)
    def criterion(upsampled_vids, hq_vids):
        return closs(upsampled_vids, hq_vids)
    
    return criterion

def flow_loss_func(eps = 1e-8):
    floss = RAFT_LOSS(eps).cuda()
    def criterion(upsampled_vids, hq_vids):
        return floss(upsampled_vids, hq_vids)
    
    return criterion

def charbonnier_flow_loss(w_c = 1.0, w_f = 0.5):
    charbonnier_criterion = charbonnier_loss_func()
    flow_criterion = flow_loss_func()
    
    def criterion(upsampled_vids, hq_vids):
        return w_c*charbonnier_criterion(upsampled_vids, hq_vids) + w_f*flow_criterion(upsampled_vids, hq_vids)
    
    return criterion