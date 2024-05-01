from SRGAN_PyTorch.model import _ResidualConvBlock, _UpsampleBlock, srresnet_x4
from SRGAN_PyTorch.utils import load_pretrained_state_dict
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np

import torch.utils.checkpoint as checkpoint
from positional_encodings.torch_encodings import PositionalEncodingPermute3D
from torchvision.models.optical_flow import raft_small, raft_large
from torchvision.models.optical_flow import Raft_Small_Weights, Raft_Large_Weights
from torchvision.models.optical_flow import raft_small, raft_large
from torchvision.models.optical_flow import Raft_Small_Weights, Raft_Large_Weights
from torch.autograd import Variable

# __all__ = [ZeroConvolution, ControlledSRResNet, ControlledSRResNetWithAttention, ControlledSRResNetWithFlow]

class ZeroConvolution(nn.Module):
    def __init__(self, channels):
        super(ZeroConvolution, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.conv.weight.data.fill_(0)  # Initialize weights to zero
        self.conv.bias.data.fill_(0)    # Initialize biases to zero

    def forward(self, x):
        # Apply zero-convolution
        return self.conv(x)



class ControlledSRResNetWithFlow(nn.Module):
    def __init__(
            self,
            path_to_weights: str,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            num_rcb: int = 16,
            
    ) -> None:
        super().__init__()
        srresnet = srresnet_x4(in_channels = in_channels,
                                 out_channels = out_channels,
                                 channels = channels,
                                 num_rcb = num_rcb)
        weights = Raft_Large_Weights.DEFAULT
        self.flow = raft_large(weights=weights, progress=False).eval()
        for params in self.flow.parameters():
            params.requires_grad = False
        
        self.transforms = weights.transforms()
        
        self.model = load_pretrained_state_dict(srresnet, compile_state = False, model_weights_path = path_to_weights)
        self.entry_zero_conv = ZeroConvolution(in_channels)
        # Low frequency information extraction layer
        conv1_clone = nn.Sequential(
            nn.Conv2d(in_channels, channels, (9, 9), (1, 1), (4, 4)),
            nn.PReLU(),
        )
        conv1_clone.load_state_dict(self.model.conv1.state_dict())

        # High frequency information extraction block
        trunk_clone = []
        for i in range(num_rcb):
            residual_block_clone = _ResidualConvBlock(channels)
            residual_block_clone.load_state_dict(self.model.trunk[i].state_dict())
            trunk_clone.append(residual_block_clone)
        trunk_clone = nn.Sequential(*trunk_clone)

        # High-frequency information linear fusion layer
        conv2_clone = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )
        conv2_clone.load_state_dict(self.model.conv2.state_dict())


        self.control_net = nn.ModuleDict()

        self.control_net['conv1'] = nn.ModuleDict()
        self.control_net.conv1['module'] = conv1_clone
        self.control_net.conv1['zero_conv'] = ZeroConvolution(channels)

        self.control_net['trunk'] = nn.ModuleDict()
        for i in range(num_rcb):
            self.control_net.trunk[str(i)] = nn.ModuleDict()
            self.control_net.trunk[str(i)]['module'] = trunk_clone[i]
            self.control_net.trunk[str(i)]['zero_conv'] = ZeroConvolution(channels)

        self.control_net['conv2'] = nn.ModuleDict()
        self.control_net.conv2['module'] = conv2_clone
        self.control_net.conv2['zero_conv'] = ZeroConvolution(channels)
    
    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())
        
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask
    
    def flow_warped_frames(self, x):
        x = x.detach()
        b, t, c, h, w = x.shape
        nex = torch.cat([x[:, 1:, :, :, :], x[:, -1, :, :, :].unsqueeze(1)], dim = 1)
        x = x.view(-1, c,h,w)
        nex = nex.view(-1, c, h, w)
        t_x, t_nex = self.transforms(x, nex)

        flow = self.flow(t_x, t_nex, num_flow_updates = 12)[-1]
        warped_imgs = self.warp(x.view(-1, c,h,w), flow[-1])
        return warped_imgs.view(b, t, c, h, w).detach()

    def run_function(self, i):
        def custom_forward(x, x_control):
            x_control = self.control_net.trunk[str(i)].module(x_control)
            x = self.model.trunk[i](x) + self.control_net.trunk[str(i)].zero_conv(x_control)
            return x, x_control
        return custom_forward
    
    def forward(self, x: Tensor) -> Tensor:
        # import pdb; pdb.set_trace()
        b,t,c,h,w = x.shape
        warped_frames = self.flow_warped_frames(x)
        
        x = x.view(-1, c, h, w)
        warped_frames = warped_frames.view(-1, c, h, w)
        
        x_control = self.entry_zero_conv(warped_frames)
        
        conv1_control = self.control_net.conv1.module(x + x_control)
        conv1 = self.model.conv1(x) + self.control_net.conv1.zero_conv(conv1_control)

        x_control = conv1_control
        x = conv1
        
        for i in range(len(self.model.trunk)):
            x, x_control = checkpoint.checkpoint(self.run_function(i), x, x_control)
        
        x_control = self.control_net.conv2.module(x_control)
        x = self.model.conv2(x) + self.control_net.conv2.zero_conv(x_control)

        x = torch.add(x, conv1)
        x = self.model.upsampling(x)
        x = self.model.conv3(x)

        x = torch.clamp_(x, 0.0, 1.0)
        x = x.view(b, t, c, h*4, w*4)
        return x

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
          self.forward(x)
    
    def set_train_mode(self, control = True, net = False):
        for param in self.control_net.parameters():
            param.requires_grad = control
        
        for param in self.model.parameters():
            param.requires_grad = net


            

class ControlledSRResNetWithAttention(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            num_rcb: int = 16,
            path_to_weights = str,
    ) -> None:
        super().__init__()
        srresnet = srresnet_x4(in_channels = in_channels,
                                 out_channels = out_channels,
                                 channels = channels,
                                 num_rcb = num_rcb)
        self.channels = channels
        self.model = load_pretrained_state_dict(srresnet, compile_state = False, model_weights_path = path_to_weights)
        
        # Low frequency information extraction layer
        conv1_clone = nn.Sequential(
            nn.Conv2d(in_channels, channels, (9, 9), (1, 1), (4, 4)),
            nn.PReLU(),
        )
        conv1_clone.load_state_dict(self.model.conv1.state_dict())

        # High frequency information extraction block
        trunk_clone = []
        for i in range(num_rcb):
            residual_block_clone = _ResidualConvBlock(channels)
            residual_block_clone.load_state_dict(self.model.trunk[i].state_dict())
            trunk_clone.append(residual_block_clone)
        trunk_clone = nn.Sequential(*trunk_clone)

        # High-frequency information linear fusion layer
        conv2_clone = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )
        conv2_clone.load_state_dict(self.model.conv2.state_dict())


        self.control_net = nn.ModuleDict()
        
        self.control_net['conv1'] = nn.Sequential(ZeroConvolution(in_channels), conv1_clone, ZeroConvolution(channels))

        trunk_control = []
        for i in range(num_rcb):
            trunk_control.append(nn.Sequential(ZeroConvolution(channels), trunk_clone[i], ZeroConvolution(channels)))
        self.control_net['trunk'] = nn.Sequential(*trunk_control)

        self.control_net['conv2'] = nn.Sequential(ZeroConvolution(channels), conv2_clone, ZeroConvolution(channels))
        self.control_net['transformer1'] = ResidualTransformer(num_features=channels, internal_channels=64, patch_size=8)
        self.control_net['transformer2'] = ResidualTransformer(num_features=channels, internal_channels=64, patch_size=8)
        
    def run_function(self, i):
        def custom_forward(x1, x2):
            x = self.model.trunk[i](x1) + self.control_net.trunk[i](x2)
            return x
        return custom_forward
    
    def forward(self, x: Tensor) -> Tensor:
        # import pdb; pdb.set_trace()
        
        b,t,c,h,w = x.shape
        x = x.view(-1, c, h, w)        
        out_conv1 = self.model.conv1(x) + self.control_net.conv1(x)
        
        x1 = out_conv1
        x2 = self.control_net.transformer1(x1.view(b, t, self.channels, h, w)).view(-1, self.channels, h, w)
        for i in range(len(self.model.trunk)):
            x = checkpoint.checkpoint(self.run_function(i), x1, x2)
            x1, x2 = x, x
        
        x2 = self.control_net.transformer2(x2.view(b, t, self.channels, h, w)).view(-1, self.channels, h, w)
        
        x = self.model.conv2(x1) + self.control_net.conv2(x2)

        x = torch.add(x, out_conv1)
        x = self.model.upsampling(x)
        x = self.model.conv3(x)

        x = torch.clamp_(x, 0.0, 1.0)
        x = x.view(b, t, c, h*4, w*4)
        return x

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
          self.forward(x)
    
    def set_train_mode(self, control = True, net = False):
        for param in self.control_net.parameters():
            param.requires_grad = control
        
        for param in self.model.parameters():
            param.requires_grad = net



class ControlledSRResNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            num_rcb: int = 16,
            path_to_weights = str,
    ) -> None:
        super().__init__()
        srresnet = srresnet_x4(in_channels = in_channels,
                                 out_channels = out_channels,
                                 channels = channels,
                                 num_rcb = num_rcb)

        self.model = load_pretrained_state_dict(srresnet, compile_state = False, model_weights_path = path_to_weights)

        # Low frequency information extraction layer
        conv1_clone = nn.Sequential(
            nn.Conv2d(in_channels, channels, (9, 9), (1, 1), (4, 4)),
            nn.PReLU(),
        )
        conv1_clone.load_state_dict(self.model.conv1.state_dict())

        # High frequency information extraction block
        trunk_clone = []
        for i in range(num_rcb):
            residual_block_clone = _ResidualConvBlock(channels)
            residual_block_clone.load_state_dict(self.model.trunk[i].state_dict())
            trunk_clone.append(residual_block_clone)
        trunk_clone = nn.Sequential(*trunk_clone)

        # High-frequency information linear fusion layer
        conv2_clone = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )
        conv2_clone.load_state_dict(self.model.conv2.state_dict())


        self.control_net = nn.ModuleDict()

        self.control_net['conv1'] = nn.Sequential(ZeroConvolution(in_channels), conv1_clone, ZeroConvolution(channels))

        trunk_control = []
        for i in range(num_rcb):
            trunk_control.append(nn.Sequential(ZeroConvolution(channels), trunk_clone[i], ZeroConvolution(channels)))
        self.control_net['trunk'] = nn.Sequential(*trunk_control)

        self.control_net['conv2'] = nn.Sequential(ZeroConvolution(channels), conv2_clone, ZeroConvolution(channels))

    def run_function(self, i):
        def custom_forward(x):
            x = self.model.trunk[i](x) + self.control_net.trunk[i](x)
            return x
        return custom_forward
    
    def forward(self, x: Tensor) -> Tensor:
        # import pdb; pdb.set_trace()
        b,t,c,h,w = x.shape
        x = x.view(-1, c, h, w)
        conv1 = self.model.conv1(x) + self.control_net.conv1(x)

        x = conv1
        for i in range(len(self.model.trunk)):
            x = checkpoint.checkpoint(self.run_function(i), x)
        x = self.model.conv2(x) + self.control_net.conv2(x)

        x = torch.add(x, conv1)
        x = self.model.upsampling(x)
        x = self.model.conv3(x)

        x = torch.clamp_(x, 0.0, 1.0)
        x = x.view(b, t, c, h*4, w*4)
        return x

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
          self.forward(x)
    
    def set_train_mode(self, control = True, net = False):
        for param in self.control_net.parameters():
            param.requires_grad = control
        
        for param in self.model.parameters():
            param.requires_grad = net
        

class AttentionModule(nn.Module):
  def __init__(self, n_channels, patch_size, 
               n_heads=None, dropout = 0.0):
    super().__init__()
    self.n_channels = n_channels
    self.patch_size = patch_size
    if n_heads is None:
      self.n_heads = n_channels
    else:
      self.n_heads = n_heads

    self.dim = patch_size**2 * n_channels
    self.hidden_dim = self.dim // self.n_heads

    self.to_q = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, groups = n_channels)
    self.to_k = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, groups = n_channels)
    self.to_v = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)

    self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)

    for module in self.modules():
      if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
          nn.init.constant_(module.bias, 0)
      elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)

  def forward(self, x):
    b, t, c, h, w = x.shape
    num_patches = (h // self.patch_size) * (w // self.patch_size)

    q = self.to_q(x.view(-1, c, h, w))
    k = self.to_k(x.view(-1, c, h, w))
    v = self.to_v(x.view(-1, c, h, w))

    unfold_q = F.unfold(q, kernel_size = self.patch_size, 
                        stride = self.patch_size, padding = 0)
    
    unfold_k = F.unfold(k, kernel_size = self.patch_size, 
                        stride = self.patch_size, padding = 0)
    unfold_v = F.unfold(v, kernel_size = self.patch_size, 
                        stride = self.patch_size, padding = 0)

    unfold_q = unfold_q.view(b, t, 
                             self.n_heads, self.hidden_dim, 
                             num_patches)
    unfold_k = unfold_k.view(b, t, 
                             self.n_heads, self.hidden_dim, 
                             num_patches)
    unfold_v = unfold_v.view(b, t, 
                             self.n_heads, self.hidden_dim, 
                             num_patches)
    
    unfold_q = unfold_q.permute(0, 2, 3, 1, 4).contiguous()
    unfold_k = unfold_k.permute(0, 2, 3, 1, 4).contiguous()
    unfold_v = unfold_v.permute(0, 2, 3, 1, 4).contiguous()

    unfold_q = unfold_q.view(b, self.n_heads, self.hidden_dim, 
                             t*num_patches).permute(0, 1, 3, 2).contiguous()
    unfold_k = unfold_k.view(b, self.n_heads, self.hidden_dim, 
                             t*num_patches).permute(0, 1, 3, 2).contiguous()
    unfold_v = unfold_v.view(b, self.n_heads, self.hidden_dim, 
                             t*num_patches).permute(0, 1, 3, 2).contiguous()
    
#     attention = torch.matmul(unfold_q.transpose(2, 3), unfold_k)
#     attention = attention / np.sqrt(self.hidden_dim)
#     attention = attention.softmax(dim = -1)

#     attention_x = torch.matmul(attention, unfold_v.transpose(2, 3))
#     attention_x = attention_x.view(b, self.n_heads, t, num_patches, self.hidden_dim)
#     attention_x = attention_x.permute(0, 2, 1, 4, 3).contiguous()
#     attention_x = attention_x.view(b*t, self.dim, num_patches)
    attention_x = F.scaled_dot_product_attention(unfold_q, unfold_k, unfold_v)
    attention_x = attention_x.view(b, self.n_heads, t, num_patches, self.hidden_dim)
    attention_x = attention_x.permute(0, 2, 1, 4, 3).contiguous()
    attention_x = attention_x.view(b*t, self.dim, num_patches)
    features = F.fold(attention_x, output_size=(h, w), 
                      kernel_size=self.patch_size, stride=self.patch_size, 
                      padding = 0)
    out = self.conv(features).view(b,t,c,h,w)
    out += x
    return out

class ResidualPreActNorm(nn.Module):
  def __init__(self, num_features, internal_channels = 64, kernel_size = 3):
    super().__init__()
    self.prelu1 = nn.PReLU()
    self.norm1 = nn.BatchNorm2d(num_features)
    self.conv1 = nn.Conv2d(num_features, internal_channels, kernel_size, padding = kernel_size//2)

    self.prelu2 = nn.PReLU()
    self.norm2 = nn.BatchNorm2d(internal_channels)
    self.conv2 = nn.Conv2d(internal_channels, num_features, kernel_size, padding = kernel_size//2)

    self.module = nn.Sequential(self.prelu1, self.norm1, self.conv1, self.prelu2, self.norm2, self.conv2)
    
    for module in self.modules():
      if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
          nn.init.constant_(module.bias, 0)
      elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)

  def forward(self, x):
    out = checkpoint.checkpoint_sequential(self.module, 3, x)
    return x + out

class ResidualTransformer(nn.Module):
  def __init__(self, num_features, num_frames= 7, internal_channels = 64, patch_size = 8,
               kernel_size = 3):
    super().__init__()
    self.pos_embedding = PositionalEncodingPermute3D(num_frames)
    self.attention = AttentionModule(num_features, patch_size)
    self.resff = ResidualPreActNorm(num_features, internal_channels, kernel_size)

  def forward(self, x):
    b, t, c, h, w = x.shape
    x = x + self.pos_embedding(x)
    x = self.attention(x)
    x = x.view(b*t, c, h, w)
    x = self.resff(x)
    return x.view(b, t, c, h, w)