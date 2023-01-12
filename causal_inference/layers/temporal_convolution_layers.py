import torch
from torch import nn 
from torch.nn import functional as F

class ResidualAdd(nn.Module):
    r"""Residual connection

    # Arguments
    ____________
    fn : sub-class of nn.Module          
    
    # Returns
    _________
    returns residual connection           
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

# projection layer
class ProjectionConv1x1Layer(nn.Module): 
    r"""Projection layer using conv1x1
    """
    def __init__(self, in_channels, out_channels, groups, **kwargs): 
        super().__init__()     
        self.projection = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, groups= groups, **kwargs)
        self.in_channels, self.out_channels = in_channels, out_channels 

    def forward(self, pair): 
        r"""Feed forward

        pair : torch-Tensor
            generated by the function: make_input_n_mask_pairs(x)
            the shape of the tensor 'pair' is b, 2*c, t, n 
        """
        return self.projection(pair)

class CausalDilatedVerticalConv1d(nn.Module): 
    r"""Causal dilated convoltion
    Causal dilated convolution is based on the work by
    \"""
    Oord, A. V. D., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., ... & Kavukcuoglu, K. (2016). 
    Wavenet: A generative model for raw audio. 
    arXiv preprint arXiv:1609.03499.
    \"""
    This module only differs from the original work inthat 
    (1) it is a group-wise convolution 
    (2) the kernel 'moves' vertically.
    """
    def __init__(self, 
                in_channels, out_channels, 
                kernel_size,  
                groups, dilation, 
                **kwargs
                ): 
        assert kernel_size[1] == 1, "kernel[1] should have size 1."
        super().__init__()
        self.pad = (kernel_size[0] - 1) * dilation 
        self.causal_conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                        padding= (self.pad, 0), dilation= dilation, groups= groups, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size 
        self.groups = groups 
        self.dilation = dilation

    def forward(self, x): 
        x = self.causal_conv(x) 
        x = x[..., :-self.causal_conv.padding[0], :] if self.pad > 0 else x
        return x

class MultiVariateCausalDilatedLayer(nn.Module): 
    r"""Multivariate- Dilated Inception Layer 
    we propese a new convolution layer named "Multivariate Causal Dilated layer" 
    """
    def __init__(self, in_channels:int, out_channels:int, 
                kernel_size:tuple, 
                groups:int, dilation:int, 
                num_time_series:int,
                **kwargs): 
        super().__init__() 

        # layer
        assert in_channels == out_channels, 'convolution for different input/output channels are not implemented yet'
        self.causal_conv = CausalDilatedVerticalConv1d(num_time_series*in_channels, num_time_series*out_channels, kernel_size, groups, dilation, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.kernel_size = kernel_size 
        self.groups = groups 
        self.dilation = dilation 
        self.num_time_series = num_time_series

    def forward(self, x): 
        bs, c, t, d = x.shape
        x = self.ravel(x) # raveled (bs, c*d, t, 1)
        x = self.causal_conv(x) # raveled
        x = self.unravel(x, c, d)
        return x

    def ravel(self, input):
        bs, c, t, d = input.shape 
        return torch.reshape(input.permute((0, 3, 1, 2)), (bs, c*d, t, 1))

    def unravel(self, input, num_channels, num_time_series): 
        bs, cd, t, _ = input.shape 
        return torch.reshape(input.squeeze().transpose(-1,-2), (bs, num_channels, -1, num_time_series))

class MultiVariateDecodeLayer(nn.Module): 
    r"""MultiVariateDecodeLayer
    """
    def __init__(self, in_channels:int, out_channels:int, 
                kernel_size:tuple, 
                groups:int, dilation:int, 
                num_time_series:int,
                **kwargs): 
        super().__init__() 

        # layer
        assert in_channels == out_channels, 'convolution for different input/output channels are not implemented yet'
        self.causal_conv = CausalDilatedVerticalConv1d(num_time_series*in_channels, num_time_series*out_channels, kernel_size, groups, dilation, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.kernel_size = kernel_size 
        self.groups = groups 
        self.dilation = dilation 
        self.num_time_series = num_time_series

    def forward(self, x): 
        bs, c, t, d = x.shape
        x = self.ravel(x) # raveled
        x = self.causal_conv(x) # raveled
        x = self.unravel(x, c, d)
        return x

    def ravel(self, input):
        bs, c, t, d = input.shape 
        return torch.reshape(input.permute((0, 3, 1, 2)), (bs, c*d, t, 1))

    def unravel(self, input, num_channels, num_time_series): 
        bs, cd, t, _ = input.shape 
        return torch.reshape(input.squeeze().transpose(-1,-2), (bs, num_channels, -1, num_time_series))

# temporal convolution layer
class DilatedInceptionLayer(nn.Module):
    r"""Dilated inception layer
    """
    def __init__(self, in_channels, out_channels, num_time_series, **kwargs):
        super().__init__()
        self.branch1x1 = MultiVariateCausalDilatedLayer(in_channels, out_channels, (1,1), in_channels, 1, num_time_series, **kwargs)
        self.branch3x1 = MultiVariateCausalDilatedLayer(in_channels, out_channels, (3,1), in_channels, 1, num_time_series, **kwargs)
        self.branch5x1 = MultiVariateCausalDilatedLayer(in_channels, out_channels, (3,1), in_channels, 2, num_time_series, **kwargs)
        self.branch7x1 = MultiVariateCausalDilatedLayer(in_channels, out_channels, (3,1), in_channels, 3, num_time_series, **kwargs)

        self.in_channels, self.out_channels = in_channels, out_channels

    def forward(self, x): 
        b, c, n, p = x.shape
        outs = torch.zeros(b, 4*c, n, p).to(x.device)
        for i in range(4): 
            branch = getattr(self, f'branch{2*i+1}x1')
            outs[:, i::4, ...] = branch(x) 
            # we have c groups of receptive channels...
            # = 4 channels form one group.
        return outs
    
class TemporalConvolutionModule(nn.Module): 
    r"""TemporalConvolutionModule
    """
    def __init__(self, in_channels, out_channels, num_heteros, num_time_series, **kwargs): 
        super().__init__()
        self.dil_filter = DilatedInceptionLayer(in_channels, out_channels, num_time_series, **kwargs)
        self.dil_gate = DilatedInceptionLayer(in_channels, out_channels, num_time_series, **kwargs) 
        self.conv_inter = nn.Conv2d(4*in_channels, in_channels, 1, groups= num_heteros, **kwargs)
        self.in_channels, self.out_channels = in_channels, out_channels  
        self.num_heteros = num_heteros

    def forward(self, x): 
        out_filter = torch.tanh(self.dil_filter(x))
        out_gate = torch.sigmoid(self.dil_gate(x)) 
        out = out_filter*out_gate
        return self.conv_inter(out)        