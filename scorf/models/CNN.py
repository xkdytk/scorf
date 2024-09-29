import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath
from tqdm import tqdm

class Spatial(nn.Module):
    def __init__(self, dim):
        super(Spatial, self).__init__()
        kernel_size = 7
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, dilation=1, groups=1, bias=False)
        self.pwconv1 = nn.Linear(dim, dim) # pointwise/1x1 convs, implemented with linear layers
        #self.layer_norm = LayerNorm(in_channels, eps=1e-6, data_format="channels_first")
    def forward(self, x):
        #x_compress = torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
        #x = self.layer_norm(x)
        x_compress = torch.max(x,1)[0]
        x_out = self.pwconv1(x_compress).unsqueeze(1)
        #x_out = self.conv(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x

class ConvNeighbor(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        
        # self.conv = nn.Conv2d(1, output_dim, kernel_size, 1, padding)
        self.weight = nn.Parameter(torch.Tensor(output_dim, 1, kernel_size))
        self.param = self.weight.expand(output_dim, input_dim, kernel_size).clone()
        # self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim, kernel_size))
        self.param_bias = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1))
            self.param_bias = self.bias.expand(1, output_dim, 1).clone()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.param)
        if self.param_bias is not None:
            nn.init.constant_(self.param_bias, 0.)
        
    def forward(self, x):
        # print(self.weight[0, 0, :])
        # param = torch.repeat_interleave(self.weight, self.repeat_tensor, dim=1)
        # param = self.weight.repeat(1, self.input_dim, 1)
        output = F.conv1d(x, self.param, stride=self.stride, padding=self.padding)
        # output = F.conv1d(x, self.weight.expand(self.output_dim, self.input_dim, self.kernel_size).clone(), padding=self.padding)
        # output = F.conv1d(x, self.weight.repeat(1, self.input_dim, 1), padding=self.padding)
        if self.param_bias is not None:
            output = output + self.param_bias
        # x = x.unsqueeze(1)
        # print(x.shape).reapeat(input_dim)
        # x = self.conv(x)
        # print(x.shape)
        # x = torch.sum(x, dim=-2)
        return output
    
class ConvTransposeNeighbor(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, padding, bias=True):
        super().__init__()
        self.padding = padding
        
        # self.conv = nn.Conv2d(1, output_dim, kernel_size, 1, padding)
        self.weight = nn.Parameter(torch.Tensor(output_dim, 1, kernel_size), requires_grad=True)
        self.param = self.weight.expand(output_dim, input_dim, kernel_size).clone()
        # self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim, kernel_size))
        self.param_bias = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1), requires_grad=True)
            self.param_bias = self.bias.expand(1, output_dim, 1).clone()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.param)
        if self.param_bias is not None:
            nn.init.constant_(self.param_bias, 0.)
        
    def forward(self, x):
        # print(self.weight[0, 0, :])
        # param = torch.repeat_interleave(self.weight, self.repeat_tensor, dim=1)
        # param = self.weight.repeat(1, self.input_dim, 1)
        output = F.conv_transpose1d(x, self.param, padding=self.padding)
        # output = F.conv1d(x, self.weight.expand(self.output_dim, self.input_dim, self.kernel_size).clone(), padding=self.padding)
        # output = F.conv1d(x, self.weight.repeat(1, self.input_dim, 1), padding=self.padding)
        if self.param_bias is not None:
            output = output + self.param_bias
        # x = x.unsqueeze(1)
        # print(x.shape).reapeat(input_dim)
        # x = self.conv(x)
        # print(x.shape)
        # x = torch.sum(x, dim=-2)
        return output

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        # self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim*2) # pointwise/1x1 convs, implemented with linear layers
        # self.pwconv1 = nn.Conv1d(dim, dim*4, kernel_size=1)
        self.act = nn.ReLU()
        self.pwconv2 = nn.Linear(dim*2, dim)
        # self.pwconv2 = nn.Conv1d(dim*4, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        inputs = x
        # x = x.transpose(1,2)
        # x = self.norm(x)
        # x = x.transpose(1,2)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # x = x.transpose(1,2)

        x = inputs + self.drop_path(x)
        x = F.relu(x)
        return x

class CNN(nn.Module):
    def __init__(self, 
                 in_channels=1,
                 block_num=6,
                 dims=256, 
                 drop_path_rate=0.2,
                 input_ch=21,
                 input_ch_views=9,
                 sample=64,
                 layer_scale_init_value=1e-6, 
                 head_init_scale=1.,
        ):
        
        super().__init__()

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views

        self.block_num = block_num
        
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, block_num)] 
        cur = 0
        for i in range(block_num):
            stage = nn.Sequential(
                *[Block(dim=dims, drop_path=dp_rates[cur], 
                layer_scale_init_value=layer_scale_init_value)]
            )
            self.stages.append(stage)
            cur += 1
        
        self.channel_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.channel_proj = nn.ModuleList()
        self.downsample = nn.ModuleList()
        self.proj = nn.ModuleList()

        channel_layer = nn.Sequential(
            nn.Conv1d(input_ch, dims, kernel_size=3, padding=1),
            # nn.ReLU(),
            LayerNorm(dims, eps=1e-6, data_format="channels_first")
        )
        channel_proj = nn.Sequential(
            # nn.LayerNorm(dims, eps=1e-6),
            nn.Conv1d(sample, sample, kernel_size=5, padding=2, groups=sample),#, bias=False),
            # nn.ReLU(),
        )
        downsample = nn.Sequential(
            nn.Conv1d(dims, dims, kernel_size=2, stride=2, bias=False),
            nn.ReLU()
        )
        proj = nn.Sequential(
            nn.Linear(dims, dims),
            # nn.ReLU()
        )
        self.channel_layers.append(channel_layer)
        self.downsample.append(downsample)
        self.proj.append(proj)
        self.channel_proj.append(channel_proj)
        for i in range(block_num-1):
            channel_layer = nn.Sequential(
                nn.Conv1d(dims, dims, kernel_size=3, padding=1),
                # nn.ReLU(),
                LayerNorm(dims, eps=1e-6, data_format="channels_first")
            )
            channel_proj = nn.Sequential(
                # nn.LayerNorm(dims, eps=1e-6),
                nn.Conv1d(sample, sample, kernel_size=5, padding=2, groups=sample),#, bias=False),
                # nn.ReLU(),
            )
            downsample = nn.Sequential(
                nn.Conv1d(dims, dims, kernel_size=2, stride=2, bias=False),
                nn.ReLU()
            )
            proj = nn.Sequential(
                nn.Linear(dims, dims),
                # nn.ReLU()
            )
            self.channel_layers.append(channel_layer)
            self.downsample.append(downsample)
            self.proj.append(proj)
            self.channel_proj.append(channel_proj)
        
        self.alpha_linear = nn.Linear(dims, 1)
        self.norm2 = nn.LayerNorm(sample)
        self.norm = nn.LayerNorm([sample, dims])
        # self.coord_linear = nn.Sequential(
        #     # nn.Linear(dims, dims),
        #     nn.Conv1d(sample, sample, kernel_size=3, padding=1, groups=sample),
        #     # nn.ReLU()
        #     # nn.LayerNorm([sample, dims])
        # )
        self.coord_linear = nn.Linear(dims, dims)
        self.pts_linear = nn.Sequential(
            nn.Linear(dims+input_ch, (dims+input_ch)//2),
            nn.ReLU()
        )
        self.z_linear = nn.Linear(dims+input_ch_views, dims//2)
        self.z_linear1 = nn.Linear(dims//2, 1)
        self.z_cnn = nn.Conv1d(sample, sample, kernel_size=dims//2, groups=sample)
        self.trans_cnn = nn.Sequential(
            nn.ConvTranspose1d(dims, input_ch, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.scale = nn.Parameter(torch.randn(sample))
        self.scale1 = nn.Parameter(-torch.ones(1))
        self.feature_cnn = nn.Conv1d(dims, dims, kernel_size=3, padding=1, bias=False)

        self.feature_linear = nn.Linear(dims, dims)
        self.view_linear = nn.Linear(dims+input_ch_views, dims//2)
        self.rgb_linear = nn.Linear(dims//2, 3)
    
    #     self.apply(self._init_weights)
    #     # self.z_linear.bias.data.add_(0.5)

    # def _init_weights(self, m):
    #     if isinstance(m, (nn.Conv1d)):
    #         # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    #         trunc_normal_(m.weight, std=.02)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)    
    #     elif isinstance(m, nn.LayerNorm):
    #         m.bias.data.zero_()
    #         m.weight.data.fill_(1.0)
    
    def forward_features(self, x, input_views):
        x = x.transpose(1, 2)
        for i in range(self.block_num):
            # x = x.transpose(1, 2)
            residual = x
            x = self.channel_layers[i](x)
            # x = x.transpose(1, 2)
            # x = x + point
            x = x.transpose(1, 2)
            x = self.proj[i](x)
            x = x.transpose(1, 2)
           
            if i != 0:
                x = x + residual
            x = F.relu(x)
            
            # x = self.downsample[i](x)
            
        x = x.transpose(1, 2)
        coord = x

        coord = self.coord_linear(coord)
        coord = torch.cat([coord, input_views], -1)
        
        z_val = self.z_linear(coord)
        z_val = F.relu(z_val)
        z_val = self.z_linear1(z_val).squeeze(-1)
        # z_val = self.norm2(z_val)
        z_val = F.sigmoid(z_val)
        
        for i in range(self.block_num):
            x = self.stages[i](x)
            
        # x = self.trans_cnn(x)
        return x, z_val
    
    def replace_with_max_previous(self, x):
        max_previous = x[:, :-1].cummax(dim=1).values
        mask = x[:, 1:] < max_previous
        output = x.clone()
        output = torch.where(mask, max_previous, x[:, 1:])
        # output[:, 1:, :][mask] = max_previous[mask]
        return torch.cat((x[:, :1], output), dim=1)

    def replace_with_previous(self, x):
        prev_value = x[:, :-1].clamp(max=x[:, 1:].cummax(dim=1).values)
        mask = x[:, 1:] < prev_value
        output = x.clone()
        output[:, 1:][mask] = prev_value[mask]
        return output

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        # input_pts = input_pts.transpose(1, 2)

        pts, z_val = self.forward_features(input_pts, input_views)
        # pts = self.spatial(pts)
        # pts = pts.transpose(1,2)
        # pts = self.norm(pts)

        alpha = self.alpha_linear(pts)
        # coord = pts

        # coord = self.coord_linear(coord)
        # coord = torch.cat([coord, input_views], -1)
        
        # z_val = self.z_linear(coord).squeeze(-1)
        # z_val = F.sigmoid(z_val)
        # z_val = z_val + z_vals.squeeze(-1)
        # z_val = self.replace_with_max_previous(z_val)
        
        # z_val = torch.cumsum(z_val, -1)
        
        # z_val = z_val.clamp(0, 1)
        # z_val = self.norm2(z_val)
        # z_val = F.sigmoid(z_val)

        
        # z_val = torch.cumsum(z_val, -1)

        # z_val = z_val + self.scale1
        # # z_val = self.norm2(z_val)
        # z_val = F.sigmoid(z_val)

        # z_val = self.z_cnn(z_val.unsqueeze(-1)).squeeze(-1)
        # z_val = F.relu(z_val)

        # z_val = self.norm(z_val)
        # z_val = z_val - torch.min(z_val, dim=-1, keepdim=True)[0]
        
        # z_val = torch.cumsum(z_val, -1)
        # z_val = self.norm(z_val)
        # z_val = F.sigmoid(z_val)
        
        # alpha, feature = alpha[..., 0].unsqueeze(-1), alpha[..., 1:]
        
        feature = pts
        feature = self.feature_linear(pts)
        # feature = feature.transpose(1, 2)
        # feature = self.feature_cnn(feature)
        # feature = feature.transpose(1, 2)
        rgb = torch.cat([feature, input_views[:, :feature.shape[-2], :]], -1)#.transpose(1,2)
        
        # rgb = rgb.transpose(1, 2)
        # rgb = self.rgb_cnn(rgb)#.transpose(1,2)
        # rgb = rgb.transpose(1, 2)
        rgb = self.view_linear(rgb)
        rgb = F.relu(rgb)
        rgb = self.rgb_linear(rgb)#.transpose(1,2)

        outputs = torch.cat([rgb, alpha], -1)
        return outputs, z_val.unsqueeze(-1)
