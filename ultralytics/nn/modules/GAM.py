import torch.nn as nn
import torch
 
class GAM_Attention(nn.Module):
    def __init__(self, in_channels, rate=4, *args, **kwargs):
        super(GAM_Attention, self).__init__()
        
        # 参数验证
        if not isinstance(in_channels, (int, float)) or in_channels <= 0:
            raise ValueError(f"in_channels must be a positive number, got {in_channels}")
        if not isinstance(rate, (int, float)) or rate <= 0:
            raise ValueError(f"rate must be a positive number, got {rate}")
            
        # 确保 in_channels 是整数
        in_channels = int(in_channels)
        rate = int(rate)
        
        # 确保中间层通道数至少为1
        mid_channels = max(1, in_channels // rate)
 
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels)
        )
 
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels)
        )
 
    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).sigmoid()
 
        x = x * x_channel_att
 
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att
 
        return out
 
if __name__ == '__main__':
    x = torch.randn(1, 64, 20, 20)
    b, c, h, w = x.shape
    net = GAM_Attention(in_channels=c)
    y = net(x)
    print(y.size())