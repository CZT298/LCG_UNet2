import torch
from torch import nn

from typing import Tuple, Union
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from .swin_unetr import SwinTransformer,PatchMerging,PatchMergingV2
from typing import Optional, Sequence, Tuple, Type, Union
from monai.utils import ensure_tuple_rep, look_up_option, optional_import




class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, (3, 3, 3), padding=1),  # in_ch、out_ch是通道数
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class UnetEncoder(nn.Module):
    def __init__(self,in_channels,feature_size):
        super(UnetEncoder,self).__init__()
        size = feature_size
        self.conv0 = DoubleConv(in_channels, 1 * size)
        self.conv1 = DoubleConv(1 * size, 1 * size)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = DoubleConv(1 * size, 2 * size)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = DoubleConv(2 * size, 4 * size)
        self.pool3 = nn.MaxPool3d(2)



    def forward(self,x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        p1 = self.pool1(x1)
        x2 = self.conv2(p1)
        p2 = self.pool2(x2)
        x3 = self.conv3(p2)
        p3 = self.pool3(x3)

        return x0,p1,p2,p3
       
class LGFF(nn.Module):
    def __init__(self,channel, r=16):
        super(LGFF, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )
        self.conv = nn.Sequential(
            nn.Conv3d(channel, channel//2, (3, 3, 3), padding=1),  # in_ch、out_ch是通道数
            torch.nn.Dropout(0.5),
            nn.BatchNorm3d(channel//2),
            nn.ReLU(inplace=True),
        )


    def forward(self, c_in,st_in):
        x = torch.cat([c_in, st_in], dim=1)
        b, c, _, _,_ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1, 1)
        # Fusion
        y = torch.mul(x, y)
        y = self.conv(y)
        c_out = c_in + y
        st_out = st_in + y
        return c_out, st_out   
    
    
class LGFF0(nn.Module):
    def __init__(self):
        super(LGFF0, self).__init__()

    def forward(self, c_in,st_in):
        c_out = c_in
        st_out = st_in
        return c_out, st_out   

class MERGE(nn.Module):
    def __init__(self):
        super(MERGE, self).__init__()
    
        self.conv = nn.Sequential(
            nn.Conv3d(2, 1, (3, 3, 3), padding=1), 
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, cin,stin):
        avgout1 = torch.mean(cin, dim=1, keepdim=True)
        maxout1, _ = torch.max(cin, dim=1, keepdim=True)
        c = torch.cat([avgout1, maxout1], dim=1)
        c = self.conv(c)
        c_sita = self.sigmoid(c)
        
        avgout2 = torch.mean(stin, dim=1, keepdim=True)
        maxout2, _ = torch.max(stin, dim=1, keepdim=True)
        st = torch.cat([avgout2, maxout2], dim=1)
        st = self.conv(st)
      
        cout = cin * c_sita
        stout = stin * (1-c_sita)
        
        return cout + stout 

class MERGE0(nn.Module):
    def __init__(self):
        super(MERGE0, self).__init__()
       
    def forward(self, cin,stin):
        out = cin + stin
        return out
MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}
class LGC_UNet2(nn.Module):
    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        lgff = 1,
        merge = 1,
    ) -> None:

        super(LGC_UNet2,self).__init__()   
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)   
        window_size = ensure_tuple_rep(7, spatial_dims)
        self.normalize = normalize
        self.UnetEncoder = UnetEncoder(in_channels=in_channels,feature_size=feature_size)
        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
        )

        if lgff == 1:
            self.LGFF_1 = LGFF(feature_size*2)
            self.LGFF_2 = LGFF(feature_size*4)

        elif lgff == 0:
            self.LGFF_1 = LGFF0()
            self.LGFF_2 = LGFF0()

        self.transenc1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.transenc2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.transenc3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.transenc4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.cnndec3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.cnndec2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.cnndec1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.transdec3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.transdec2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.transdec1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        ) 
        
        if merge==1 :
            self.merge = MERGE()
        if merge==0 :
            self.merge = MERGE0()
            
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

    def forward(self, x_in):
        c_enc0,u_out1,u_out2,u_out3 = self.UnetEncoder(x_in) 
        hidden_states_out = self.swinViT(x_in, self.normalize)

        st_enc0 = self.transenc1(x_in)
        st1 = self.transenc2(hidden_states_out[0])

        c_enc1, st_enc1 = self.LGFF_1(u_out1,st1)
        st2= self.transenc3(hidden_states_out[1])
        c_enc2, st_enc2 = self.LGFF_2(u_out2,st2)
        
        st3 = self.transenc4(hidden_states_out[2])

        c_dec3 = self.cnndec3(u_out3,c_enc2)
        c_dec2 = self.cnndec2(c_dec3,c_enc1)
        c_dec1 = self.cnndec1(c_dec2,c_enc0)
        
        st_dec3 = self.transdec3(st3,st_enc2)
        st_dec2 = self.transdec2(st_dec3,st_enc1)
        st_dec1 = self.transdec1(st_dec2,st_enc0)

        out = self.merge(c_dec1,st_dec1)
        logits = self.out(out)
        
        return logits
    
if __name__ == '__main__':
    model = LGC_UNet2(img_size =(128,128,128) ,
                          in_channels=1,
                          out_channels=1,
                          lgff = 1,
                          merge = 1
                          ).to('cuda')
    
    mask = model(torch.rand(1, 1, 128,128,128).to('cuda'))

    
