import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########Pytorch########

class unet_assemblynet(nn.Module):
    #this is the implementation of assemblynet architecture from tenserflow/keras model.
    def __init__(self,nf=24,nc=2,dropout_rate=0.5,in_mod=1):
        super().__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv3d(in_mod, nf, 3, padding= 1)
        self.conv1_bn = nn.BatchNorm3d(nf)
        self.conv2 = nn.Conv3d(nf, 2*nf, 3, padding= 1)
        self.conv2_bn = nn.BatchNorm3d(2*nf)
        self.conv3 = nn.Conv3d(2*nf, 2*nf, 3, padding= 1)
        self.conv3_bn = nn.BatchNorm3d(2*nf)
        self.conv4 = nn.Conv3d(2*nf, 4*nf, 3, padding= 1)
        self.conv4_bn = nn.BatchNorm3d(4*nf)
        self.conv5 = nn.Conv3d(4*nf, 4*nf, 3, padding= 1)
        self.conv5_bn = nn.BatchNorm3d(4*nf)
        self.conv6 = nn.Conv3d(4*nf, 8*nf, 3, padding= 1)
        self.conv6_bn = nn.BatchNorm3d(8*nf)
        self.conv7 = nn.Conv3d(8*nf, 16*nf, 3, padding= 1)
        #bottleneck
        self.concat1_bn = nn.BatchNorm3d(8*nf+16*nf)
        #print('hak swalhak')
        self.conv8 = nn.Conv3d((8*nf+16*nf), 8*nf, 3, padding= 1)

        self.concat2_bn = nn.BatchNorm3d(8*nf+4*nf)
        self.conv9 = nn.Conv3d(8*nf+4*nf, 4*nf, 3, padding= 1)

        self.concat3_bn = nn.BatchNorm3d(4*nf+2*nf)
        self.conv10 = nn.Conv3d(4*nf+2*nf, 4*nf, 3, padding= 1)

        self.conv_out = nn.Conv3d(4*nf, nc, 3, padding= 1)
        self.up= nn.Upsample(scale_factor=2,mode='trilinear', align_corners=False)
        #self.up= nn.Upsample(scale_factor=2,mode='nearest')
        self.pool= nn.MaxPool3d(2)
        self.dropout= nn.Dropout(dropout_rate)
        self.softmax=nn.Softmax(dim=1)

    def encoder(self,in_x):
        self.x1=self.conv1_bn(F.relu(self.conv1(in_x)))
        self.x1= F.relu(self.conv2(self.x1))
        self.x2= self.conv2_bn(self.dropout(self.pool(self.x1)))
        self.x2=self.conv3_bn(F.relu(self.conv3(self.x2)))
        self.x2= F.relu(self.conv4(self.x2))
        self.x3= self.conv4_bn(self.dropout(self.pool(self.x2)))
        self.x3=self.conv5_bn(F.relu(self.conv5(self.x3)))
        self.x3= F.relu(self.conv6(self.x3))
        self.x4= self.conv6_bn(self.dropout(self.pool(self.x3)))
        self.x4=F.relu(self.conv7(self.x4))#bottleneck
        return self.x4,self.x3,self.x2,self.x1

    def decoder(self,x4,x3,x2,x1):
        self.x5=self.up(x4)
        self.x5=self.concat1_bn(  torch.cat((self.x5,x3), dim=1)  )
        #self.x5=self.cat1_bn(  torch.cat((self.x5,x3), dim=1)  )
        self.x5= F.relu(self.conv8(self.x5))
        self.x6=self.up(self.x5)
        self.x6=self.concat2_bn(  torch.cat((self.x6,x2), dim=1)  )
        #self.x6=self.cat2_bn(  torch.cat((self.x6,x2), dim=1)  )
        self.x6= F.relu(self.conv9(self.x6))
        self.x7=self.up(self.x6)
        self.x7=self.concat3_bn(  torch.cat((self.x7,x1), dim=1)  )
        #self.x7=self.cat3_bn(  torch.cat((self.x7,x1), dim=1)  )
        self.x7= F.relu(self.conv10(self.x7))
        return self.softmax(self.conv_out(self.x7))

    def forward(self, x):
        x4,x3,x2,x1 = self.encoder(x)
        decoder_out=self.decoder(x4,x3,x2,x1)
        return decoder_out


#######################