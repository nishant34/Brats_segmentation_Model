import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

class _residual_block(nn.Sequential):
    def __init__(self,num_input_features,num_output_features,is_first=False):
        super(_residual_block,self).__init__()
        if is_first:
            self.conv1 = nn.Conv3d(num_input_features,num_output_features,kernel_size = 3,stride=1,padding=1,bias=False)
            self.downsampling = 1  
        else :
            self.conv1 = nn.Conv3d(num_input_features,num_output_features,kernel_size = 3,stride=2,padding=1,bias=False)
            self.downsampling = 2

        self.bn = nn.BatchNorm3d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(num_output_features,num_output_features,kernel_size=3,stride=1,padding=1)
        self.shortcut = nn.Sequential(nn.Conv3d(num_input_features, num_output_features, kernel_size=1,stride=self.downsampling, bias=False),nn.BatchNorm3d(num_output_features))
            
                      
            
          
    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        residual = self.shortcut(input)
        x+=residual

        return x

class squeeze_excite_block(nn.Sequential):
    def __init__(self,num_input_features,ratio=16):
        super(squeeze_excite_block,self).__init__()
        self.linear1 = nn.Linear(num_input_features, num_input_features // ratio, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(num_input_features // ratio, num_input_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):
        y = F.avg_pool3d(x, kernel_size=x.size()[2:5])
        y = y.permute(0, 2, 3, 4, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 4, 1, 2, 3)
        y = x * y
        return y


class FCN1(nn.Module):
    def __init__(self,num_input_features=32,drop_rate=0,num_classes=4):
        super(FCN1,self).__init__()
        self.initial_conv_block = nn.Conv3d(4,24,kernel_size=3,stride=1,padding=1,bias=False)
        self.drop_rate=drop_rate
        self.pool1 = _residual_block( num_input_features=24, num_output_features=32,is_first=False)
    
        self.conv2 = _residual_block(num_input_features=32, num_output_features=64,is_first=True) 
        self.pool2 = _residual_block(num_input_features=64, num_output_features=64,is_first=False)
    
        self.conv3 = _residual_block(num_input_features=64, num_output_features=128,is_first=True) 
        self.pool3 = _residual_block(num_input_features=128, num_output_features=128,is_first=False)

        self.conv4 = _residual_block(num_input_features=64, num_output_features=128,is_first=True) 

        self.up_5 = nn.ConvTranspose3d(256, 256, kernel_size=4, stride=2 , padding=1, groups=num_classes, bias=False)
        self.conv5 = _residual_block(num_input_features=384, num_output_features=256,is_first=True)

        self.up_6 = nn.ConvTranspose3d(128, 128, kernel_size=4, stride=2 , padding=1, groups=num_classes, bias=False)
        self.conv6 = _residual_block(num_input_features=192, num_output_features=128,is_first=True)  

        self.up_7 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2 , padding=1, groups=num_classes, bias=False)
        self.conv7 = _residual_block(num_input_features=88, num_output_features=64,is_first=True)                               

        self.conv1r = nn.Conv3d(4, 64, kernel_size=7, stride=1, padding=3)
        self.bn1r = nn.BatchNorm3d(64)
        self.relur = nn.ReLU(inplace=True)

        self.block1 = _residual_block(num_input_features=64, num_output_features=64,is_first=True)
        self.block10 = _residual_block(num_input_features=64, num_output_features=64,is_first=False)

        self.se1 = squeeze_excite_block(num_input_features=64)
        self.gate1 = nn.Sigmoid()

        self.block2 = _residual_block(num_input_features=64, num_output_features=128,is_first=True)
        self.block20 = _residual_block(num_input_features=128, num_output_features=128,is_first=False)

        self.se2 = squeeze_excite_block(num_input_features=128)
        self.gate2 = nn.Sigmoid()

        self.block3 = _residual_block(num_input_features=128, num_output_features=256,is_first=True)
        self.se3 = squeeze_excite_block(num_input_features=256)
        self.gate3 = nn.Sigmoid()

        self.block4 = _residual_block(num_input_features=128, num_output_features=128,is_first=True)
        self.block40 = _residual_block(num_input_features=128, num_output_features=128,is_first=False)

        self.se4 = squeeze_excite_block(num_input_features=128)
        self.gate4 = nn.Sigmoid()

        self.up2_5 = nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2 , padding=1, groups=num_classes, bias=False)
        self.conv2_5 = _residual_block(num_input_features=512, num_output_features=256,is_first=True)

        self.up2_6 = nn.ConvTranspose3d(128, 128, kernel_size=4, stride=2 , padding=1, groups=num_classes, bias=False)
        self.conv2_6 = _residual_block(num_input_features=256, num_output_features=128,is_first=True)

        self.up2_7 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2 , padding=1, groups=num_classes, bias=False)
        self.conv2_7 = _residual_block(num_input_features=128, num_output_features=64,is_first=True)
           
        self.up2_8 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2 , padding=1, groups=num_classes, bias=False)
        self.conv21_8 = _residual_block(num_input_features=96, num_output_features=16,is_first=True)
        self.conv22_8 = _residual_block(num_input_features=16, num_output_features=8,is_first=True)
        self.conv23_8 = _residual_block(num_input_features=8, num_output_features=4,is_first=True)


        self.bn_class = nn.BatchNorm3d(8)
        self.conv_class = nn.Conv3d(4 , 4, kernel_size=1, padding=0)

        for m in self.modules():
         if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal(m.weight)
                
         elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant(m.weight, 1)
            nn.init.constant(m.bias, 0)


    def forward(self,x):
        fconv1 = self.initial_conv_block(x)
        out1 = self.pool1(fconv1)

        out1 = self.conv2(out1)
        fconv2 = out1
        out1 = self.pool2(fconv2)         
           
        out1 = self.conv3(out1)
        fconv3 = out1
        out1 = self.pool3(fconv3)

        out1 = self.conv4(out1)
        fconv4 = out1
        out1 = F.dropout(out1, p=self.drop_rate, training=self.training)

        out1 = self.up_5(out1)
        out1 = torch.cat([fconv3,out1],1)
        out1 = self.conv5(out1)

        out1 = self.up_6(out1)
        out1 = torch.cat([fconv2,out1],1)
        out1 = self.conv6(out1)
        fconv6 = out1

        out1 = self.up_7(out1)
        out1 = torch.cat([fconv1,out1],1)
        out1 = self.conv7(out1)
        fconv7 = out1
        print(fconv7.size())

        out1 = self.conv1r(x)
        out1 = self.bn1r(out1)
        out1 = self.relur(out1)
        fconv1r = out1

        out1 = self.block1(out1)
        fse1 = self.se1(out1)
        out1 = self.gate1(fconv7)
        #print(fse1.size())
        #print(out1.size())
        out2 = fse1*out1
        #print(out1.size())
        out1=self.se1(out2)
        #print(out1.size())
        out1 = self.block10(out1)
        #print(out1.size())
        fblock1b = out1


        fblock1 = self.block1(fconv7)
        fse1 = self.se1(fblock1)
        fgate1 = self.gate1(fse1)
        fblock1concat = torch,mul(fse1*fgate1)
        fblock1se=self.se1(fblock1concat)
        fblock1b = self.block1(fblock1se)
            
            
        out1 = self.block2(out1)
        #print(out1.size())
        fse2 = self.se2(out1)
        #print(fse2.size())
        out1 = self.gate2(fconv6)
        #print(out1.size())
        #print(fse2.size())
        out1 = torch.mul(fse2,out1)
        out1=self.se2(out1)
        out1 = self.block20(out1)
        #print(out1.size())
        fblock2b = out1


            
        out1 = self.block3(fblock2b)
        fse3 = self.se3(out1)
        out1 = self.gate3(out1)
        out1 = torch.mul(fse3,out1)
        out1=self.se3(out1)
        out1 = self.block3(out1)
        fblock3b = out1
           
        out1 = self.block3(fblock3b)
        out1=self.se3(out1)
        out1 = self.block3(out1)

        out1 = self.up2_5(out1)
        out1 = torch.concat([fblock3b,out1],1)
        out1 = self.conv2_5(out1)
        out1 = self.block4(out1)
        #print(out1.size())
        out1 = self.se4(out1)
        ##print(out1.size())
        out1 = self.block40(out1)
        #print(out1.size())

        out1 = self.up2_6(out1)
        #print(out1.size())
        out1 = torch.cat([fblock2b,out1],1)
        #print(out1.size())
        out1 = self.conv2_6(out1)
        #print(out1.size())

        out1 = self.up2_7(out1)
        #print(out1.size())
        out1 = torch.cat([fblock1b,out1],1)
        #print(out1.size())
        out1 = self.conv2_7(out1)
        #print(out1.size())

        out1 = self.up2_8(out1)
        #print(out1.size())
        out1 = torch.cat([fconv1r,out1],1)
        #print(out1.size())
        out1 = self.conv21_8(out1)
        #print(out1.size())
        out1 = self.conv22_8(out1)
        #print(out1.size())
        out1 = self.conv23_8(out1)
        #print(out1.size())

        #out1 = self.bn_class(out1)
        #print(out1.size())
        out1 = self.conv_class(out1)
        #print(out1.size())
        return out1
