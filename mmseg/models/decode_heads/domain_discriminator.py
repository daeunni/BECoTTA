import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from mmcv.cnn import ConvModule
import pdb 
from ..builder import HEADS
from .decode_head import BaseDecodeHead

    
@HEADS.register_module()
class DomainDiscriminator(BaseDecodeHead):
    def __init__(self,
                 n_outputs = 4,       
                 **kwargs):
        
        super(DomainDiscriminator, self).__init__(**kwargs)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 32 * 32, 250)
        self.fc2 = nn.Linear(250, 4)
        self.dropout = nn.Dropout(0.5)

    def transform(self, x) : 
        transform = transforms.Compose([
        transforms.Resize((256, 256)),    
        ])
        x = transform(x)
        return x
                
    def forward(self, inputs):
        x = self.transform(inputs)  

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 32 * 32 * 32)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
