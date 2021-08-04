import torch.nn as nn
import torch

from models.BaseModel import BaseModel


class DeepBasicBlock(nn.Module):
    def __init__(self, inchannels, outchannels, kernel_size=3, stride=1, padding=0, dilation=1):
        super(DeepBasicBlock, self).__init__()

        self.Conv1D = nn.Conv1d(inchannels, outchannels, kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(outchannels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size

    def _compute_size(self, input_size, dilation):
        return int((input_size + 2 * self.padding - dilation * (self.kernel_size - 1) - 1) / self.stride + 1)

    def output_size(self, input_size):
        return self._compute_size(input_size, dilation=self.dilation)

    def forward(self, x):
        out = self.Conv1D(x)
        out = self.bn(out)
        out = self.relu(out)

        return out
    
class DeepOneDCNN(BaseModel):
    def __init__(self, c, input_dim, input_channels, output_dim):
        super(DeepOneDCNN, self).__init__(c.device)
        self.input_dim = input_dim
        self.avg_pool_bool = c.avg_pool
        self.stddev = c.stddev
        outchannels_step = int(c.CNN_num_channels/4)
        avgPadding = 0
        self.avgPool1 = torch.nn.AvgPool1d(kernel_size=c.CNN_avg_pool_width, stride=c.CNN_avg_pool_width, padding=avgPadding) # 430
        self.block1 = DeepBasicBlock(inchannels=input_channels, outchannels=outchannels_step, kernel_size=c.CNN_kernel_size, stride=1, dilation=c.CNN_dilation)#86x50
        pool_out_size = int((input_dim + 2 * avgPadding - c.CNN_avg_pool_width)/c.CNN_avg_pool_width + 1)
        self.block2 = DeepBasicBlock(inchannels=outchannels_step, outchannels=outchannels_step*2, kernel_size=c.CNN_kernel_size, stride=1, dilation=c.CNN_dilation) # 28x50
        self.block3 = DeepBasicBlock(inchannels=outchannels_step*2, outchannels=outchannels_step*3, kernel_size=c.CNN_kernel_size, stride=1, dilation=c.CNN_dilation) # 9x100
        self.block4 = DeepBasicBlock(inchannels=outchannels_step*3, outchannels=c.CNN_num_channels, kernel_size=c.CNN_kernel_size, stride=1, dilation=c.CNN_dilation) #3x400

        if self.avg_pool_bool:
            block1_out_size = self.block1.output_size(pool_out_size)
        else:
            block1_out_size = self.block1.output_size(input_dim)

        block2_out_size = self.block2.output_size(block1_out_size)
        block3_out_size = self.block3.output_size(block2_out_size)
        block4_out_size = self.block4.output_size(block3_out_size)

        # print(block1_out_size, block2_out_size, block3_out_size, block4_out_size)
        self.fc_layer = nn.Sequential(
            nn.Linear(block4_out_size*c.CNN_num_channels, 800),
            nn.ReLU(inplace=True),
            nn.Dropout(p=c.dropout_percent),
            nn.Linear(800, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(p=c.dropout_percent),
            nn.Linear(200, output_dim)
        )
        

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if self.avg_pool_bool:
            x = self.avgPool1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
      
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = self.fc_layer(x)           
        return x
