import torch.nn as nn

from models.layer_norm import LayerNorm1D
from models.BaseModel import BaseModel


class BasicBlock(nn.Module):

    def __init__(self, inchannels, outchannels, stride=1, padding = 0, dilation = 1, kernel_size = 3, norm='batch_norm'):
        super(BasicBlock, self).__init__()
        self.kernel_size = kernel_size
        outchannels_1 = (int)(outchannels)
        self.Conv1D_1 = nn.Conv1d(inchannels, outchannels_1, kernel_size = self.kernel_size, stride=stride,
                                padding = padding, dilation = 1)
        if norm == 'layer_norm':
            self.norm1 = LayerNorm1D(outchannels_1)
        else:
            self.norm1 = nn.BatchNorm1d(outchannels_1)

        self.relu1 = nn.ReLU(inplace=True)
        self.Conv1D_more = nn.Conv1d(outchannels_1, outchannels, kernel_size = self.kernel_size, stride=stride,
                                padding = padding, dilation = dilation)

        if norm == 'layer_norm':
            self.norm2 = LayerNorm1D(outchannels)
        else:
            self.norm2 = nn.BatchNorm1d(outchannels)

        self.relu2 = nn.ReLU(inplace=True)
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        out = self.Conv1D_1(x)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.Conv1D_more(out)
        out = self.norm2(out)
        out = self.relu2(out)

        return out

    def _compute_size(self, input_size, dilation):
        return int((input_size + 2 * self.padding - dilation * (self.kernel_size - 1) - 1) / self.stride + 1)

    def output_size(self, input_size):
        layer1_size = self._compute_size(input_size, dilation = 1)
        return self._compute_size(layer1_size, dilation = self.dilation)

class OneDCNN(BaseModel):
    def __init__(self, c, input_dim, input_channels, output_dim):
        super(OneDCNN, self).__init__(c.device)
        norm = 'batch_norm'
        self.input_dim = input_dim
        self.avgPool = nn.AvgPool1d(c.CNN_avg_pool_width, c.CNN_avg_pool_width, padding=2)
        pool_out_size = int((input_dim + 2 * 2 - c.CNN_avg_pool_width)/c.CNN_avg_pool_width + 1)
        self.bnAvgPool = nn.BatchNorm1d(1)
        self.block1 = BasicBlock(inchannels=1, outchannels=c.CNN_num_channels, stride=1, dilation=c.CNN_dilation, kernel_size=c.CNN_kernel_size,norm=norm) # 431 -> 216 -> 108

        if c.avg_pool:
            block1_out_size = self.block1.output_size(pool_out_size)
            #self.fc1 = nn.Linear(108 * 10, 200)
        else:
            block1_out_size = self.block1.output_size(input_dim)
            # self.fc1 = nn.Linear(538 * 10, 200)

        hidden_size = 1000
        self.fc1 = nn.Linear(block1_out_size * c.CNN_num_channels, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        #self.fc2 = nn.Linear(5000, 1000)

        self.dropout = nn.Dropout(p=c.dropout_percent)
        self.fc3 = nn.Linear(hidden_size, output_dim)
        self.avg_pool_bool = c.avg_pool


    def forward(self, x):
        x = x.view(-1, 1, self.input_dim)
        if self.avg_pool_bool:
            x = self.avgPool(x)
            x = self.bnAvgPool(x)

        x = self.block1(x)
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        #x = self.fc2(x)
        #x = self.relu(x)
        #x = self.dropout(x)
        x = self.fc3(x)
        return x
