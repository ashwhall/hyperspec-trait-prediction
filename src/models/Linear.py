import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import BaseModel


class Linear(BaseModel):
    def __init__(self, c, input_dim, input_channels, output_dim):
        super(Linear, self).__init__(c.device)
        self.avgPool = nn.AvgPool1d(5, 5, 2)
        if c.avg_pool == True:
            self.fc1 = nn.Linear(401, 200)
        else:
            self.fc1 = nn.Linear(input_dim,200)
        self.fc2 = nn.Linear(200,output_dim)
        self.avg_pool_bool = c.avg_pool
        self.input_dim = input_dim

    def forward(self, x):
        if self.avg_pool_bool == True:
            x = x.view(-1, 1, self.input_dim)
            x = self.avgPool(x)
            x = x.view(-1, x.shape[2])
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

