import torch.nn as nn

from models.BaseModel import BaseModel


class LSTM(BaseModel):
    def __init__(self, c, input_dim, input_channels, output_dim):
        super(LSTM, self).__init__(c.device)
        self.avgPool = nn.AvgPool1d(c.CNN_avg_pool_width, c.CNN_avg_pool_width, 2)
        if c.LSTM_bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.hidden_dim = c.LSTM_hidden_dim
        self.num_layers = c.LSTM_number_layers
        self.bidirectional_bool = c.LSTM_bidirectional
        self.lstm = nn.LSTM(1, c.LSTM_hidden_dim, c.LSTM_number_layers, bidirectional=c.LSTM_bidirectional, batch_first=True)
        self.Linear1 = nn.Sequential(nn.Linear(c.LSTM_hidden_dim * c.LSTM_number_layers * self.num_directions, 200), nn.ReLU())
        self.Linear2 = nn.Linear(200, output_dim)
        self.input_dim = input_dim

    def forward(self, x):
        x = x.view(-1, 1, self.input_dim)
        x = self.avgPool(x)
        x = x.transpose(1, 2)
        output, (h_n, c_n) = self.lstm(x)
        out = h_n.transpose(0, 1)
        out = out.contiguous().view(-1,  self.hidden_dim * self.num_layers * self.num_directions)
        out = self.Linear1(out)
        out = self.Linear2(out)
        return out

