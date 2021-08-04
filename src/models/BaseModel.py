import torch
import torch.nn as nn

class BaseModel(nn.Module):
  def __init__(self, device=None):
    super().__init__()
    self.device = device
  def summary(self, input_shape):
    from torchsummaryX import summary
    print("\n================================================================")
    print("SUMMARY FOR MODEL:", type(self).__name__)
    print("Input dimensions:", input_shape)
    print("================================================================")
    print(str(self.device))
    device = 'cuda' if 'cuda' in str(self.device) else 'cpu'
    summary(self, torch.zeros(input_shape).to(device))
    print(flush=True)

  def additional_outputs(self):
    """For additional logging during evaluation, return a dict of numpy arrays"""
    return None
