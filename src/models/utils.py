import importlib


def load_model(c, input_dim, input_channels, output_dim):
  '''
  Import the model module and instantiate
  '''
  model_module = importlib.import_module('models.' + c.model_type)
  model_class = getattr(model_module, c.model_type)

  return model_class(c, input_dim, input_channels, output_dim)


