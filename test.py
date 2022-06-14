'''import torch
print(torch.__version__)
print(torch.cuda.is_available())'''



params = [[True, True, 3, True], [True, True, 3, False], [False, True, 3, True]]
for param in params:
  with_context, with_lm, preview_length, degraded_noise = param
  
