'''import torch
print(torch.__version__)
print(torch.cuda.is_available())'''

import os
os.system('python test_attention_basic.py')

params = [[True, True, 3, True], [True, True, 3, False], [False, True, 3, True]]
for param in params:
    with_context, with_lm, preview_length, degraded_noise = param
    os.system(f'python test_attention_preview.py --WITH_CONTEXT {with_context} --WITH_LM {with_lm} --previewLength {preview_length} --degradedNoise {degraded_noise}')
    os.system(f'python test_attention_saccadic_length.py --WITH_CONTEXT {with_context} --WITH_LM {with_lm} --previewLength {preview_length} --degradedNoise {degraded_noise}')
