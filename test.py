'''import torch
print(torch.__version__)
print(torch.cuda.is_available())'''

import os
import numpy as np
embedding_useds = ["None", "CWE", "JWE"] # "None"

lambdas = list(np.array([*range(2500, 5500, 125)])/1000) #[3.125, 3.375, 3.625, 3.875, 4.125, 4.375, 4.625, 4.875, 5.125, 5.375, 5.625, 5.875] #[3.0, 3.5, 4.0, 4.5, 5.0] #[2.625, 2.875, 3.125, 3.375, 3.625, 3.875, 4.125, 4.375, 4.625, 4.875, 5.125] #[1.5, 2, 2.5] #[3.5, 4, 4.5, 5, 5.5] #[1.5, 2, 2.5]
params = [[True, True, 3, False]] #, [True, True, 3, False], [False, True, 3, True]]

for embedding_used in embedding_useds:
    for lambda_ in lambdas:
        #os.system(f'python test_attention_basic.py --embedding_used {embedding_used} --LAMBDA {lambda_}')

        for param in params:
            with_context, with_lm, preview_length, degraded_noise = param
            print("Test on parameters:", with_context, with_lm, preview_length, degraded_noise, lambda_)
            #os.system(f'python test_attention_preview.py --WITH_CONTEXT {with_context} --WITH_LM {with_lm} --previewLength {preview_length} --degradedNoise {degraded_noise} --embedding_used {embedding_used} --LAMBDA {lambda_}')
            os.system(f'python test_attention_saccadic_length.py --WITH_CONTEXT {with_context} --WITH_LM {with_lm} --previewLength {preview_length} --degradedNoise {degraded_noise} --embedding_used {embedding_used} --LAMBDA {lambda_}')
    
