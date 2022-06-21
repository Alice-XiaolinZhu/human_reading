'''import torch
print(torch.__version__)
print(torch.cuda.is_available())'''

import os
embedding_useds = ["None", "CWE", "JWE"] # "None"

lambdas = lambdas = [1.5, 2, 2.5] #[3.5, 4, 4.5, 5, 5.5] #[1.5, 2, 2.5]
params = [[True, True, 3, False]] #, [True, True, 3, False], [False, True, 3, True]]

for embedding_used in embedding_useds:
    for lambda_ in lambdas:
        #os.system(f'python test_attention_basic.py --embedding_used {embedding_used} --LAMBDA {lambda_}')

        for param in params:
            with_context, with_lm, preview_length, degraded_noise = param
            print("Test on parameters:", with_context, with_lm, preview_length, degraded_noise, lambda_)
            #os.system(f'python test_attention_preview.py --WITH_CONTEXT {with_context} --WITH_LM {with_lm} --previewLength {preview_length} --degradedNoise {degraded_noise} --embedding_used {embedding_used} --LAMBDA {lambda_}')
            os.system(f'python test_attention_saccadic_length.py --WITH_CONTEXT {with_context} --WITH_LM {with_lm} --previewLength {preview_length} --degradedNoise {degraded_noise} --embedding_used {embedding_used} --LAMBDA {lambda_}')
    
