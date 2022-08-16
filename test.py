'''import torch
print(torch.__version__)
print(torch.cuda.is_available())'''

import os
import numpy as np
embedding_useds = ["CWE"]
gaussian_var1 = 0.0
gaussian_var2 = 0.1
gaussian_var3 = 0.5

lambdas = list(np.array([*range(2500, 5750, 250)])/1000)
reward_factors = [0.25] #[0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
entropy_weights = [0.001] #[0.001, 0.005, 0.01, 0.015, 0.02]
#params = [[True, False, 3, False], [False, True, 3, False], [True, True, 3, False], [False, False, 3, False], [False, False, 3, True]] 
params = [[True, False, 1, False]]#[[True, False, 1, False], [False, True, 1, False], [True, True, 1, False], [False, False, 3, False], [False, False, 2, False], [False, False, 3, True]] 

for embedding_used in embedding_useds:
    for lambda_ in lambdas:
        for reward_factor_ in reward_factors:
            for entropy_weight_ in entropy_weights:
                #os.system(f'python test_attention_basic.py --embedding_used {embedding_used} --LAMBDA {lambda_} --REWARD_FACTOR {reward_factor_} --ENTROPY_WEIGHT {entropy_weight_}')
            
                for param in params:
                    with_context, with_lm, preview_length, degraded_noise = param
                    print("Test on parameters:", with_context, with_lm, preview_length, degraded_noise, lambda_)
                    if preview_length == 1:
                        os.system(f'python test_attention_preview.py --WITH_CONTEXT {with_context} --WITH_LM {with_lm} --previewLength {preview_length} --degradedNoise {degraded_noise} --gaussianVars {gaussian_var1} --embedding_used {embedding_used} --LAMBDA {lambda_} --REWARD_FACTOR {reward_factor_} --ENTROPY_WEIGHT {entropy_weight_}')
                    if preview_length == 2:
                        os.system(f'python test_attention_preview.py --WITH_CONTEXT {with_context} --WITH_LM {with_lm} --previewLength {preview_length} --degradedNoise {degraded_noise} --gaussianVars {gaussian_var1} {gaussian_var2} --embedding_used {embedding_used} --LAMBDA {lambda_} --REWARD_FACTOR {reward_factor_} --ENTROPY_WEIGHT {entropy_weight_}')
                    if preview_length == 3:
                        os.system(f'python test_attention_preview.py --WITH_CONTEXT {with_context} --WITH_LM {with_lm} --previewLength {preview_length} --degradedNoise {degraded_noise} --gaussianVars {gaussian_var1} {gaussian_var2} {gaussian_var3} --embedding_used {embedding_used} --LAMBDA {lambda_} --REWARD_FACTOR {reward_factor_} --ENTROPY_WEIGHT {entropy_weight_}')
                        os.system(f'python test_attention_saccadic_length.py --WITH_CONTEXT {with_context} --WITH_LM {with_lm} --previewLength {preview_length} --degradedNoise {degraded_noise} --gaussianVars {gaussian_var1} {gaussian_var2} {gaussian_var3} --embedding_used {embedding_used} --LAMBDA {lambda_} --REWARD_FACTOR {reward_factor_} --ENTROPY_WEIGHT {entropy_weight_}')
    
