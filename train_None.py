import os
import numpy as np

embedding_used = "None"
print("Batch size, learning rate, dropout, embedding:", 128, 0.1, 0.1, embedding_used)

lambdas = [4.0] #list(np.array([*range(3250, 6250, 250)])/1000)
reward_factors = [0.25] #[0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
entropy_weights = [0.02] #[0.001, 0.005, 0.01, 0.015, 0.02]

#params = [[False, False, 3, False]] #, [True, True, 3, False], [False, True, 3, True]]

for lambda_ in lambdas:
    for reward_factor_ in reward_factors:
        for entropy_weight_ in entropy_weights:
            print("Train on parameters:", lambda_, reward_factor_, entropy_weight_)
            os.system(f'python train_attention_basic.py --batchSize 128 --learning_rate 0.1 --dropout 0.1 --embedding_used {embedding_used} --LAMBDA {lambda_} --REWARD_FACTOR {reward_factor_} --ENTROPY_WEIGHT {entropy_weight_}')
            
            #for param in params:
                #with_context, with_lm, preview_length, degraded_noise = param
                #print("Train on parameters:", with_context, with_lm, preview_length, degraded_noise, lambda_)
                #os.system(f'python train_attention_preview.py --batchSize 128 --learning_rate 0.1 --dropout 0.1 --WITH_CONTEXT {with_context} --WITH_LM {with_lm} --previewLength {preview_length} --degradedNoise {degraded_noise} --embedding_used {embedding_used} --LAMBDA {lambda_}')
                #os.system(f'python train_attention_saccadic_length.py --batchSize 128 --learning_rate 0.1 --dropout 0.1 --WITH_CONTEXT {with_context} --WITH_LM {with_lm} --previewLength {preview_length} --degradedNoise {degraded_noise} --embedding_used {embedding_used} --LAMBDA {lambda_}')
