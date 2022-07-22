import os

embedding_used = "None"
print("Batch size, learning rate, dropout, embedding:", 128, 0.1, 0.1, embedding_used)

lambdas = [2.5] #[3.125, 3.375, 3.625, 3.875, 4.125, 4.375, 4.625, 4.875, 5.125, 5.375, 5.625, 5.875] #[1.5, 2, 2.5]
params = [[True, False, 0, True]] #, [True, True, 3, False], [False, True, 3, True]]

for lambda_ in lambdas:
    #os.system(f'python train_attention_basic.py --batchSize 128 --learning_rate 0.1 --dropout 0.1 --embedding_used {embedding_used} --LAMBDA {lambda_}')
    for param in params:
        with_context, with_lm, preview_length, degraded_noise = param
        print("Train on parameters:", with_context, with_lm, preview_length, degraded_noise, lambda_)
        os.system(f'python train_attention_preview.py --batchSize 128 --learning_rate 0.1 --dropout 0.1 --WITH_CONTEXT {with_context} --WITH_LM {with_lm} --previewLength {preview_length} --degradedNoise {degraded_noise} --embedding_used {embedding_used} --LAMBDA {lambda_}')
        #os.system(f'python train_attention_saccadic_length.py --batchSize 128 --learning_rate 0.1 --dropout 0.1 --WITH_CONTEXT {with_context} --WITH_LM {with_lm} --previewLength {preview_length} --degradedNoise {degraded_noise} --embedding_used {embedding_used} --LAMBDA {lambda_}')
