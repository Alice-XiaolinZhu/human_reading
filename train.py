import os

embedding_used = "None"
print("Batch size, learning rate, dropout, embedding:", 32, 0.1, 0.1, embedding_used)
#os.system(f'python train_attention_basic.py --batchSize 32 --learning_rate 0.1 --dropout 0.1 --embedding_used {embedding_used}')

params = [[True, True, 3, False], [False, True, 3, True]] # [[True, True, 3, True], [True, True, 3, False], [False, True, 3, True]]
for param in params:
    with_context, with_lm, preview_length, degraded_noise = param
    print("Train on parameters:", with_context, with_lm, preview_length, degraded_noise)
    os.system(f'python train_attention_preview.py --batchSize 32 --learning_rate 0.1 --dropout 0.1 --WITH_CONTEXT {with_context} --WITH_LM {with_lm} --previewLength {preview_length} --degradedNoise {degraded_noise} --embedding_used {embedding_used}')
    os.system(f'python train_attention_saccadic_length.py --batchSize 32 --learning_rate 0.1 --dropout 0.1 --WITH_CONTEXT {with_context} --WITH_LM {with_lm} --previewLength {preview_length} --degradedNoise {degraded_noise} --embedding_used {embedding_used}')
