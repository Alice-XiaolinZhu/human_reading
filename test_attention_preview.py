# Code based on michael hahn
import torch
import random
import time
from torch.distributions import Normal
import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser()

parser.add_argument('--WITH_CONTEXT', type=lambda x:bool(strtobool(x)), default=True)
parser.add_argument('--WITH_LM', type=lambda x:bool(strtobool(x)), default=True)
parser.add_argument('--previewLength', type=int, default=3)
parser.add_argument('--degradedNoise', type=lambda x:bool(strtobool(x)), default=True)
parser.add_argument('--gaussianVars', type=float, nargs="+", default=[]) # default=[0.02, 0.1, 0.5]
parser.add_argument('--embedding_used', type=str, default="None")
parser.add_argument('--LAMBDA', type=float, default=2.25) #random.choice([1.5, 1.75, 2, 2.25, 2.5]))
parser.add_argument('--REWARD_FACTOR', type=float, default=0.1)
parser.add_argument('--ENTROPY_WEIGHT', type=float, default=0.005)

args = parser.parse_args()
print("Parameters:", args)

SEQUENCE_LENGTH = 30
TEXT_LENGTH_BOUND = 500

OOV = 2
SKIPPED = 1
PAD = 0
PLACEHOLDER = 3

# ONLY_REC = True
WITH_CONTEXT = args.WITH_CONTEXT
WITH_LM = args.WITH_LM
previewLength = args.previewLength
degradedNoise = args.degradedNoise

vocabulary = [x.split("\t") for x in open(f"./data/vocabulary/clue_corpus_small.txt", "r").read().strip().split("\n")]
itos = [x[1] for x in vocabulary]
stoi = dict([(x[1], int(x[0])) for x in vocabulary])

def numerify(token):
    if token == "@placeholder":
        return PLACEHOLDER
    elif token not in stoi or stoi[token] >= 50000:
        return OOV
    else:
        return stoi[token]+4

def loadCorpus(partition, batchSize):
    assert partition in ["testing", "training", "validation"]
    with open(f"./data/{partition}_data/bsc_sentences.txt", "rb") as inFile:
        while True:
            buff = []
            #print("Filling buffer...")
            for _ in range(SEQUENCE_LENGTH*batchSize):
                try:
                    buff.append(list(next(inFile).decode().strip()))
                except StopIteration:
                    break
            if len(buff) == 0:
                break
            #random.shuffle(buff)
        
            concatenated = []
            for x in buff:
                for y in x:
                    concatenated.append(y)
        
            partitions = []
            for i in range(int(len(concatenated)/SEQUENCE_LENGTH)+1):
                r = concatenated[i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]
                if len(r) > 0:
                    partitions.append(r)
            #random.shuffle(partitions)
        
            for i in range(int(len(partitions)/batchSize)+1):
                r = partitions[i*batchSize:(i+1)*batchSize]
                if len(r) > 0:
                    yield r

                    
def parameters():
    for c in components_lm:
        for param in c.parameters():
            yield param
    for c in components_attention:
        for param in c.parameters():
            yield param
    yield runningAverageParameter
   

dropout = 0.2
learning_rate = 0.01
batchSize = 32

char_embeddings = torch.nn.Embedding(num_embeddings = 50000+4, embedding_dim = 200).cuda()
# char_embeddings.weight.data[0], char_embeddings(torch.LongTensor([0]))
# char_embeddings(torch.LongTensor([0])).size()

reader = torch.nn.LSTM(200, 1024, 1).cuda()
reconstructor = torch.nn.LSTM(200, 1024, 1).cuda()
output = torch.nn.Linear(1024, 50000 + 4).cuda()

input_dropout = torch.nn.Dropout(dropout)

nllLoss = torch.nn.NLLLoss(reduction="none", ignore_index=PAD)
crossEntropy = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=PAD)

components_lm = [char_embeddings, reader, reconstructor, output]

loaded = torch.load(f"./models/autoencoder_{args.embedding_used}.ckpt")
print(f"Load autoencoder model: ./models/autoencoder_{args.embedding_used}.ckpt") 

for i in range(len(loaded["components"])):
    components_lm[i].load_state_dict(loaded["components"][i])

# Use a linear module for derive attention logit from word embedding
# bilinear = torch.nn.Linear(200, 1).cuda()
if WITH_CONTEXT:
    # Use a bilinear module to combine word embedding with context information
    bilinear = torch.nn.Bilinear(1024, 200, 1).cuda()
else:
    # Use a linear module for derive attention logit from word embedding
    bilinear = torch.nn.Linear(200, 1).cuda()

bilinear.weight.data.zero_()
bilinear.bias.data.zero_()

components_attention = [bilinear]
# components_attention = [bilinear, char_embeddings_preview, char_embeddings]
runningAverageParameter = torch.FloatTensor([0]).cuda()

# optimizer = torch.optim.SGD(parameters(), lr = learning_rate)


state = torch.load(f"./models/attention_{args.WITH_CONTEXT}_{args.WITH_LM}_{args.previewLength}_{args.degradedNoise}_{args.embedding_used}_{args.LAMBDA}_{args.REWARD_FACTOR}_{args.ENTROPY_WEIGHT}.ckpt")
print(f"Load attention model: ./models/attention_{args.WITH_CONTEXT}_{args.WITH_LM}_{args.previewLength}_{args.degradedNoise}_{args.embedding_used}_{args.LAMBDA}_{args.REWARD_FACTOR}_{args.ENTROPY_WEIGHT}.ckpt") 

# print("args", state["args"])
# print(state["devRewards"])

if len(state["devRewards"]) < 2:
    quit()
for i in range(len(components_lm)):
    components_lm[i].load_state_dict(state["components_lm"][i])
for i in range(len(components_attention)):
    components_attention[i].load_state_dict(state["components_attention"][i])


def forward(batch, calculateAccuracy=False):
    texts = [[PAD] + [numerify(y) for y in x] + [PAD] for x in batch] # [:500]
    text_length = max([len(x) for x in texts])
    for text in texts:
        while len(text) < text_length:
            text.append(PAD)
    texts = torch.LongTensor(texts).cuda()
    
    texts_ = [[PAD] + [numerify(y) for y in x] + [PAD for _ in range(previewLength)] for x in batch] # [:500]
    text_length_ = max([len(x) for x in texts_])
    for text in texts_:
        while len(text) < text_length_:
            text.append(PAD)
    texts_preview = [[texts_[i][j:j+previewLength] for j in range(text_length_-previewLength+1)] for i in range(len(texts_))]
    texts_preview = torch.LongTensor(texts_preview).cuda()
    #print("texts_preview:", texts_preview.size())
    
    texts_preview_embedded = char_embeddings(texts_preview)
    #print("texts_preview_embedded:", texts_preview_embedded.size())
    #print(texts_preview_embedded[0])
    
    assert len(gaussian_vars) == previewLength
    if previewLength > 0:
        noise_size = [texts_preview_embedded.size()[0], texts_preview_embedded.size()[1], 1, texts_preview_embedded.size()[3]]
        if degradedNoise:
            # insert degraded gaussian noise to preview text embeddings input
            preview_noise = []
            #preview_noise.append(torch.zeros(noise_size).cuda())
            for gaussian_var in gaussian_vars:
                gaussian_noise = Normal(torch.tensor([0.0]).cuda(), torch.tensor([gaussian_var]).cuda())
                preview_noise.append(gaussian_noise.sample(noise_size).squeeze(-1))
            texts_preview_noise = torch.cat(preview_noise, dim=2)
            # print(texts_preview_embedded.size(), texts_preview_noise.size())
    
        else:
            # insert gaussian noise only to the last character of the preview text
            gaussian_noise = Normal(torch.tensor([0.0]).cuda(), torch.tensor([gaussian_vars[-1]]).cuda())
            preview_noise = torch.zeros(texts_preview_embedded.size()).cuda()
            preview_noise[:,:,-1:,:] = gaussian_noise.sample(noise_size).squeeze(-1)
            texts_preview_noise = preview_noise
        
        noised_texts_preview_embedded = texts_preview_embedded + texts_preview_noise
        # print(texts_preview_embedded[0], texts_preview_noise[0], noised_texts_preview_embedded[0])
    
    elif previewLength == 1:
        noised_texts_preview_embedded = texts_preview_embedded
        
    noised_texts_preview_embedded = noised_texts_preview_embedded.mean(dim=2)
    # print("noised_texts_preview_embedded:", noised_texts_preview_embedded.size())
    # print(noised_texts_preview_embedded[0])
    
    mask = torch.FloatTensor([1 for _ in range(len(batch))]).cuda()
    masked = torch.LongTensor([SKIPPED]).cuda().expand(len(batch))  #.cuda().expand(len(batch))
    hidden = None
    outputs = []

    attentionProbability_ = []
    attentionDecisions_ = []
    attentionLogit_ = []

    if not WITH_CONTEXT:
        # Calculate the context-independent attention logits
        attention_logits_total = bilinear(noised_texts_preview_embedded)  # without context
    
    # Iterate over the input
    for i in range(texts.size()[1]-1):
        #print("size:", mask.size(), texts[:,i].size(), masked.size())  # 16
        #print("texts:", texts[:,i])
        #print("mask:", mask)
        #print("masked:", masked)
        #print("masked texts:", torch.where(mask==1.0, texts[:,i], masked))
        embedded_ = char_embeddings(torch.where(mask==1.0, texts[:,i], masked)).unsqueeze(0)  # 0: mask
        #print(embedded_.size())  # 1,16,200
        _, hidden = reader(embedded_, hidden)
        outputs.append(hidden[0])
        #print("HIDDEN size:", hidden[0].size(), hidden[1].size())  # hidden, cell: 1,16,1024
        
        if WITH_CONTEXT:
            # Calculate the context-dependent attention logits
            embedded_ = noised_texts_preview_embedded[:,i+1]
            attention_logits = bilinear(hidden[0].squeeze(0), embedded_).squeeze(1)
            attentionProbability = torch.sigmoid(attention_logits)
            attentionDecisions = torch.bernoulli(torch.clamp(attentionProbability, min=0.01, max=0.99))
        else:
            attention_logits = attention_logits_total[:,i+1].squeeze(1)
            #print(attention_logits.size())
            attentionProbability = torch.sigmoid(attention_logits)
            attentionDecisions = torch.bernoulli(torch.clamp(attentionProbability, min=0.01, max=0.99))  # bernoulli, generate 0/1 based on input prob
        
        mask = attentionDecisions
        #print("attention_logits:", attention_logits)
        #print("attentionProbability", attentionProbability)
        #print("attentionDecisions", attentionDecisions)
        
        attentionProbability_.append(attentionProbability)
        attentionDecisions_.append(attentionDecisions)
        attentionLogit_.append(attention_logits)

    attentionProbability = torch.stack(attentionProbability_, dim=0)
    attentionDecisions = torch.stack(attentionDecisions_, dim=0)
    attentionLogit = torch.stack(attentionLogit_, dim=0)
    #print("attentionProbability:", attentionProbability.size(), attentionProbability)  # 51,16

    embedded = char_embeddings(texts).transpose(0,1)
    outputs_decoder, _ = reconstructor(embedded[:-1], hidden)
   
    targets = texts.transpose(0,1).contiguous()
    targets = targets[1:]
    outputs_cat = output(outputs_decoder)
    loss = crossEntropy(outputs_cat.view(-1, 50004), targets.view(-1)).view(outputs_cat.size()[0], outputs_cat.size()[1])

    outputs_reader = torch.cat(outputs, dim=0)
    loss_reader = crossEntropy(output(outputs_reader).view(-1, 50004), targets.view(-1)).view(outputs_cat.size()[0], outputs_cat.size()[1])

    # attentionLogProbability = torch.nn.functional.logsigmoid(torch.where(attentionDecisions == 1, attentionLogit, -attentionLogit))

    # calculate perplexity of decoder LM
    loss_lm = loss.sum(0)/(loss!=0).sum(0)  # torch.mean(loss, dim=0)
    perplexity_lm = torch.exp(loss_lm)
    
    # At random times, print surprisals and reconstruction losses
    '''if random.random() < 0.1:
        print(len(texts), loss.size(), targets.size(), attentionProbability.size(), attentionDecisions.size())
        # loss_reconstructor = loss[:SEQUENCE_LENGTH, 0].cpu()
        if WITH_LM:
            loss_reader = loss[:10, 0].cpu()
            loss_reconstructor = loss[51:61, 0].cpu()
        else:
            loss_reconstructor = loss[:10, 0].cpu()
        attentionProbability_ = attentionProbability[:SEQUENCE_LENGTH, 0].cpu()
        attentionDecisions_ = attentionDecisions[:SEQUENCE_LENGTH, 0].cpu()
        print("\t".join(["Pos", "Word", "Pred", "Rec", "AttProb", "Att?"]))
        for j in range(SEQUENCE_LENGTH):
            try:
                if WITH_LM:
                    print("\t".join([str(y) for y in [j, batch[0][j]] +[round(float(x),4) for x in [loss_reader[j], loss_reconstructor[j], attentionProbability_[j], attentionDecisions_[j]]]]))
                else:
                    print("\t".join([str(y) for y in [j, batch[0][j]] +[round(float(x),4) for x in [-1, loss_reconstructor[j], attentionProbability_[j], attentionDecisions_[j]]]]))
                    # Note that I'm using -1 for surprisal as this version of the model doesn't compute surprisal
            except IndexError:
                print(j, "IndexError")
    #       quit()'''
 
    #print("size:", loss.size(), loss.mean(dim=0).size())  # 51,16
    #print("loss:", loss.mean(dim=0))
    #print("size:", attentionDecisions.size(), attentionDecisions.mean(dim=0).size())
    #print("attentionDecisions:", attentionDecisions.mean(dim=0))
    
    text_from_batch = []

    if True:
        sequenceLengthHere = text_length-2
        #loss_reader = loss.cpu()
        loss_ = loss.cpu()
        loss_reader_ = loss_reader.cpu()
        attentionProbability_ = attentionProbability.cpu()
        attentionDecisions_ = attentionDecisions.cpu()
        for batch_ in range(loss.size()[1]):
            for pos in range(loss.size()[0]):
                try:
                    # print(batch[batch_][pos])
                    lineForWord = batch[batch_][pos]
                    text_from_batch.append([str(y) for y in [pos, lineForWord, "InVocab" if stoi.get(lineForWord, 100000) < 50000 else "OOV"] +[round(float(x),4) for x in [loss_[pos,batch_], loss_reader_[pos,batch_], attentionProbability_[pos,batch_], attentionDecisions_[pos, batch_]]]])
                except IndexError:
                    pass
    
    loss = loss.mean(dim=0)
    # attentionDecisions = attentionDecisions.mean(dim=0)
    return loss, text_from_batch #, attentionLogProbability, attentionDecisions
  
  
gaussian_vars = args.gaussianVars
devLosses = []
lossRunningAverage = 6.4
#devAccuracies = []
devRewards = []
noImprovement = 0


concatenated = []
with open(f"./results/test_attention_{args.WITH_CONTEXT}_{args.WITH_LM}_{args.previewLength}_{args.degradedNoise}_{args.embedding_used}_{args.LAMBDA}_{args.REWARD_FACTOR}_{args.ENTROPY_WEIGHT}.txt", "w") as outFile:
    validLoss = []
    examplesNumber = 0
    counter = 1
    TEXT_ = []
    batches = list(loadCorpus("testing", batchSize))
    print("Number of batches", len(batches))
    for batch in batches:
        with torch.no_grad():
            loss, TEXT = forward(batch, calculateAccuracy = True)
            loss = float(loss.mean())
            #print("VALID", loss, examplesNumber)
            for x in TEXT:
                TEXT_.append(x)
        
        #print("  |Batch", counter, ": loss =", loss)
        validLoss.append(float(loss)*len(batch))
        examplesNumber += len(batch)
        count = 0
        counter += 1
    for x in TEXT_:
        print("\t".join(x), file=outFile)
