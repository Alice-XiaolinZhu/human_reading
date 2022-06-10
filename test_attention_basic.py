# Code based on michael hahn
import torch
import random
import time
import argparse

parser = argparse.ArgumentParser()

args = parser.parse_args()

SEQUENCE_LENGTH = 30
TEXT_LENGTH_BOUND = 500

OOV = 2
SKIPPED = 1
PAD = 0
PLACEHOLDER = 3

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
# word_embeddings.weight.data[0], word_embeddings(torch.LongTensor([0]))
# word_embeddings(torch.LongTensor([0])).size()

reader = torch.nn.LSTM(200, 1024, 1).cuda()
reconstructor = torch.nn.LSTM(200, 1024, 1).cuda()
output = torch.nn.Linear(1024, 50000 + 4).cuda()

input_dropout = torch.nn.Dropout(dropout)

nllLoss = torch.nn.NLLLoss(reduction="none", ignore_index=PAD)
crossEntropy = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=PAD)

components_lm = [char_embeddings, reader, reconstructor, output]

loaded = torch.load(f"./models/autoencoder.ckpt")
for i in range(len(loaded["components"])):
    components_lm[i].load_state_dict(loaded["components"][i])

# Use a linear module for derive attention logit from word embedding
bilinear = torch.nn.Linear(200, 1).cuda()
bilinear.weight.data.zero_()
bilinear.bias.data.zero_()

components_attention = [bilinear]
runningAverageParameter = torch.FloatTensor([0]).cuda()

state = torch.load(f"./models/attention_basic.ckpt")

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

    mask = torch.FloatTensor([1 for _ in range(len(batch))]).cuda()
    masked = torch.LongTensor([SKIPPED]).cuda().expand(len(batch))  #.cuda().expand(len(batch))
    hidden = None
    outputs = []

    attentionProbability_ = []
    attentionDecisions_ = []
    attentionLogit_ = []

    # Calculate the context-independent attention logits
    attention_logits_total = bilinear(char_embeddings(texts))  # without context
    
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
        
        attention_logits = attention_logits_total[:,i+1].squeeze(1)
        #print(attention_logits.size())
        attentionProbability = torch.nn.functional.sigmoid(attention_logits)
        attentionDecisions = torch.bernoulli(attentionProbability)  # bernoulli, generate 0/1 based on input prob
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
    
    # Collect target values for decoding loss
    targets = texts.transpose(0,1).contiguous()
    targets = targets[1:]
    outputs_cat = output(outputs_decoder)
    loss = crossEntropy(outputs_cat.view(-1, 50004), targets.view(-1)).view(outputs_cat.size()[0], outputs_cat.size()[1])

    # attentionLogProbability = torch.nn.functional.logsigmoid(torch.where(attentionDecisions == 1, attentionLogit, -attentionLogit))

    # At random times, print surprisals and reconstruction losses
    '''if random.random() < 0.1:
        print(len(texts), loss.size(), targets.size(), attentionProbability.size(), attentionDecisions.size())
        loss_reconstructor = loss[:SEQUENCE_LENGTH, 0].cpu()
        attentionProbability_ = attentionProbability[:SEQUENCE_LENGTH, 0].cpu()
        attentionDecisions_ = attentionDecisions[:SEQUENCE_LENGTH, 0].cpu()
        print("\t".join(["Pos", "Word", "Pred", "Rec", "AttProb", "Att?"]))
        for j in range(SEQUENCE_LENGTH):
            try:
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
        loss_reader = loss.cpu()
        attentionProbability_ = attentionProbability.cpu()
        attentionDecisions_ = attentionDecisions.cpu()
        for batch_ in range(loss.size()[1]):
            for pos in range(loss.size()[0]):
                try:
                    # print(batch[batch_][pos])
                    lineForWord = batch[batch_][pos]
                    text_from_batch.append([str(y) for y in [pos, lineForWord, "InVocab" if stoi.get(lineForWord, 100000) < 50000 else "OOV"] +[round(float(x),4) for x in [loss_reader[pos,batch_], attentionProbability_[pos,batch_], attentionDecisions_[pos, batch_]]]])
                except IndexError:
                    pass
    
    loss = loss.mean(dim=0)
    # attentionDecisions = attentionDecisions.mean(dim=0)
    return loss, text_from_batch # , attentionLogProbability, attentionDecisions
    

devLosses = []
lossRunningAverage = 6.4
#devAccuracies = []
devRewards = []
noImprovement = 0


concatenated = []
with open(f"./results/test_attention_basic.txt", "w") as outFile:
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
