# Code based on michael hahn
import torch
import random
import time
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', type=str, default="random")
args = parser.parse_args()

baseline = args.baseline
SEQUENCE_LENGTH = 30
TEXT_LENGTH_BOUND = 500

OOV = 2
SKIPPED = 1
PAD = 0
PLACEHOLDER = 3

vocabulary = [x.split("\t") for x in open(f"./data/vocabulary/clue_corpus_small.txt", "r").read().strip().split("\n")]
itos = [x[1] for x in vocabulary]
stoi = dict([(x[1], int(x[0])) for x in vocabulary])

char_freq = {}
with open('./data/testing_data/char_freq.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count != 0:
            char_freq[row[1]] = {'CF_BLI':row[2], 'CF_SUB':row[3]}
        line_count += 1

def numerify(token):
    if token == "@placeholder":
        return PLACEHOLDER
    elif token not in stoi or stoi[token] >= 50000:
        return OOV
    else:
        return stoi[token]+4


def freq(token):
    if token == "@placeholder":
        return float('inf')
    else:
        if baseline == 'freq_BLI':
            return char_freq[token]['CF_BLI']
        elif baseline == 'freq_SUB':
            return char_freq[token]['CF_SUB']

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
fixation_rate = 0.8

char_embeddings = torch.nn.Embedding(num_embeddings = 50000+4, embedding_dim = 200).cuda()

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


def forward(batch, calculateAccuracy=False):
    texts = [[PAD] + [numerify(y) for y in x] + [PAD] for x in batch] # [:500]
    texts_freq = [[float('inf')] + [freq(y) for y in x] + [float('inf')] for x in batch]
    texts_freq_ = texts_freq.copy()
    text_length = max([len(x) for x in texts])
    for text, text_freq in zip(texts, texts_freq):
        while len(text) < text_length:
            text.append(PAD)
            text_freq.append(float('inf'))
    texts = torch.LongTensor(texts).cuda()
    if (baseline == 'freq_BLI') or (baseline == 'freq_SUB'):
        texts_freq = torch.FloatTensor(texts_freq).cuda()
        freq_threshold = torch.quantile(texts_freq_, fixation_rate)

    mask = torch.FloatTensor([1 for _ in range(len(batch))]).cuda()
    masked = torch.LongTensor([SKIPPED]).cuda().unsqueeze(1).expand(len(batch), texts.size()[1]-1)  #.cuda().unsqueeze(1).expand(len(batch), texts.size()[1]-1)
    hidden = None
    outputs = []

    if baseline == 'random':
        mask = torch.bernoulli(torch.FloatTensor([[fixation_rate for _ in range(texts.size()[0])] for _ in range(texts.size()[1]-1)])).cuda().transpose(0,1)  #.cuda()).transpose(0,1)
    
    if (baseline == 'freq_BLI') or (baseline == 'freq_SUB'):
        mask = torch.Tensor(texts_freq[:,:-1] <= freq_threshold, dtype=torch.int32).cuda().transpose(0,1)
        
    embedded_ = char_embeddings(torch.where(mask==1.0, texts[:,:-1], masked)).transpose(0,1)
    outputs_reader, hidden = reader(embedded_)

    embedded = char_embeddings(texts).transpose(0,1)
    if not calculateAccuracy:
        embedded = input_dropout(embedded)
    outputs_decoder, _ = reconstructor(embedded[:-1], hidden)
    targets = texts.transpose(0,1)
    targets = torch.cat([targets[1:], targets[1:]], dim=0)
    outputs_cat = output(torch.cat([outputs_reader, outputs_decoder], dim=0))
    loss = crossEntropy(outputs_cat.view(-1, 50004), targets.view(-1)).view(outputs_cat.size()[0], outputs_cat.size()[1])

    text_from_batch = []

    if True:
        sequenceLengthHere = text_length-2
        loss_reader = loss.cpu()
        for batch_ in range(loss.size()[1]):
            for pos in range(loss.size()[0]):
                try:
                    # print(batch[batch_][pos])
                    lineForWord = batch[batch_][pos]
                    text_from_batch.append([str(y) for y in [pos, lineForWord, "InVocab" if stoi.get(lineForWord, 100000) < 50000 else "OOV"] +[round(float(x),4) for x in [loss_reader[pos,batch_], mask[pos, batch_]]]])
                except IndexError:
                    pass
    
    loss = loss.mean(dim=0)
    return loss, text_from_batch
    

devLosses = []
lossRunningAverage = 6.4
#devAccuracies = []
devRewards = []
noImprovement = 0


concatenated = []
with open(f"./results/test_attention_baseline_{baseline}.txt", "w") as outFile:
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
        
