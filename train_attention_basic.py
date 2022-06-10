# Code based on michael hahn
import torch
import random
import time
import argparse

parser = argparse.ArgumentParser()

#parser.add_argument('--corpus', type=str, default="cnn")
parser.add_argument('--batchSize', type=int, default=64) #random.choice([16, 32]))
parser.add_argument('--learning_rate', type=float, default=random.choice([1.0]))
#parser.add_argument('--glove', type=bool, default=True)
parser.add_argument('--dropout', type=float, default=random.choice([0.0, 0.05, 0.1, 0.15, 0.2]))
#parser.add_argument('--myID', type=int, default=random.randint(1000,100000000))
#parser.add_argument('--SEQUENCE_LENGTH', type=int, default=50)
#parser.add_argument('--LAMBDA', type=float, default=2.25) #random.choice([1.5, 1.75, 2, 2.25, 2.5]))
#parser.add_argument('--REWARD_FACTOR', type=float, default=0.1)
#parser.add_argument('--ENTROPY_WEIGHT', type=float, default=0.005) #random.choice([0.0001, 0.001, 0.01, 0.1]))


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
    with open(f"./data/{partition}_data/clue_corpus_seg_small.txt", "rb") as inFile:
        while True:
            buff = []
            #print("Filling buffer...")
            for _ in range(SEQUENCE_LENGTH*batchSize):
                try:
                    buff.append(next(inFile).decode().strip().split(" "))
                except StopIteration:
                    break
            if len(buff) == 0:
                break
            random.shuffle(buff)
            
            concatenated = []
            for x in buff:
                for y in x:
                    concatenated.append(y)
        
            partitions = []
            for i in range(int(len(concatenated)/SEQUENCE_LENGTH)+1):
                r = concatenated[i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]
                if len(r) > 0:
                    partitions.append(r)
            random.shuffle(partitions)
        
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
    
dropout = args.dropout
learning_rate = args.learning_rate
batchSize = args.batchSize

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

optimizer = torch.optim.SGD(parameters(), lr = learning_rate)

my_save_path = f"./models/attention_basic.ckpt"
def SAVE():
       torch.save({"devRewards" : devRewards, "args" : args, "components_lm" : [x.state_dict() for x in components_lm], "components_attention" : [x.state_dict() for x in components_attention], "learning_rate" : learning_rate}, my_save_path)
          
    
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

    attentionLogProbability = torch.nn.functional.logsigmoid(torch.where(attentionDecisions == 1, attentionLogit, -attentionLogit))

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
    loss = loss.mean(dim=0)
    attentionDecisions = attentionDecisions.mean(dim=0)
    return loss, attentionLogProbability, attentionDecisions
  
clip_type = random.choice([2, "inf", "None"])
clip_bound = random.choice([2, 5, 10, 15])

fixationRunningAverageByCondition = [0.5,0.5]
rewardAverage = 10.0
lossAverageByCondition = [10.0, 10.0]

LAMBDA = 2.25
REWARD_FACTOR = 0.1
ENTROPY_WEIGHT = 0.005

def backward(loss, action_logprob, fixatedFraction, printHere=True):
    global rewardAverage
    '''if random.random() < 0.99:
        batchSize = fixatedFraction.size()[0]
        fix1 = float(fixatedFraction.mean())
        fixationRunningAverageByCondition[0] = 0.99 * fixationRunningAverageByCondition[0] + (1-0.99) * fix1

        loss1 = float(loss.mean())
        lossAverageByCondition[0] = 0.99 * lossAverageByCondition[0] + (1-0.99) * loss1
        if printHere:
            print("FIXATION RATE", fixationRunningAverageByCondition, "REWARD", rewardAverage, "LossByCondition", lossAverageByCondition)'''
    
    optimizer.zero_grad()
    action_prob = torch.exp(action_logprob).clamp(min=0.0001, max=1-0.0001)
    oneMinusActionProb = 1-action_prob
    negEntropy = (action_prob * action_logprob + oneMinusActionProb * oneMinusActionProb.log()).mean()
    reward = (loss.detach() + LAMBDA * fixatedFraction)
    loss_ = REWARD_FACTOR * (action_logprob * (reward - rewardAverage)).mean() + loss.mean() + ENTROPY_WEIGHT * negEntropy
    loss_.backward()
    rewardAverage = 0.99 * rewardAverage + (1-0.99) * float(reward.mean())
    #print("reward:", reward.mean())
    
    if clip_type != "None":
        # print("NROM MAX", max(p.grad.detach().abs().max().cpu() for p in parameters()))
        torch.nn.utils.clip_grad_norm_(parameters(), clip_bound, norm_type=clip_type)
    optimizer.step()
    

devLosses = []
lossRunningAverage = 6.4
#devAccuracies = []
devRewards = []
noImprovement = 0
for epoch in range(4):
    print()
    print("----------------- Epoch:", epoch, "-----------------")
    print("Start Validation...")
    if epoch >= 0:
        validLoss = []
        examplesNumber = 0
        validAccuracy = []
        validReward = []
        validAccuracyPerCondition = [0.0, 0.0]
        validFixationsPerCondition = [0.0, 0.0]
        counter2 = 0
        print("Validation loss and reward:")
        for batch in loadCorpus("validation", batchSize):
            with torch.no_grad():
                loss, action_logprob, fixatedFraction = forward(batch, calculateAccuracy = True)
                if loss is None:
                    continue
                reward = float((loss.detach() + LAMBDA * fixatedFraction).mean())
                loss = float(loss.mean())
                #print("VALID", loss, examplesNumber, reward)
            
            if counter2 % 1000 == 0:
                print("  |Batch", counter2, ": loss =", loss, ", reward =", reward)
            fixationsCond1 = float(fixatedFraction.sum())
            validFixationsPerCondition[0] += fixationsCond1

            validLoss.append(float(loss)*len(batch))
            validReward.append(reward*len(batch))
            examplesNumber += len(batch)
            counter2 += 1
        
        devLosses.append(sum(validLoss)/examplesNumber)
        devRewards.append(sum(validReward)/examplesNumber)
        validFixationsPerCondition[0] /= examplesNumber
        print("Mean valid loss:", sum(validLoss)/examplesNumber)
        print("Mean valid reward:", sum(validReward)/examplesNumber)
        
        with open(f"./results/train_attention_basic_result.txt", "w") as outFile:
            #print(args, file=outFile)
            #print(devAccuracies, file=outFile)
            print(devLosses, file=outFile)
            print(devRewards, file=outFile)
            print(fixationRunningAverageByCondition[0], fixationRunningAverageByCondition[1], "savePath="+my_save_path, file=outFile)
            print(rewardAverage, "\t", validAccuracyPerCondition[0], validAccuracyPerCondition[1], "\t", validFixationsPerCondition[0], validFixationsPerCondition[1], file=outFile)
        
        if len(devRewards) >1 and devRewards[-1] > devRewards[-2]:
            learning_rate *= 0.8
            optimizer = torch.optim.SGD(parameters(), lr = learning_rate)
            noImprovement += 1
        elif len(devRewards) > 1 and devRewards[-1] == min(devRewards):
            SAVE()
  
        # noImprovement = 0
        if noImprovement > 5:
            print("End training, no improvement for 5 epochs")
            break
    
    print()
    print("Start Training...")
    sent_counter = 0
    timeStart = time.time()
    counter = 0
    print("Training loss:")
    for batch in loadCorpus("training", batchSize):
        counter += 1
        printHere = (counter % 5) == 0
        loss, action_logprob, fixatedFraction = forward(batch)
        if loss is None:
            continue
        backward(loss, action_logprob, fixatedFraction, printHere=printHere)
        loss = float(loss.mean())
        lossRunningAverage = 0.99 *lossRunningAverage + (1-0.99) * float(loss)
        
        if counter % 1000 == 0:
          print("  |Batch", counter, ": loss =", loss)
        sent_counter += len(batch)
        '''if printHere:
            # print(devAccuracies)
            print(devLosses)
            print(devRewards)
            print(float(loss), lossRunningAverage, counter)
            print(sent_counter/(time.time()-timeStart), "sentences per second.")'''
        if noImprovement == 0 and counter % 10000 == 0:
            SAVE()
    
    print("Trained", sent_counter/(time.time()-timeStart), "sentences per second.")
