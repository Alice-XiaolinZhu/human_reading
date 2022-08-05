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
parser.add_argument('--embedding_used', type=str, default="None")

args = parser.parse_args()

SEQUENCE_LENGTH = 30
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
    
def parameters():
    for c in components_lm:
        for param in c.parameters():
            yield param
            
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
                    
dropout = args.dropout
learning_rate = args.learning_rate
batchSize = args.batchSize

char_embeddings = torch.nn.Embedding(num_embeddings = 50000+4, embedding_dim = 200).cuda()
# char_embeddings.weight.data[0], char_embeddings(torch.LongTensor([0]))
# char_embeddings(torch.LongTensor([0])).size()

# embedding_used = 'CWE'
embed_vec_cat = {}
if args.embedding_used == 'CWE':
    print(f"Loading {args.embedding_used} embeddings...")
    with open("./data/embeddings/charCWE.txt", "r", encoding='utf-8') as inFile:
        next(inFile)
        for line in inFile:
            line = line.strip().split('\t')
            char = line[0]
            if char not in embed_vec_cat:
                embed_vec_cat[char] = []
            if char in embed_vec_cat:
                embed_vec = torch.FloatTensor([float(x) for x in line[2:]]).cuda()
                embed_vec_cat[char].append(embed_vec)
    counter = 0
    for char in embed_vec_cat:
        counter += 1
        if char in stoi and stoi[char] < 50000:
            embedding = torch.mean(torch.stack(embed_vec_cat[char]),dim=0)
            char_embeddings.weight.data[stoi[char]+4] = embedding
        if counter > 100000:
            break
    print("Done loading embeddings.")
    
if args.embedding_used == 'JWE':
    print(f"Loading {args.embedding_used} embeddings...")
    with open("./data/embeddings/charJWE.txt", "r", encoding='utf-8') as inFile:
        next(inFile)
        counter = 0
        for line in inFile:
            counter += 1
            line = line.strip().split('\t')
            char = line[0]
            if char in stoi and stoi[char] < 50000:
                embedding = torch.FloatTensor([float(x) for x in line[1].split(' ')]).cuda()
                char_embeddings.weight.data[stoi[char]+4] = embedding
            if counter > 100000:
                break
    print("Done loading embeddings.")
    
if args.embedding_used == 'CW2VEC':
    print(f"Loading {args.embedding_used} embeddings...")
    with open("./data/embeddings/charCW2VEC.txt", "r", encoding='utf-8') as inFile:
        next(inFile)
        counter = 0
        for line in inFile:
            counter += 1
            line = line.strip().split('\t')
            char = line[0]
            if char in stoi and stoi[char] < 50000:
                embedding = torch.FloatTensor([float(x) for x in line[1].split(' ')]).cuda()
                print(embedding)
                print(jk)
                char_embeddings.weight.data[stoi[char]+4] = embedding
            if counter > 500000:
                break
    print("Done loading embeddings.")

reader = torch.nn.LSTM(200, 1024, 1).cuda()
reconstructor = torch.nn.LSTM(200, 1024, 1).cuda()
output = torch.nn.Linear(1024, 50000 + 4).cuda()

input_dropout = torch.nn.Dropout(dropout)

nllLoss = torch.nn.NLLLoss(reduction="none", ignore_index=PAD)
crossEntropy = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=PAD)

components_lm = [char_embeddings, reader, reconstructor, output]

optimizer = torch.optim.SGD(parameters(), lr = learning_rate)

def forward(batch, calculateAccuracy=False):
    texts = [[PAD] + [numerify(y) for y in x] + [PAD] for x in batch] # [:500]
    text_length = max([len(x) for x in texts])
    for text in texts:
        while len(text) < text_length:
            text.append(PAD)
    texts = torch.LongTensor(texts).cuda()

    mask = torch.FloatTensor([1 for _ in range(len(batch))]).cuda()
    masked = torch.LongTensor([SKIPPED]).cuda().unsqueeze(1).expand(len(batch), texts.size()[1]-1)  #.cuda().unsqueeze(1).expand(len(batch), texts.size()[1]-1)
    hidden = None
    outputs = []

    mask = torch.bernoulli(torch.FloatTensor([[0.95 for _ in range(texts.size()[0])] for _ in range(texts.size()[1]-1)]).cuda()).transpose(0,1)  #.cuda()).transpose(0,1)
    #print("size:", mask.size(), texts.size(), masked.size())
    #print("texts:", texts)
    #print("mask:", mask)
    #print("masked:", masked)
    #print("masked texts:", torch.where(mask==1.0, texts[:,:-1], masked))
    
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
    
    # calculate perplexity of decoder LM
    loss_lm = loss[outputs_decoder.size()[0]:].sum(0)/(loss[outputs_decoder.size()[0]:]!=0).sum(0)  # torch.mean(loss[outputs_decoder.size()[0]:], dim=0)
    perplexity_lm = torch.exp(loss_lm)
    
    '''if random.random() < 0.02:
        sequenceLengthHere= text_length-2
        # assert sequenceLengthHere == SEQUENCE_LENGTH, (text_length, SEQUENCE_LENGTH)
        loss_reader = loss[:SEQUENCE_LENGTH, 0].cpu()
        loss_reconstructor = loss[(sequenceLengthHere+1):(sequenceLengthHere+SEQUENCE_LENGTH+1), 0].cpu()
        print("\t".join(["Pos", "Char", "Pred", "Rec", "AttProb", "Att?"]))
        for j in range(SEQUENCE_LENGTH):
             print("\t".join([str(y) for y in [j, batch[0][j]] +[round(float(x),4) for x in [loss_reader[j], loss_reconstructor[j]]]]))
        # quit()'''

    #print("size:", loss.size(), loss.mean(dim=0).size())
    #print("loss:", loss.mean(dim=0))
    loss = loss.mean(dim=0)  # avg reader and decoder loss of a batch of sents
    return loss, perplexity_lm
  
lossAverageByCondition = [10.0, 10.0]
clip_type = random.choice([2, "inf", "None"])
clip_bound = random.choice([2, 5, 10, 15])

def backward(loss, printHere=True):
    '''if random.random() < 0.99:
        loss1 = float(loss.mean())
        lossAverageByCondition[0] = 0.99 * lossAverageByCondition[0] + (1-0.99) * loss1
        if printHere:
            print("LossByCondition", lossAverageByCondition)'''
    optimizer.zero_grad()
    loss_ = loss.mean()
    loss_.backward()
    if clip_type != "None":
        # print("NROM MAX", max(p.grad.detach().abs().max().cpu() for p in parameters()))
        torch.nn.utils.clip_grad_norm_(parameters(), clip_bound, norm_type=clip_type)
    optimizer.step()
    
trainLosses = []
trainPerplexities = []
devLosses = []
devPerplexity = []
trainLossesAll = []
devLossesAll = []
lossRunningAverage = 6.4
noImprovement = 0
timeStart_ = time.time()
for epoch in range(50):
    print()
    print("----------------- Epoch:", epoch, "-----------------")
    print("Start Training...")
    
    sent_counter = 0
    timeStart = time.time()
    counter = 0
    trainLoss = []
    trainPerplexity = []

    examplesNumber2 = 0
    print("Training loss:")
    losses_train_ = []
    for batch in loadCorpus("training", batchSize):
        counter += 1
        printHere = (counter % 25) == 0
        loss, perplexity_lm = forward(batch)
        backward(loss, printHere=printHere)
        loss = float(loss.mean())
        perplexity_lm = float(perplexity_lm.mean())
        if counter % 1000 == 0:
            print("  |Batch", counter, ": loss =", loss, ", perplexity =", perplexity_lm)
        
        trainLoss.append(float(loss)*len(batch))
        trainPerplexity.append(float(perplexity_lm)*len(batch))
        examplesNumber2 += len(batch)
        trainLossesAll.append(loss)
        
        lossRunningAverage = 0.99 *lossRunningAverage + (1-0.99) * float(loss)
        sent_counter += len(batch)
#         if counter % 20000 == 0 and SEQUENCE_LENGTH < args.SEQUENCE_LENGTH:
#             SEQUENCE_LENGTH = min(SEQUENCE_LENGTH+5, args.SEQUENCE_LENGTH)
#         if counter % 200000 == 0 and SEQUENCE_LENGTH == args.SEQUENCE_LENGTH and epoch == 0:
#             #torch.save({"devLosses" : devLosses, "args" : args, "components" : [x.state_dict() for x in components_lm], "learning_rate" : learning_rate}, f"/u/scr/mhahn/NEURAL_ATTENTION_TASK/checkpoints_2020/{__file__}_{args.myID}.ckpt")
#             torch.save({"devLosses" : devLosses, "components" : [x.state_dict() for x in components_lm], "learning_rate" : learning_rate}, f"./models/model.ckpt")        
        '''if printHere:
            print(devLosses)
            print("Mean loss over a batch:", float(loss), "lossRunningAverage", lossRunningAverage, counter)
            print("Trained", SEQUENCE_LENGTH*sent_counter/(time.time()-timeStart), "characters per second.")
            print()'''
    
    trainLosses.append(sum(trainLoss)/examplesNumber2)
    trainPerplexities.append(sum(trainPerplexity)/examplesNumber2)
    print("Trained", SEQUENCE_LENGTH*sent_counter/(time.time()-timeStart), "characters per second.")
    '''for c in components_lm:
        for name, param in c.named_parameters():
            print(name, ":", param.size())
            print(param)'''
    
    print()
    print("Start Validation...")
    #if epoch > 0:
    validLoss = []
    validPerplexity = []
    examplesNumber = 0
    counter2 = 0
    print("Validation loss:")
    for batch in loadCorpus("validation", batchSize):
        counter2 += 1
        with torch.no_grad():
            loss, perplexity_lm = forward(batch, calculateAccuracy = True)
            loss = float(loss.mean())
            perplexity_lm = float(perplexity_lm.mean())
            if counter2 % 1000 == 0:
                print("  |Batch", counter2, ": loss =", loss, ", perplexity =", perplexity_lm)

        validLoss.append(float(loss)*len(batch))
        validPerplexity.append(float(perplexity_lm)*len(batch))
        examplesNumber += len(batch)
        devLossesAll.append(loss)
    
    devLosses.append(sum(validLoss)/examplesNumber)
    devPerplexity.append(sum(validPerplexity)/examplesNumber)
    print("Mean valid loss:", sum(validLoss)/examplesNumber)
    print("Mean valid perplexity:", sum(validPerplexity)/examplesNumber)
        
    with open(f"./results/autoencoder_accuracy_{args.embedding_used}.txt", "w") as outFile:
        # print(args, file=outFile)
        print(f"Mean Training Loss for Epochs: {trainLosses}", file=outFile)
        print(f"Mean Training Perplexity for Epochs: {trainPerplexities}", file=outFile)
        print(f"Mean Validation Loss for Epochs: {devLosses}", file=outFile)
        print(f"Mean Validation Perplexity for Epochs: {devPerplexity}", file=outFile)
        
    
    if len(devLosses) >1 and devLosses[-1] > devLosses[-2]:
        learning_rate *= 0.8
        print("Tune learning rate:", learning_rate)
        optimizer = torch.optim.SGD(parameters(), lr = learning_rate)
        noImprovement += 1
    elif len(devLosses) > 1 and devLosses[-1] < max(devLosses):
        torch.save({"devLosses" : devLosses, "args" : args, "components" : [x.state_dict() for x in components_lm], "learning_rate" : learning_rate}, f"./models/autoencoder_{args.embedding_used}.ckpt")
        # torch.save({"devLosses" : devLosses, "components" : [x.state_dict() for x in components_lm], "learning_rate" : learning_rate}, f"./models/autoencoder.ckpt")
        noImprovement = 0
    if noImprovement > 5:
        print("End training, no improvement for 5 epochs")
        break
    print("Processing time:", time.time()-timeStart)
    
print(f"Total processing time:", time.time()-timeStart_)

'''from matplotlib import pyplot as plt
x = [i for i in range(len(trainLosses))]
plt.figure(figsize=(6,4))
plt.plot(x, trainLosses, label='Training loss')
plt.plot(x, valLosses, label='Validation loss')
plt.title('Training loss vs validation loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
#plt.show()
plt.savefig(f"./results/Train_autoencoder_losses_{args.embedding_used}.png")

x = [i for i in range(len(trainLossesAll))]
lt.figure(figsize=(6,4))
plt.plot(x, trainLossesAll, label='Training loss')
plt.plot(x, valLossesAll, label='Validation loss')
plt.title('Training losses vs validation losses over batches')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.legend()
#plt.show()
plt.savefig(f"./results/Train_autoencoder_all_losses_{args.embedding_used}.png")'''
