import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from torch import Tensor
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

BATCH_SIZE = 128
EMB_SIZE = 128
VOCAB_SZ = 20 + 3
SRC_VOCAB_SIZE = VOCAB_SZ
TGT_VOCAB_SIZE = 2



###############################
NUM_EPOCHS = 100
LEARNING_RATE = 0.00005 # best = 0.000004
DOWN_WEIGHT = 1.0
CUTOFF = 1.0
###############################






PAD_IDX = 'O'
BOS_IDX, EOS_IDX = 1, 2
MAX_LEN = 16


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AAs=np.array(list('ARNDCEQGHILKMFPSTWYV'))
curPath=os.getcwd()
AAidx_file='AAidx_PCA.txt' ## works like a charm!!!
gg=open(AAidx_file)
AAidx_Names=gg.readline().strip().split('\t')
AAidx_Dict={}
for ll in gg.readlines():
    ll=ll.strip().split('\t')
    AA=ll[0]
    tag=0
    vv=[]
    for xx in ll[1:]:
        vv.append(float(xx))
    if tag==1:
        continue
    AAidx_Dict[AA]=vv
    
sym_to_num= dict([(x, i) for i, x in enumerate(AAidx_Dict)])

AAidx_Dict["O"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
Nf=len(AAidx_Dict['C'])
pre = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unsqueeze(0)
post = torch.tensor([2.0, 2.0,  2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]).unsqueeze(0)

def AAindexEncoding(Seq):
    Ns=len(Seq)
    AAE=np.zeros([Ns, Nf])
    for kk in range(Ns):
        ss=Seq[kk]
        AAE[kk,]=AAidx_Dict[ss]
    #AAE=np.transpose(AAE.astype(np.float32)) # When shape is 15x18
    AAE=AAE.astype(np.float32) # When shape is 18x15
    AAE = torch.from_numpy(AAE)
    return AAE

def vocab_transform(seq):
    assert len(seq) <= MAX_LEN
    t = []
    for a in seq:
        t.append(a)
    k = len(t) // 2
    t = t[:k] + [PAD_IDX]*(MAX_LEN-len(t)) + t[k:]
    return t


def create_mask(inputs):
    rand = torch.rand(len(inputs))
    # where the random array is less than 0.15, we set true
    mask_arr = rand < 0.15
    return mask_arr



class AAsData(Dataset):
    def __init__(self, fn):
        d = pd.read_csv(fn, delimiter='\t', header=0)
        d = d.dropna()
        d = d[d['CDR3'].str.find('X') < 0]
        self.data = d
        self.weight  = np.ones(len(d))
        print('load data: %s, size=%d' % (fn.split('/')[-1], len(self.data)))
    


    def adjustWeights(self, model):
        
        _dataloader = DataLoader(self, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)
        
        cprobs=[]
        
        for i, (inputs, labels, weights) in enumerate(tqdm(_dataloader)):
          
            inputs = inputs.to(DEVICE)
            #print("input shape: ",inputs.shape)
            weights = weights.to(DEVICE)
            #print("labels shape: ",labels.shape)
            logits, softmaxOut = model(inputs)
            cancerProb = softmaxOut[:,1].detach().cpu().numpy()
            cprobs.append(cancerProb)
           
        cprobs=np.concatenate(cprobs)
        
        for i in range(len(cprobs)):
            if cprobs[i] < CUTOFF and self.data['Value'][i] == 1:
                self.weight[i] = self.weight[i] * DOWN_WEIGHT
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        d = self.data.iloc[item]
        return d['Value'], d['CDR3'], self.weight[item]



def collate_fn(data):
    labels, sequences, weights = [], [], []
    
    for label, sequence, weight in data:
        labels.append(label)
        weights.append(weight)
        result = AAindexEncoding(vocab_transform(sequence))
        temp = torch.cat((pre, torch.tensor(result), post))
        sequences.append(temp)
        
        
    weights = torch.FloatTensor(weights)
    labels = torch.IntTensor(labels)
    sequences = torch.stack(sequences)
    return sequences, labels, weights



class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:,:token_embedding.size(1), :])


'''
class CDRTransformer(nn.Module):
    def __init__(self, encoder_layer, num_layers, dropout: float = 0.1):
        super(CDRTransformer, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear0 = nn.Linear(15, EMB_SIZE)
        self.linear = nn.Linear(EMB_SIZE, 20)
        self.linear1 = nn.Linear(EMB_SIZE, 2)
        self.positional_encoding = PositionalEncoding(
            EMB_SIZE, dropout=dropout)

    def forward(self, inputs):
        inputs = self.linear0(inputs)
        src_enc = self.positional_encoding(inputs)
        out = self.transformer_encoder(src_enc)
        out = self.linear1(out)[:, 0]
        softOut = F.softmax(out, dim=-1)
        return out, softOut
'''

class CDRTransformer(nn.Module):
    def __init__(self, encoder_layer, num_layers, dropout: float = 0.1):
        super(CDRTransformer, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear0 = nn.Linear(15, EMB_SIZE)
        self.linear = nn.Linear(EMB_SIZE, 2)
        self.positional_encoding = PositionalEncoding(
            EMB_SIZE, dropout=dropout)

    def forward(self, inputs):
        inputs = self.linear0(inputs)
        src_enc = self.positional_encoding(inputs)
        out = self.transformer_encoder(src_enc)
        out = self.linear(out)[:, 0]
        softOut = F.softmax(out, dim=-1)
        return out, softOut



encoder_layer = nn.TransformerEncoderLayer(d_model=EMB_SIZE, nhead=8, batch_first=True)
transformer = CDRTransformer(encoder_layer, num_layers = 3)
model = transformer.to(DEVICE)
optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9) 






def softAccuracy(true_labels, pred_labels):
    #print("tru..",  true_labels.view(-1).float().shape)
    #print("pred.. ", pred_labels.float().shape)
    #print(pred_labels.float())
    accuracy = torch.sum(true_labels.view(-1).float() == pred_labels.float()) / BATCH_SIZE
    return accuracy.item()





def evaluate(model, roc= False):
    model.eval()
    losses = 0
    val_iter = AAsData('out/testing_data.csv')
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    acc = 0
    total = 0
    for i, (inputs, labels, weights) in enumerate(tqdm(val_dataloader)):
        total += 1
        inputs = inputs.to(torch.float32).to(DEVICE)
        labels = labels.to(torch.float32).to(DEVICE)
        logits, softlogits = model(inputs)
        acc += softAccuracy(labels, torch.argmax(logits, dim=1))
        
        labels = labels.type(torch.LongTensor).to(DEVICE)
        #logits = logits.float().to(DEVICE)

        loss = nn.CrossEntropyLoss()(logits, labels)
        losses += loss.item()
    out = acc/total
    print("Evaluation Accuracy: " , out)
    return (losses / len(val_dataloader)), out


def cancer_score(X):
    X = F.softmax(X, dim=1)
    length = len(X)
    X = torch.sum(X, axis =0)
    return X / length



def evaluate_cancer(model, file):
    model.eval()
    losses = 0

    val_iter = AAsData(file)
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)

    all_logits = torch.Tensor([]).to(DEVICE)
    all_labels = torch.Tensor([]).to(DEVICE)

    for i, (inputs, labels, weights) in enumerate(tqdm(val_dataloader)):
        inputs = inputs.to(torch.float32).to(DEVICE)
        #labels = labels.to(torch.float32).to(DEVICE)
        logits, softlogits = model(inputs)
        #all_labels = torch.cat((all_labels, labels))
        all_logits = torch.cat((all_logits, logits))
        

    out = F.softmax(logits, dim=1)[:,1]
    out = cancer_score(all_logits)
    #print(index_list)
    print("\nCancer Score: ", out, '\n')
    return out



if __name__ == '__main__':
    transformer.load_state_dict(torch.load("savedModels/classifier.pt", map_location=torch.device('cpu')))

    for i in range(0, 5):
        if i == 0:
            ds = 'Stromnes'
            dirt = "Cancer/"
            target = "Cancer/" + ds
        if i == 1:
            ds = 'Beausang'
            dirt = "Cancer/"
            target = "Cancer/" + ds
        if i == 2:
            ds = 'Robert'
            dirt = "Cancer/"
            target = "Cancer/" + ds
        if i == 3:
            ds = 'Mansfield'
            dirt = "Cancer/"
            target = "Cancer/" + ds
        if i == 4:
            ds = 'EmersonC'
            dirt = "Control/"
            target = "Control/" + ds
    
        file = open(dirt + "Cancer_score_" + ds +".csv", "w+")
    
        print(torch.cuda.is_available())
    
        plt.subplots_adjust(hspace=.5)
    
        file.write("File,Score\n")
        count = 1
        for filename in os.listdir(target):
            f = os.path.join(target, filename)
            out = evaluate_cancer(transformer, f)
            ScoreDict = out.detach().cpu().numpy()
            out = filename + "," + str(out[1].item()) + '\n'
            print("File Number ", count)
            count +=1
            '''
            # PLot Socres
            plt.subplot(2, 2, count)
            plt.title(f) 
            #plt.xlabel("Sequences") 
            #plt.ylabel("CancerScores") 
            plt.hist(ScoreDict, 50) 
            count+=1
            '''
    
            file.write(out)
    
        #plt.show()
    
        file.close()
    
    df = pd.read_csv('Cancer/Cancer_score_Beausang.csv', low_memory=False) # ES BC
    df1 = pd.read_csv('Cancer/Cancer_score_Mansfield.csv', low_memory=False) # Ovarian
    df2 = pd.read_csv('Cancer/Cancer_score_Robert.csv', low_memory=False) #  Melanoma
    df3 = pd.read_csv('Cancer/Cancer_score_Stromnes.csv', low_memory=False) # Pancreas
    df4 = pd.read_csv('Control/Cancer_score_EmersonC.csv', low_memory=False)
    
    
    breast = df['Score'].values
    ovarian = df1['Score'].values
    melanoma = df2['Score'].values
    pancreas = df3['Score'].values
    control = df4['Score'].values
    
    y_true = np.ones(len(breast))
    y_true2 = np.zeros(len(control))
    
    x = np.concatenate([breast, control])
    y = np.concatenate([y_true, y_true2])
    

    print("Beausang (BC): ", roc_auc_score(y, x))
    x = np.concatenate([ovarian, control])
    y_true = np.ones(len(ovarian))
    y = np.concatenate([y_true, y_true2])
    print("Emerson (Ovar): ", roc_auc_score(y, x))
    x = np.concatenate([melanoma, control])
    y_true = np.ones(len(melanoma))
    y = np.concatenate([y_true, y_true2])
    print("Stromnes (Mel)): ", roc_auc_score(y, x))
    x = np.concatenate([pancreas, control])
    y_true = np.ones(len(pancreas))
    y = np.concatenate([y_true, y_true2])
    print("Robert (Panc): ", roc_auc_score(y, x))








