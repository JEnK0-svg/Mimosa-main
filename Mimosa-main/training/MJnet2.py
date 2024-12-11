import pandas as pd
import numpy as np
import sys,os, re
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm

from sklearn.metrics import recall_score,f1_score
from sklearn.metrics import accuracy_score,average_precision_score, precision_score
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils import *

torch.manual_seed(417)
torch.set_num_threads(20)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class myDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        self.sample = self.data[index]
        self.mirna, self.mrna, self.label = self.sample
        self.reverse_mrna = reverse_seq(self.mrna)
        self.mirna = self.mirna + 'X' * (26 - len(self.mirna))
        #数据集编码
        # onehot_m = get_onehot_embedding(self.reverse_mrna)
        # onehot_mi = get_onehot_embedding(self.mirna)
        
        pairing_m, pairing_mi = get_interaction_map(self.mirna,self.reverse_mrna)

        ND_m = to_ND(self.reverse_mrna)
        ND_mi = to_ND(self.mirna)

        C2_m = to_C2(self.reverse_mrna)
        C2_mi = to_C2(self.mirna)

        NCP_m = to_NCP(self.reverse_mrna)
        NCP_mi = to_NCP(self.mirna)

        # onehot_m  = torch.tensor(onehot_m, dtype=torch.float32).to(device)
        # onehot_mi = torch.tensor(onehot_mi, dtype=torch.float32).to(device)
        
        C2_m = torch.tensor(C2_m, dtype=torch.float32).to(device)
        C2_mi = torch.tensor(C2_mi, dtype=torch.float32).to(device)

        NCP_m = torch.tensor(NCP_m, dtype=torch.float32).to(device)
        NCP_mi = torch.tensor(NCP_mi, dtype=torch.float32).to(device)

        ND_m = torch.tensor(ND_m, dtype=torch.float32).to(device)
        ND_mi = torch.tensor(ND_mi, dtype=torch.float32).to(device)

        pairing_m = torch.tensor(pairing_m, dtype=torch.float32).to(device)
        pairing_mi = torch.tensor(pairing_mi, dtype=torch.float32).to(device)

        label = torch.tensor(self.label, dtype=torch.float32).to(device)
        
        return {
                # 'onehot_m': onehot_m,
                # 'onehot_mi': onehot_mi,
                'C2_m' : C2_m,
                'C2_mi' : C2_mi,
                'NCP_m' : NCP_m,
                'NCP_mi' : NCP_mi,
                'ND_m' : ND_m,
                'ND_mi' : ND_mi,
                'pairing_m': pairing_m,
                'pairing_mi': pairing_mi,
                'label': label,
               }

class MJnet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout, output_size):
        super(MJnet, self).__init__()
        
        # 使用双向GRU（输入维度加上pairing特征，共5维）
        self.gru_m = nn.GRU(input_size, hidden_size, num_layers,  dropout=dropout, bidirectional=True, batch_first=True)
        self.gru_mi = nn.GRU(input_size, hidden_size, num_layers,  dropout=dropout, bidirectional=True, batch_first=True)

        # Cross attention mechanism (hidden_size * 2 due to bidirectional GRU)
        self.cross_attention = nn.MultiheadAttention(hidden_size * 2, num_heads, dropout=dropout)

        # self.batch_norm1 = nn.BatchNorm1d(40)
        # Fully connected layers for final classification
        self.fc1 = nn.Linear(40, 16)
        self.dropout = nn.Dropout(0.1)
        self.batch_norm1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, output_size)

    def forward(self, C2_m, C2_mi, NCP_m, NCP_mi, ND_m, ND_mi, pairing_m, pairing_mi):
        # 将pairing特征与one-hot编码拼接，形成5维输入 (4维one-hot + 1维pairing)
        m_input = torch.cat((C2_m, NCP_m, ND_m.unsqueeze(-1), pairing_m.unsqueeze(-1)), dim=-1)  # (batch_size, seq_len, 5)
        mi_input = torch.cat((C2_mi, NCP_mi, ND_mi.unsqueeze(-1), pairing_mi.unsqueeze(-1)), dim=-1)

        # Bi-directional GRU Encoder
        m_emb, _ = self.gru_m(m_input)  # Output shape: (batch_size, seq_len, 2 * hidden_size)
        mi_emb, _ = self.gru_mi(mi_input)

        # Permute for multihead attention
        m_emb = m_emb.permute(1, 0, 2)  # (seq_len, batch_size, 2 * hidden_size)
        mi_emb = mi_emb.permute(1, 0, 2)

        # Cross attention mechanism
        cross_attend, _ = self.cross_attention(m_emb, mi_emb, mi_emb)

        # Mean pooling after attention, followed by fully connected layers
        output = cross_attend.permute(1, 0, 2).mean(dim=2)
        # output = self.batch_norm1(output)
        # Classification
        output = self.fc1(output)
        output = self.dropout(output)
        output = torch.relu(output)
        output = self.batch_norm1(output)
        output = self.fc2(output)

        return output

# define the model architecture of Mimosa
class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout, output_size):
        super(Transformer, self).__init__()

        self.embedding_m = nn.Embedding(input_size, hidden_size)
        self.position_encoding_m = nn.Parameter(torch.zeros(1, 100, hidden_size))
        self.interaction_embedding_m = nn.Embedding(3, hidden_size)
        nn.init.normal_(self.position_encoding_m, mean=0, std=0.1)


        self.embedding_mi = nn.Embedding(input_size, hidden_size)
        self.position_encoding_mi = nn.Parameter(torch.zeros(1, 100, hidden_size))
        self.interaction_embedding_mi = nn.Embedding(3, hidden_size)
        nn.init.normal_(self.position_encoding_mi, mean=0, std=0.1)


        encoder_layers_m = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout)
        self.encoder_m = nn.TransformerEncoder(encoder_layers_m, num_layers)

        encoder_layers_mi = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout)
        self.encoder_mi = nn.TransformerEncoder(encoder_layers_mi, num_layers)

        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads)

        self.fc1 = nn.Linear(40, 12)
        self.fc2 = nn.Linear(12, output_size)


    def forward(self, emb_m, emb_mi, pairing_m, pairing_mi):


        m_emb = self.embedding_m(emb_m) + self.position_encoding_m[:, :emb_m.size(1), :] + self.interaction_embedding_m(pairing_m)
        mi_emb = self.embedding_mi(emb_mi)  + self.position_encoding_mi[:,:emb_mi.size(1),:] + self.interaction_embedding_mi(pairing_mi)

        m_emb = m_emb.permute(1, 0, 2)
        mi_emb = mi_emb.permute(1, 0, 2)

        encoder_output_m = self.encoder_m(m_emb)
        encoder_output_mi = self.encoder_mi(mi_emb)

        cross_attend, _ = self.cross_attention(encoder_output_m, encoder_output_mi, encoder_output_mi)
        output = cross_attend.permute(1,0,2).mean(dim=2)


        output = self.fc1(output)
        output = torch.relu(output)
        output = self.fc2(output)
        out = torch.softmax(output, dim=1)
        return out


def Deep_train(model, dataloader, optimizer, criterion):

    model.train()
    counter = 0
    train_loss = 0.0

    for i, data in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        counter += 1

        features1, features2, features3, features4, target = data['C2_m'], data['C2_mi'], data['NCP_m'], data['NCP_mi'], data['label']
        features5, features6, features7, features8= data['ND_m'], data['ND_mi'], data['pairing_m'], data['pairing_mi']

        # 将特征和标签移动到与模型相同的设备
        features1 = features1.to(device)
        features2 = features2.to(device)
        features3 = features3.to(device)
        features4 = features4.to(device)
        features5 = features5.to(device)
        features6 = features6.to(device)
        features7 = features7.to(device)
        features8 = features8.to(device)
        target = target.to(device)

        target = target.unsqueeze(1)
        outputs = model(features1, features2, features3, features4, features5, features6, features7, features8).to(device)
        loss = criterion(outputs,target)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_avg_loss = train_loss/counter
    return train_avg_loss


def Deep_validate(model, dataloader, criterion):
    model.eval()
    counter = 0
    val_loss = 0.0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc="Validating", leave=False)):
            counter += 1
            features1, features2, features3, features4, target = data['C2_m'], data['C2_mi'], data['NCP_m'], data['NCP_mi'], data['label']
            features5, features6, features7, features8= data['ND_m'], data['ND_mi'], data['pairing_m'], data['pairing_mi']

            # 将特征和标签移动到与模型相同的设备
            features1 = features1.to(device)
            features2 = features2.to(device)
            features3 = features3.to(device)
            features4 = features4.to(device)
            features5 = features5.to(device)
            features6 = features6.to(device)
            features7 = features7.to(device)
            features8 = features8.to(device)
            target = target.to(device)

            target = target.unsqueeze(1)
            outputs = model(features1, features2, features3, features4, features5, features6, features7, features8).to(device)
            loss = criterion(outputs,target)

            val_loss += loss.item()
            predictions = []
            outputs = outputs.cpu().numpy()
            target = target.cpu().numpy()
            # print(type(outputs))
            for i in outputs:
                if i > 0.5:
                    predictions.append(1)
                else:
                    predictions.append(0)
            all_predictions.extend(predictions)
            all_targets.extend([i for i in target])

        val_total_loss = val_loss/counter

        acc = accuracy_score(all_targets, all_predictions)
        pre = precision_score(all_targets, all_predictions)
        recall = recall_score(all_targets,all_predictions)
        spec = specificity_score(all_targets,all_predictions)
        f1 = f1_score(all_targets,all_predictions)
        npv = NPV(all_targets, all_predictions)


        print('acc',acc)
        print('pre',pre)
        print('recall',recall)
        print('specificity',spec)
        print('f1',f1)
        print('npv',npv)
        

        return val_total_loss


def perform_train(filepath):
    # train positive: 26995, train negative: 27469, val positive: 2193, val negative: 2136
    batchsize = 128
    learningrate = 4e-5
    epochs = 50
    train, val = read_data(filepath)

    train_dataset = myDataset(train)
    val_dataset = myDataset(val)
    train_loader = DataLoader(train_dataset,batch_size=batchsize,shuffle=True) #collate_fn=my_collate_fn,
    val_loader = DataLoader(val_dataset, batch_size=batchsize,shuffle=True) #,collate_fn=my_collate_fn
    

    # model = Transformer(input_size=5, hidden_size=64, num_layers=16, num_heads=8, dropout=0.1, output_size=2).to(device)
    model = MJnet(input_size = 7, hidden_size = 128, num_layers = 2, num_heads = 8, dropout = 0.3, output_size=1).to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=learningrate,weight_decay=1e-5) #1、1e-5 2、1e-4

    best_val_loss = 1
    train_loss = []
    valid_loss = []
    for epoch in range(epochs):
        print(f'Epoch {epoch+1} of {epochs}')
        train_epoch_loss = Deep_train(model, train_loader,optimizer, criterion)
        valid_epoch_loss = Deep_validate(model,val_loader,criterion)

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        print('Train Loss:', train_epoch_loss)
        print('Val Loss:',valid_epoch_loss)
        if valid_epoch_loss < best_val_loss:
            best_val_loss = valid_epoch_loss
            torch.save(model,'model_concate_{}.pth'.format(epoch))
        if epoch == epochs -1:
            torch.save(model,'model_final.pth')





def get_cts(rmrna, stepsize):
    '''segment full-length mRNAS into 40-nt segments using a sliding window with predefined stepsize'''
    kmers = []

    if len(rmrna) >= 40:
        for i in range(0, len(rmrna),stepsize):
            if i + 40 <= len(rmrna):
                cut = rmrna[i:i + 40]
                kmers.append(cut)

        return kmers
    else:
        pad_rmrna = rmrna + 'X' * (40 - len(rmrna))
        kmers.append(pad_rmrna)
        return kmers




def kmers_predict(kmers,mirna,model):

    mirna = mirna + 'X'*(26-len(mirna))
    fea_C2_m = []
    fea_C2_mi = []
    fea_NCP_m = []
    fea_NCP_mi = []
    fea_ND_m = []
    fea_ND_mi = []
    fea_pairing_m = []
    fea_pairing_mi = []
    

    if len(kmers) == 0:
        return 0
    else:
        for i in kmers:
            # onehot_m = get_onehot_embedding(i)
            # onehot_mi = get_onehot_embedding(mirna)
            C2_m = to_C2(i)
            C2_mi = to_C2(mirna)

            NCP_m = to_NCP(i)
            NCP_mi = to_NCP(mirna)

            ND_m = to_ND(i)
            ND_mi = to_ND(mirna)
            if 'X' in i:
                pairing_m, pairing_mi = get_interaction_map_for_test_short(mirna, i)
            else:
                pairing_m, pairing_mi = get_interaction_map_for_test(mirna, i)
            
            fea_C2_m.append(C2_m)
            fea_C2_mi.append(C2_mi)
            fea_NCP_m.append(NCP_m)
            fea_NCP_mi.append(NCP_mi)
            # fea_m.append(onehot_m)
            # fea_mi.append(onehot_mi)
            fea_ND_m.append(ND_m)
            fea_ND_mi.append(ND_mi)
            fea_pairing_m.append(pairing_m)
            fea_pairing_mi.append(pairing_mi)
            

        fea_C2_m = torch.tensor(np.array(fea_C2_m), dtype=torch.float32).to(device)
        fea_C2_mi = torch.tensor(np.array(fea_C2_mi), dtype=torch.float32).to(device)
        fea_NCP_m = torch.tensor(np.array(fea_NCP_m), dtype=torch.float32).to(device)
        fea_NCP_mi = torch.tensor(np.array(fea_NCP_mi), dtype=torch.float32).to(device)
        fea_ND_m = torch.tensor(np.array(fea_ND_m), dtype=torch.float32).to(device)
        fea_ND_mi = torch.tensor(np.array(fea_ND_mi), dtype=torch.float32).to(device)
        fea_pairing_m = torch.tensor(np.array(fea_pairing_m), dtype=torch.float32).to(device)
        fea_pairing_mi = torch.tensor(np.array(fea_pairing_mi), dtype=torch.float32).to(device)
        
        model = model.to(device)
        pros = model(fea_C2_m, fea_C2_mi, fea_NCP_m, fea_NCP_mi, fea_ND_m, fea_ND_mi, fea_pairing_m, fea_pairing_mi ).detach().cpu().numpy().tolist()
        pppp = decision_for_whole(pros)

        return pppp





def perform_test(pathfile, stepsize, model_type):

    test = read_test(pathfile)
    y_true = []
    y_pred = []

    model = torch.load(model_type)
    model = model.to(device)
    print('个数',len(test))

    for index in range(len(test)): #range(len(test))
        fasta = test[index]


        mirna = fasta[0].upper().replace('T', 'U')

        mrna = fasta[1].upper().replace('T', 'U')
        reverse_mrna = reverse_seq(mrna)
        y_true.append(fasta[2])

 
        kmers = get_cts(reverse_mrna,stepsize)

        if kmers is None:
            pre = 0
            y_pred.append(pre)

        else:

            pre = kmers_predict(kmers, mirna, model)
            y_pred.append(pre)

    print(y_true)
    print(y_pred)
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    spec = specificity_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = NPV(y_true, y_pred)

    print('acc', acc)
    print('PPV', pre)
    print('recall', recall)
    print('specificity', spec)
    print('f1', f1)
    print('NPV', auc)



# path = '/your/path/to/Mimosa'
# train_dataset_path = path + '/Data/miRAW_Train_Validation.txt'
# test_dataset_path = path + '/Data/miRAW_Test0.txt'
# perform_train(train_dataset_path)
# perform_test(test_dataset_path, stepsize=1)