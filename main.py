import numpy as np
import torch
import pickle as pkl
import random
from model import LightGCN
from utils import *
import pandas as pd
from parser import args
from tqdm import tqdm, trange
import torch.utils.data as data
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
import os

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
fix_seed(2024)

np.set_printoptions(suppress=True)

if args.gpu_id != '-1':
    device = 'cuda:' + args.gpu_id
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
d = args.d
l = args.gnn_layer
batch_user = args.batch
epoch_no = args.epoch
lambda_1 = args.lambda1
dropout = args.dropout
lr = args.lr

best_epoch = 0
best_valrecall = 0

dataset_name = args.dataset

if not os.path.exists('logs'):
    os.mkdir('logs')

if not os.path.exists('embs'):
    os.mkdir('embs')

############# load data
# load train set
path = 'dataset/' + args.dataset + '/'
f = open(path+'trnMat.pkl','rb')
train = pkl.load(f)
n_u, n_i=train.shape[0], train.shape[1]
train_csr = (train!=0).astype(np.float32)

# load test set
f = open(path+'tstMat.pkl','rb')
test_raw = pkl.load(f)
print('Data loaded.')

# load valid set
if os.path.exists(path+'valMat.pkl'):
    f = open(path+'valMat.pkl', 'rb')
    valid = pkl.load(f)
    test = test_raw
else:
    # if valid not exists, split from test set.
    print("Split valid and test")
    # split train and valid
    num_interactions = len(test_raw.data)
    test_indices = [i for i in range(num_interactions)]
    val_indices, test_indices = train_test_split(test_indices, test_size=0.5, random_state=42)
    valid = coo_matrix((test_raw.data[val_indices], (test_raw.row[val_indices], test_raw.col[val_indices])), shape=(n_u, n_i))
    test = coo_matrix((test_raw.data[test_indices], (test_raw.row[test_indices], test_raw.col[test_indices])), shape=(n_u, n_i))

valid_csr = (valid!=0).astype(np.float32)
test_csr = (test!=0).astype(np.float32)

# process valid set
val_labels = [[] for i in range(valid.shape[0])]
for i in range(len(valid.data)):
    row = valid.row[i]
    col = valid.col[i]
    val_labels[row].append(col)
print('Valid data processed.')

# process test set
test_labels = [[] for i in range(test.shape[0])]
for i in range(len(test.data)):
    row = test.row[i]
    col = test.col[i]
    test_labels[row].append(col)
print('Test data processed.')

print('user_num:',train.shape[0],'item_num:',train.shape[1])

############## data preprocessing
# get item popularity
popularity, item_grp, i_num_classes, item_grp_num, head = get_item_attr(train)

print(f'{len(head)} head items among {train.shape[1]} items')

# normalizing the adj matrix
rowD = np.array(train.sum(1)).squeeze()
colD = np.array(train.sum(0)).squeeze()
for i in range(len(train.data)):
    train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)

adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
adj_norm = adj_norm.coalesce().cuda(torch.device(device))

print('Adj matrix normalized.')

degree = popularity.cuda(torch.device(device))

model = LightGCN(n_u, n_i, d, train_csr, adj_norm, l, lambda_1, dropout, batch_user, device, args, degree)
model.cuda(torch.device(device))
optimizer = torch.optim.Adam(model.parameters(),weight_decay=0,lr=lr)

# construct data loader
train_data = TrnData(train)
train_loader = data.DataLoader(train_data, batch_size=args.inter_batch, shuffle=True, num_workers=0)
#early_stopping = EarlyStopping(patience=args.patience, verbose=True)

for epoch in range(1, epoch_no+1):
    epoch_loss = 0
    epoch_loss_r = 0
    train_loader.dataset.neg_sampling()
    
    for i, batch in enumerate(tqdm(train_loader)):
        uids, pos, neg = batch
        uids = uids.long().cuda(torch.device(device))
        pos = pos.long().cuda(torch.device(device))
        neg = neg.long().cuda(torch.device(device))
        iids = torch.concat([pos, neg], dim=0)

        # get loss
        optimizer.zero_grad()
        loss, loss_r = model(uids, pos, neg)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.cpu().item()
        epoch_loss_r += loss_r.cpu().item()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    batch_no = len(train_loader)
    epoch_loss = epoch_loss/batch_no
    epoch_loss_r = epoch_loss_r/batch_no
    
    print('Epoch:',epoch,'Loss:',epoch_loss,'Loss_r:',epoch_loss_r)

    if epoch % 5 == 0:  # validate every 3 epochs
        val_uids = np.array([i for i in range(n_u)])
        batch_no = int(np.ceil(len(val_uids)/batch_user))

        all_user_num = 0
        all_hitrate_20 = 0
        all_recall_20 = 0
        all_ndcg_20 = 0
        all_hitrate_40 = 0
        all_recall_40 = 0
        all_ndcg_40 = 0
        all_c_ratio = np.zeros(5)

        for batch in tqdm(range(batch_no)):
            start = batch*batch_user
            end = min((batch+1)*batch_user,len(val_uids))

            val_uids_input = torch.LongTensor(val_uids[start:end]).cuda(torch.device(device))
            predictions, _ = model(val_uids_input, None, None, head, csr=test_csr, test=True)
            predictions = np.array(predictions.cpu())

            #top@20
            user_num, _, _, hitrate_20, recall_20, ndcg_20 = metrics(val_uids[start:end],predictions,20,val_labels)
            #top@40
            user_num, _, _, hitrate_40, recall_40, ndcg_40 = metrics(val_uids[start:end],predictions,40,val_labels)
                        
            # C_ratio
            top_K_items_grp = item_grp[predictions[:,:20]] 
            c_ratio = C_Ratio(top_K_items_grp)

            all_user_num += user_num
            all_hitrate_20+=hitrate_20
            all_recall_20+=recall_20
            all_ndcg_20+=ndcg_20
            all_hitrate_40+=hitrate_40
            all_recall_40+=recall_40
            all_ndcg_40+=ndcg_40
            all_c_ratio+=c_ratio

        print('-------------------------------------------')
        print('Validation of epoch', epoch)
        print('Recall@20:', all_recall_20/all_user_num)
        print('Ndcg@20:', all_ndcg_20/all_user_num)
        print('Recall@40:', all_recall_40/all_user_num)
        print('Ndcg@40:', all_ndcg_40/all_user_num)
        print('-------------------------------------------')
        print('C_ratio:', all_c_ratio/n_u) #'ConfRate:',all_conf_rate/batch_no)
        
        with open(os.path.join('logs', f'{dataset_name}.txt'), 'a') as f:
            f.write(f'Valid-epoch-{epoch}\t')
            f.write(f'recall@20 {all_recall_20/all_user_num}\t')
            f.write(f'ndcg@20 {all_ndcg_20/all_user_num}\t')
            f.write(f'recall@40 {all_recall_40/all_user_num}\t')
            f.write(f'ndcg@40 {all_ndcg_40/all_user_num}\t')
            f.write(','.join([str(all_c_ratio[i]/n_u) for i in range(5)]) + '\n')

        
        if args.save_emb:
            E_u_emb = sum(model.E_u_list).detach().cpu()
            E_i_emb = sum(model.E_i_list).detach().cpu()
            
            with open(os.path.join('embs', dataset_name + f'_epoch{epoch}.pkl'), 'wb') as f:
                pkl.dump([E_u_emb, E_i_emb], f)
