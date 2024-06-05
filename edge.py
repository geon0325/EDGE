import os
import torch
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm, trange
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
from utils import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', default='fair_gowalla', type=str)
parser.add_argument('--gpu', default=0, type=int)

parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--reg', default=0.00001, type=float)

parser.add_argument('--alpha', default=0.0, type=float)
parser.add_argument('--beta', default=0.0, type=float)
parser.add_argument('--tau', default=0.0, type=float)
parser.add_argument('--lmbda', default=0.0, type=float)

args = parser.parse_args()

if args.lmbda == 0 and args.tau > 0:
    exit(0)
if args.lmbda > 0 and args.tau == 0:
    exit(0)

if args.gpu != -1:
    device = torch.device(f'cuda:{args.gpu}')
else:
    device = torch.device('cpu')
print(device)

### Dataset Loading ###
path = 'dataset/' + args.dataset + '/'

# Training set
f = open(path + 'trnMat.pkl','rb')
train = pkl.load(f)
n_u, n_i = train.shape[0], train.shape[1]
train_csr = (train!=0).astype(np.float32)

# Test set
f = open(path + 'tstMat.pkl','rb')
test_raw = pkl.load(f)

# split train and valid
num_interactions = len(test_raw.data)
test_indices = [i for i in range(num_interactions)]
val_indices, test_indices = train_test_split(test_indices, test_size=0.5, random_state=42)
valid = coo_matrix((test_raw.data[val_indices], (test_raw.row[val_indices], test_raw.col[val_indices])), shape=(n_u, n_i))
test = coo_matrix((test_raw.data[test_indices], (test_raw.row[test_indices], test_raw.col[test_indices])), shape=(n_u, n_i))

valid_csr = (valid!=0).astype(np.float32)
test_csr = (test!=0).astype(np.float32)
print('Data loaded.')

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

popularity, item_grp, i_num_classes, item_grp_num, head = get_item_attr(train)
print(f'{len(head)} head items among {train.shape[1]} items')

popularity = popularity.to(device)

def evaluate(user_emb, item_emb, eval_type='val', batch_size=2048):
    uids = np.array([i for i in range(n_u)])
    batch_no = int(np.ceil(len(uids) / batch_size))

    all_user_num, all_recall_20, all_ndcg_20, all_recall_40, all_ndcg_40 = 0, 0, 0, 0, 0

    for batch in trange(batch_no):
        start = batch * batch_size
        end = min((batch + 1) * batch_size, len(uids))

        batch_users = torch.LongTensor(uids[start:end])

        preds = user_emb[batch_users] @ item_emb.T

        mask = torch.Tensor(train_csr[batch_users].toarray()).to(device)
        preds = preds * (1 - mask) - 1e8 * mask

        if eval_type == 'val':
            mask = torch.Tensor(test_csr[batch_users].toarray()).to(device)
        else:
            mask = torch.Tensor(valid_csr[batch_users].toarray()).to(device)
        preds = preds * (1 - mask) - 1e8 * mask

        predictions = np.array(preds.argsort(descending=True).detach().cpu())
        
        if eval_type == 'val':
            labels = val_labels
        else:
            labels = test_labels

        user_num, _, _, _, recall_20, ndcg_20 = metrics(batch_users, predictions, 20, labels)
        user_num, _, _, _, recall_40, ndcg_40 = metrics(batch_users, predictions, 40, labels)

        all_user_num += user_num
        all_recall_20 += recall_20
        all_ndcg_20 += ndcg_20
        all_recall_40 += recall_40
        all_ndcg_40 += ndcg_40

    recall_20 = all_recall_20 / all_user_num
    ndcg_20 = all_ndcg_20 / all_user_num
    recall_40 = all_recall_40 / all_user_num
    ndcg_40 = all_ndcg_40 / all_user_num

    return recall_20, ndcg_20, recall_40, ndcg_40

### Embedding List ###
with open(os.path.join('embs', args.dataset + '.pkl'), 'rb') as f:
    user_emb, item_emb = pkl.load(f)

user_emb = user_emb.to(device)
item_emb = item_emb.to(device)

norm_orig = torch.norm(item_emb, dim=1) + 1e-12
item_emb_unit = item_emb / norm_orig[:,None]

if args.lmbda > 0:
    sim = torch.mm(item_emb_unit, item_emb_unit[head].t())
    sim = torch.softmax(sim / max(args.tau, 0.01), dim=1)

    item_emb_att = torch.mm(sim, item_emb_unit[head])
    item_emb_att = item_emb_att / (torch.norm(item_emb_att, dim=1) + 1e-12)[:,None]

    item_emb_adj = item_emb_unit + args.lmbda * item_emb_att
    item_emb_adj = item_emb_adj / (torch.norm(item_emb_adj, dim=1) + 1e-12)[:,None]

else:
    item_emb_adj = item_emb_unit
    
norm = ((popularity + 1) ** args.beta) * (norm_orig ** (1 - args.alpha))
item_emb = item_emb_adj * norm[:,None]

r20t, n20t, r40t, n40t = evaluate(user_emb, item_emb, 'test')
print(f'Recall@20: {r20t}')
print(f'NDCG@20: {n20t}')
print(f'Recall@40: {r40t}')
print(f'NDCG@40: {n40t}')


