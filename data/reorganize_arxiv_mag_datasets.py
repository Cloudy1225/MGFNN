import pickle
import numpy as np


# Reorganize ArXiv
with open(f'./arxiv/others.pkl', 'rb') as f:
    data = pickle.load(f)

feat_chunk_list = []
for i in range(2):
    with open(f'./arxiv/feats{i}.pkl', 'rb') as f:
        feat_chunk_list.append(pickle.load(f))
data['feats'] = np.concatenate(feat_chunk_list)

with open('./arxiv.pkl', 'wb') as f:
    pickle.dump(data, f)
print('ArXiv done:)')


# Reorganize MAG
with open(f'./mag/others.pkl', 'rb') as f:
    data = pickle.load(f)

feat_chunk_list = []
for i in range(6):
    with open(f'./mag/feats{i}.pkl', 'rb') as f:
        feat_chunk_list.append(pickle.load(f))
data['feats'] = np.concatenate(feat_chunk_list)

with open('./mag/adj_pap_row.pkl', 'rb') as f:
    adj_pap_row = pickle.load(f)
with open('./mag/adj_pap_col.pkl', 'rb') as f:
    adj_pap_col = pickle.load(f)
data['adjs']['PAP'] = (adj_pap_row, adj_pap_col)

with open('./mag.pkl', 'wb') as f:
    pickle.dump(data, f)
print('MAG done:)')
