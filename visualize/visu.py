import numpy as np

attr_auc = []
attr_nll = []
edge_auc = []
edge_nll = []




p = "/Users/hiro_m/Desktop/study/coma/experiment_data"
attr_auc_path = p+"/proposed_attr_auc.npy"
attr_nll_path = p+"/proposed_attr_nll.npy"
edge_auc_path = p+"/proposed_edge_auc.npy"
edge_nll_path = p+"/proposed_edge_nll.npy"


attr_auc.append(np.load(attr_auc_path))
attr_nll.append(np.load(attr_nll_path))
edge_auc.append(np.load(edge_auc_path))
edge_nll.append(np.load(edge_nll_path))



attr_mean = np.mean(attr_auc,axis=1)
attr_max = np.max(attr_auc,axis=1)
attr_min = np.min(attr_auc,axis=1)
edge_mean = np.mean(edge_auc,axis=1)
edge_max = np.max(edge_auc,axis=1)
edge_min = np.min(edge_auc,axis=1)

print(attr_mean)
print(attr_max)
print(attr_min)
print(edge_mean)
print(edge_max)
print(edge_min)