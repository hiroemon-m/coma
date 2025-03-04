"""
T=5の時点でどのくらいの精度が出るか?
"""



# Third Party Library

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix


# First Party Library

from init_real_data import init_real_data





#@profile
def execute_data(persona_num,data_name,data_type):
    ##デバッグ用
    LEARNED_TIME = 4
    GENERATE_TIME = 5
    TOTAL_TIME = 10
    load_data = init_real_data(data_name)

    edge_sparse = load_data.adj[LEARNED_TIME]
    feat_sparse = load_data.feature[LEARNED_TIME]
    edge_calc_log = np.zeros((1, 5))
    edge_calc_nll_log = np.zeros((1, 5))
    attr_calc_log = np.zeros((1, 5))
    attr_calc_nll_log = np.zeros((1, 5))
    

    for test_time in range(TOTAL_TIME - GENERATE_TIME):

        #属性値AUC
        feat_prob = (feat_sparse.to_dense()>0).float()
        feat_prob = torch.ravel(feat_prob.to_dense()).to("cpu")
        feat_prob = feat_prob.numpy()
        attr_predict_probs = np.concatenate([feat_prob], 0)
    
        if isinstance(load_data.feature[GENERATE_TIME + test_time],csc_matrix):
            dense_array = np.array(load_data.feature[GENERATE_TIME + test_time].todense())
        else:
            dense_array = np.array(load_data.feature[GENERATE_TIME + test_time].to_dense())

        attr_tensor = torch.from_numpy(dense_array)

        detach_attr = (torch.ravel(attr_tensor).detach())
        detach_attr[detach_attr > 0] = 1.0
        pos_attr = detach_attr.numpy()
        attr_test = np.concatenate([pos_attr], 0)
            
        # NLLを計算
        criterion = nn.CrossEntropyLoss()
        error_attr = criterion(torch.from_numpy(attr_predict_probs),torch.from_numpy(attr_test))  
        attr_auc = roc_auc_score(attr_test, attr_predict_probs) 
       
        print("attr auc, t={}:".format(test_time), attr_auc)
        print("attr nll, t={}:".format(test_time), error_attr.item())
        attr_calc_log[0,test_time] = attr_auc
        attr_calc_nll_log[0,test_time] = error_attr.item()
        
    

        #エッジAUC
        edge_prob = (edge_sparse.to_dense()>0).float()
        edge_prob = torch.ravel(edge_prob.to_dense()).to("cpu")
        edge_prob = edge_prob.numpy()
        edge_predict_probs = np.concatenate([edge_prob], 0)
        
        #テストデータ
        if isinstance(load_data.adj[GENERATE_TIME + test_time],csc_matrix):   
            dense_array = np.array(load_data.adj[GENERATE_TIME + test_time].todense())
        else:
            dense_array = np.array(load_data.adj[GENERATE_TIME + test_time].to_dense())

        edge_tensor = torch.from_numpy(dense_array)
        detach_edge = (
            torch.ravel(edge_tensor)
            .detach()
            .to("cpu")
        )

        pos_edge = detach_edge.numpy()
        edge_test = np.concatenate([pos_edge], 0)



        criterion = nn.CrossEntropyLoss()
        error_edge = criterion(
            torch.from_numpy(edge_predict_probs),
            torch.from_numpy(edge_test),
        )


        edge_auc = roc_auc_score(edge_test, edge_predict_probs)  
        print("edge auc, t={}:".format(test_time), edge_auc)


        edge_calc_log[0,test_time] = edge_auc
        edge_calc_nll_log[0,test_time] = error_edge.item()
        

        path_save = "experiment_data/original/{}".format(data_name)

        np.save(path_save+"/proposed_edge_auc", edge_calc_log)
        np.save(path_save+"/proposed_edge_nll", edge_calc_nll_log)
        np.save(path_save+"/proposed_attr_auc", attr_calc_log)
        np.save(path_save+"/proposed_attr_nll", attr_calc_nll_log)






if __name__ == "__main__":
    #[5,8,12,16,24,32,64,128]
    #[4,8,12,16]
    i = 5
    for data_name in ["Twitter","NIPS","DBLP"]:
        execute_data(i,data_name,"complete")
