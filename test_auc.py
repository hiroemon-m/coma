# Standard Library
import gc
import joblib
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

import pandas as pd
import config

# First Party Library
from env_sparse import Env
from init_real_data import init_real_data
from actor_review import Actor


torch.autograd.set_detect_anomaly(True)
device = config.select_device


class PPO:
    def __init__(self, obs,agent_num,input_size, action_dim, lr, gamma,T,e,r,w,rik,temperature,story_count,data_set):
        self.agent_num = agent_num
        self.action_dim = action_dim
        self.input_size = input_size
        self.gamma = gamma
        self.persona = rik
        self.story_count = story_count 
        self.obs = obs
        self.alpha = self.obs.alpha
        self.beta = self.obs.beta
        self.gamma = self.obs.gamma
        self.count = 0
        self.actor = Actor(T,e,r,w,rik,self.agent_num,temperature)
        self.new_actor = Actor(T,e,r,w,rik,self.agent_num,temperature)
        self.update_actor = Actor(T,e,r,w,rik,self.agent_num,temperature)
        #adamにモデルを登録
        #self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr) 
        #self.new_actor_optimizer = torch.optim.Adam(self.new_actor.parameters(), lr=lr)
        self.actor_optimizer = torch.optim.SGD(self.actor.parameters(), lr=lr) 
        self.new_actor_optimizer = torch.optim.SGD(self.new_actor.parameters(), lr=lr)

    def sparse_clip_grad_norm_(self,parameters, max_norm):
        for param in parameters:
            if param.grad is not None and param.grad.is_sparse:
                # スパーステンソルの非ゼロ値を取得
                grad_values = param.grad.coalesce().values()
                norm = grad_values.norm(2)  # L2ノルムを計算
                if norm > max_norm:
                    scale = max_norm / (norm + 1e-6)
                    grad_values.mul_(scale)  # 非ゼロ値をスケール
            elif param.grad is not None:
                # 通常の密テンソルの処理
                torch.nn.utils.clip_grad_norm_([param], max_norm)


    #@profile
    def get_actions(self, edges, two_hop_neighbar, feat,time,action_dim,feat_size):
        prob,edge,feat = self.actor.get_action(feat,edges,two_hop_neighbar,time,action_dim,feat_size)
        return prob,edge,feat
    
    

        



#@profile
def execute_data(persona_num,data_name,data_type):
    ##デバッグ用
    torch.autograd.set_detect_anomaly(True)
    #alpha, betaの読み込み
    if data_name == "NIPS":
        action_dim = 32
        feat_size = 2411

    if data_name == "DBLP":
        action_dim = 500
        feat_size = 3854

    if data_name == "Twitter":
        action_dim = 234305
        feat_size = 5372

    if data_name == "Reddit":
        action_dim = 8077
        feat_size = 300

    LEARNED_TIME = 4
    GENERATE_TIME = 5
    TOTAL_TIME = 10
    story_count = 5
    load_data = init_real_data(data_name)

    edge_sparse = []
    feat_sparse = []
    for i in range(LEARNED_TIME+1):
        edge_sparse.append(load_data.adj[i])
        feat_sparse.append(load_data.feature[i])
    

    agent_num = len(load_data.adj[LEARNED_TIME])
    input_size = len(load_data.feature[LEARNED_TIME][1])

    path_n = "optimize/{}/{}/".format(data_type,data_name)
    path = "experiment_data/attmpt/1/train_persona_ration.npy"
    persona_ration = np.load(path)
    persona_ration = persona_ration.astype("float32")
    #time,node,persona
    persona_ration = torch.from_numpy(persona_ration).to(device)

    path = path_n+"persona={}/means.npy".format(int(persona_num))
    means = np.load(path)
    means = means.astype("float32")
    means = torch.from_numpy(means).to(device)

    path = "experiment_data/attmpt/1/train_paramerter.npy"
    paramerter = np.load(path)
    paramerter = paramerter.astype("float32")

    T = torch.from_numpy(paramerter[:5])
    e = torch.from_numpy(paramerter[5:10])
    r = torch.from_numpy(paramerter[10:15])
    w = torch.from_numpy(paramerter[15:20])

    print(T,e,r,w)


    alpha = means[:,0]
    beta = means[:,1]
    gamma = means[:,2]
    


    #パラメータ
    if data_name == "NIPS":
        mu = 0.194
        lr = 1.563e-06
        lr = 0.001
        temperature = 0.01
       
    elif data_name == "DBLP":
        mu = 0.0229
        #lr = 0.000952
        lr = 0.001
        temperature = 0.01

  
    else:
        mu = 0.01
        lr = 0.01
        temperature = 0.01
     

    #環境の設定
    obs = Env(
        agent_num=agent_num,
        edge = edge_sparse[LEARNED_TIME],
        feature = feat_sparse[LEARNED_TIME],
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        persona=persona_ration
        )


    agents = PPO(obs,agent_num, input_size, action_dim,lr, gamma,T,e,r,w,persona_ration,temperature,story_count,data_name)

    calc_log = np.zeros((10, 5))
    calc_nll_log = np.zeros((10, 5))
    attr_calc_log = np.zeros((10, 5))
    attr_calc_nll_log = np.zeros((10, 5))

    for count in range(10):
        obs.reset(
            edge_sparse[LEARNED_TIME].clone(),
            feat_sparse[LEARNED_TIME].clone(),
            persona=persona_ration
        )
   
        for test_time in range(TOTAL_TIME - GENERATE_TIME):

            edges, two_hop_neighbar,feature = obs.state()
            edge_action,edge_prob ,feat_prob ,feat_action = agents.actor.pred(feature,edges,two_hop_neighbar,test_time,action_dim,feat_size)
            reward = obs.step(feat_action,edge_action,test_time)
            

            #属性値の評価 
            pred_prob = torch.ravel(feat_prob.to_dense()).to("cpu")
            pred_prob = pred_prob.to("cpu").detach().numpy()
            print(type(load_data.feature[GENERATE_TIME + test_time]))

            if isinstance(load_data.feature[GENERATE_TIME + test_time],csc_matrix):
                dense_array = np.array(load_data.feature[GENERATE_TIME + test_time].todense())
            else:
                dense_array = np.array(load_data.feature[GENERATE_TIME + test_time].to_dense())
            attr_tensor = torch.from_numpy(dense_array)

            detach_attr = (
                torch.ravel(attr_tensor)
                .detach()
                .to("cpu")
            )
            
            gc.collect()

            detach_attr[detach_attr > 0] = 1
            pos_attr = detach_attr.numpy()
            attr_test = np.concatenate([pos_attr], 0)
            attr_predict_probs = np.concatenate([pred_prob], 0)
            print("属性値の総数")
        
            #print("pred feat sum",attr_predict_probs.reshape((500, -1)).sum(axis=1))
            #print("target edge sum",attr_test.reshape((500, -1)).sum(axis=1))
            print("pred feat sum",attr_predict_probs.sum())
            print("feat action sum",feat_action.sum())
            print("target edge sum",attr_test.sum())
            


            try:
                
                # NLLを計算
                criterion = nn.CrossEntropyLoss()
                error_attr = criterion(
                    torch.from_numpy(attr_predict_probs),
                    torch.from_numpy(attr_test),
                )
               
               
              
                auc_actv = roc_auc_score(attr_test, attr_predict_probs)
 
            
            finally:
                print("attr auc, t={}:".format(test_time), auc_actv)
                #print("attr auc, t={}:".format(test_time), pr_auc)
                #print("attr nll, t={}:".format(t), error_attr.item())
                #attr_calc_log[count][test_time] = auc_actv
                attr_calc_log[count][test_time] = auc_actv
                attr_calc_nll_log[count][test_time] = error_attr.item()
            
      
            del pred_prob,detach_attr,auc_actv,error_attr,attr_predict_probs,attr_test,attr_tensor
            gc.collect()
            #エッジの評価u

            #予測データ
            target_prob= edge_prob.to_dense()
            #print("pi",pi_test)     
            target_prob = target_prob.view(-1)
            target_prob = target_prob.to("cpu").detach().numpy()
            edge_predict_probs = np.concatenate([target_prob], 0)
            del target_prob
            gc.collect()
            #テストデータ
            if isinstance(load_data.adj[GENERATE_TIME + test_time],csc_matrix):   
                dense_array = np.array(load_data.adj[GENERATE_TIME + test_time].todense())
            else:
                dense_array = np.array(load_data.adj[GENERATE_TIME + test_time].to_dense())
            edge_tensor = torch.from_numpy(dense_array)
            pos_edge = (
                torch.ravel(edge_tensor)
                .detach()
                .to("cpu")
            ).numpy()
            del edge_tensor
            gc.collect()

            edge_test = np.concatenate([pos_edge], 0)
            del pos_edge
            gc.collect()

            criterion = nn.CrossEntropyLoss()
            error_edge = criterion(
                torch.from_numpy(edge_predict_probs),
                torch.from_numpy(edge_test),
            )


            auc_calc = roc_auc_score(edge_test, edge_predict_probs)  
            print("edge auc, t={}:".format(test_time), auc_calc)

            print(T,e,r,w)
            calc_log[count][test_time] = auc_calc
            calc_nll_log[count][test_time] = error_edge.item()
           
            del target_prob,detach_edge,edge_predict_probs,edge_test,edge_tensor,auc_calc,error_edge
            gc.collect()

            path_save = "experiment_data"

            np.save(path_save+"/proposed_edge_auc", calc_log)
            np.save(path_save+"/proposed_edge_nll", calc_nll_log)
            np.save(path_save+"/proposed_attr_auc", attr_calc_log)
            np.save(path_save+"/proposed_attr_nll", attr_calc_nll_log)


if __name__ == "__main__":

    for i in [5]:
        execute_data(i,"Twitter","complete")

    print("time:",s-e)