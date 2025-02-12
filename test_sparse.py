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



# First Party Library
import csv
import config
from env_sparse import Env
from init_real_data import init_real_data
import time

#another file
from actor_review import Actor

torch.autograd.set_detect_anomaly(True)



device = config.select_device



class PPO:
    def __init__(self, obs,agent_num,input_size, action_dim, lr, gamma,T,e,r,w,x,s,rik,temperature,story_count,data_set):
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
        self.actor = Actor(T,e,r,w,x,s,rik,self.agent_num,temperature)
        self.new_actor = Actor(T,e,r,w,x,s,rik,self.agent_num,temperature)
        self.update_actor = Actor(T,e,r,w,x,s,rik,self.agent_num,temperature)
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
    def train(self, gamma, action_dim, feat_size, prob_sparse_memory, edge_sparse_memory,
        feat_sparse_memory, reward_memory):

        cliprange = 0.2
        storycount = self.story_count

        #方策関数の更新

        G_r = [None]*len(reward_memory)
        G_r[-1] = reward_memory[-1].detach().clone()
        for r in range(len(reward_memory) - 2, -1, -1):
            G_r[r] = gamma * G_r[r + 1] + reward_memory[r].detach().clone()



        for _ in range(3):
            loss = 0
            for i in range(storycount):
                old_policy = prob_sparse_memory[i].detach().clone() 
                new_policy = self.new_actor.train(feat_sparse_memory[i], edge_sparse_memory[i], i, action_dim, feat_size)

                # インデックスの結合と重複除去
                combined_indices = torch.cat([old_policy.indices(), new_policy.indices()], dim=1)
                unique_indices, inverse_indices = torch.unique(combined_indices.t(), dim=0, return_inverse=True)
                unique_indices = unique_indices.t()

                # 値の計算とクリッピング
                old_values = torch.zeros(unique_indices.size(1), device=old_policy.device)
                new_values = torch.zeros(unique_indices.size(1), device=new_policy.device)

                old_mask = inverse_indices[:old_policy.values().size(0)]
                new_mask = inverse_indices[old_policy.values().size(0):]

                # クリッピングを適用
                old_log_values = torch.log(torch.clamp(old_policy.values(), min=1e-10, max=1.0))
                new_log_values = torch.log(torch.clamp(new_policy.values(), min=1e-10, max=1.0))

                old_values[old_mask] = old_log_values
                new_values[new_mask] = new_log_values

                # ratio計算とクリッピング
                ratio = torch.exp(torch.clamp(new_values - old_values, min=-5.0, max=5.0))
                ratio = torch.clamp(ratio, min=1-cliprange, max=1+cliprange).detach().clone()

                # スパーステンソルの作成
                ratio_tensor = torch.sparse_coo_tensor(
                    unique_indices,
                    ratio,
                    old_policy.size()
                ).coalesce()

                # 報酬の計算
                G = G_r[i]
                reward = ratio_tensor.multiply(G)

                # 損失の計算
                filtered_policy = new_policy.coalesce()
                policy_values = torch.log(torch.clamp(filtered_policy.values(), min=1e-10))
                policy_tensor = torch.sparse_coo_tensor(
                    filtered_policy.indices(),
                    policy_values,
                    filtered_policy.size(),
                )         
                #loss -= torch.sparse.sum(policy_tensor * reward)
                loss -= torch.sparse.sum(policy_tensor * reward)


            # 勾配の計算と適用
            self.new_actor_optimizer.zero_grad()
   
            
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.new_actor.parameters(), max_norm=1.0)
            # パラメータのリストを取得
            parameters = list(self.new_actor.parameters())

            # カスタム勾配クリッピングを適用
            self.sparse_clip_grad_norm_(parameters, max_norm=5)

                
            #except RuntimeError as e:
             #   print("エラー:", e)

            # 勾配をチェック
            self.new_actor_optimizer.step()
            for name, param in self.new_actor.named_parameters():
                if param.grad is not None:
                    print(f"Gradient for {name}:", param.grad,param.grad_fn,param.is_leaf)
                    #print(f"  Moment (m): {self.new_actor_optimizer.state[param]['exp_avg']}")
                    #print(f"  Moment (v): {self.new_actor_optimizer.state[param]['exp_avg_sq']}")
                else:
                    print(f"No gradient for {name}",param.grad,param.grad_fn,param.is_leaf)


            print("param",self.new_actor.T,self.new_actor.e,self.new_actor.r,self.new_actor.W,self.new_actor.x,self.new_actor.s)


        return self.new_actor.T.clone().detach(), self.new_actor.e.clone().detach(), \
            self.new_actor.r.clone().detach(), self.new_actor.W.clone().detach(),self.new_actor.x.clone().detach(),self.new_actor.s.clone().detach()
    
    def update_reward(self,obs,T,e,r,w,x,s,mixture_ration,temperature,action_dim,feat_size,edge,feat, scaler):

        _, _ = obs.reset(edge, feat,mixture_ration)
        
        for time in range(5):

            self.update_actor = Actor(T,e,r,w,x,s,mixture_ration,self.agent_num,temperature)
            edge,two_hop_neighbar,feature = obs.state()
            edge_probs,edge_action,feat_action = self.update_actor.get_action(feature,edge,two_hop_neighbar,time,action_dim,feat_size)
            if time == 0:
                alpha_all,beta_all,gamma_all = obs.update_reward(feat_action.coalesce(),edge_action.coalesce(),time,scaler)
            else:
                alpha_all,beta_all,gamma_all = obs.update_reward(feat_action.coalesce(),edge_action.coalesce(),time,scaler,alpha_all,beta_all,gamma_all)

            

        return alpha_all,beta_all,gamma_all
        
    
#@profile
def init_mixing_param(K):
    """混合率の初期化"""
    return np.random.dirichlet([1] * K)


def calc_gaussian_prob(x, mean, sigma):
    """多次元ガウス分布の確率を計算"""
    d = x - mean
    sigma += np.eye(sigma.shape[0]) * 1e-5  # 数値安定性を確保
    det = np.linalg.det(sigma)

    # 行列の特異性チェック
    if det <= 1e-10:  # 行列式がゼロに近い場合、修正
        sigma += np.eye(sigma.shape[0]) * 1e-3
        det = np.linalg.det(sigma)

    inv_sigma = np.linalg.inv(sigma).astype("float32")

    # ガウス分布の確率をログスケールで計算
    exp_term = -0.5 * np.dot(d.T, np.dot(inv_sigma, d))
    log_prob = exp_term - 0.5 * (np.log(det) + sigma.shape[0] * np.log(2 * np.pi))


    #return np.exp(exp_term) / np.sqrt((2 * np.pi) ** sigma.shape[0] * det)
    return np.exp(log_prob)


def calc_likelihood(X, means, sigmas, pi, K):
    """データXの現在のパラメタにおける対数尤度を計算"""
    likelihood = 0.0
    for n in X:
        temp = sum(pi[k] * calc_gaussian_prob(n, means[k], sigmas[k]) for k in range(K))
        likelihood += np.log(temp)
    return likelihood


def e_step(N, K, X, means, sigmas,pi):
    # E-Step
    gamma = np.zeros((N, K))
    for n, x in enumerate(X):
        denominator = sum(pi[0][n][k] * calc_gaussian_prob(x, means[k], sigmas[k]) for k in range(K))

        gamma[n] = [pi[0][n][k] * calc_gaussian_prob(x, means[k], sigmas[k]) / denominator for k in range(K)]

    return gamma

def m_step(X,gamma,K):
    # M-Step
    Nks = gamma.sum(axis=0)
    pi = init_mixing_param(K)
    #k:ペルスナ
    for k in range(K):
        pi[k] = Nks[k] / N #混合比の計算
        means[k] = np.sum(gamma[:, k, np.newaxis] * X, axis=0) / Nks[k]
        diff = X - means[k]
        sigmas[k] = (gamma[:, k] * diff.T @ diff) / Nks[k]
    
    return pi, means ,sigmas



def em_algorithm(alpha,beta,gamma,means,sigmas,mixture_ration,scaler):
    """EMアルゴリズム"""
    N = len(mixture_ration[0])
    data = pd.DataFrame({
        "alpha": torch.squeeze(alpha).detach().numpy(), 
        "beta": torch.squeeze(beta).detach().numpy(),
        "gamma": torch.squeeze(gamma).detach().numpy()
         })

    #norm_data = StandardScaler().fit_transform(data)
    #norm_df = pd.DataFrame(norm_data, columns=["alpha", "beta", "gamma"])
    norm_data = scaler.transform(data)
    norm_df = pd.DataFrame(norm_data, columns=["alpha", "beta", "gamma"])
    norm_tensor = torch.tensor(norm_df.values)
    K = len(mixture_ration)
    pi = mixture_ration
    #M-step
    #pi, means, sigmas = m_step(X,pi,K)

    #E-step
    rik = e_step(N, K, norm_tensor, means, sigmas,pi)

    return rik



#@profile
def show():
    print("dell")

#@profile
def execute_data(persona_num,data_name,data_type):
    ##デバッグ用
    torch.autograd.set_detect_anomaly(True)
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
    action_dim = agent_num
    input_size = len(load_data.feature[LEARNED_TIME][1])
    feat_size = agent_num
    

 

    path_n = "optimize/{}/{}/".format(data_type,data_name)
    path = path_n+"persona={}/gamma.npy".format(int(persona_num))
    persona_ration = np.load(path)
    persona_ration = persona_ration.astype("float32")
    #time,node,persona
    persona_ration = torch.from_numpy(np.tile(persona_ration,(story_count,1,1))).to(device)

    path = path_n+"persona={}/means.npy".format(int(persona_num))
    means = np.load(path)
    means = means.astype("float32")
    means = torch.from_numpy(means).to(device)

    path = path_n+"persona={}/sigma.npy".format(int(persona_num))
    sigmas = np.load(path)
    sigmas = sigmas.astype("float32")
    sigmas = torch.from_numpy(sigmas).to(device)

    path = path_n+"persona={}/scaler.pkl".format(int(persona_num))
    scaler = joblib.load(path)

    alpha = means[:,0]
    beta = means[:,1]
    gamma = means[:,2]
    

    #重み(固定値)

    print("load",load_data.adj,load_data.feature)
    print("means",means)
    print("sigmas",sigmas)


    #パラメータ
    if data_name == "NIPS":
        mu = 0.194
        lr = 1.563e-06
        lr = 0.001
        temperature = 0.01
        T = torch.tensor([1.055 for _ in range(persona_num)], dtype=torch.float32)
        e = torch.tensor([1.347 for _ in range(persona_num)], dtype=torch.float32)
        r = torch.tensor([0.697 for _ in range(persona_num)], dtype=torch.float32)
        w = torch.tensor([0.026 for _ in range(persona_num)], dtype=torch.float32)
    
    elif data_name == "DBLP":
        mu = 0.0229
        #lr = 0.000952
        lr = 0.001
        temperature = 0.01
        T = torch.tensor([1.481 for _ in range(persona_num)], dtype=torch.float32)
        e = torch.tensor([0.759 for _ in range(persona_num)], dtype=torch.float32)
        r = torch.tensor([0.868 for _ in range(persona_num)], dtype=torch.float32)
        w = torch.tensor([0.846 for _ in range(persona_num)], dtype=torch.float32)
  
    else:
        mu = 0.13608351100239896
        #lr = 0.0011929646522007911
        lr = 0.005
        temperature = 0.01
        T = torch.tensor([1.00 for _ in range(persona_num)], dtype=torch.float32)
        e = torch.tensor([1.00 for _ in range(persona_num)], dtype=torch.float32)
        r = torch.tensor([0.5 for _ in range(persona_num)], dtype=torch.float32)
        w = torch.tensor([1.00 for _ in range(persona_num)], dtype=torch.float32)
        x = torch.tensor([0.5 for _ in range(persona_num)], dtype=torch.float32)
        s = torch.tensor([0.5 for _ in range(persona_num)], dtype=torch.float32)
        

    ln = 0
    ln_sub = 0

    flag = True
    episode = 0
    episodes_reward = []

    while flag and ln_sub <= 1:
        
        print("----------episode:{}----------".format(episode))

    
       

        # E-step
        #mixture_ratio:混合比率

        if episode <= 10:
            mixture_ratio = persona_ration

       
        else:
            new_mixture_ratio = em_algorithm(alpha,beta,gamma,means,sigmas,mixture_ratio,scaler)
         
     
                      
            # スムージングファクター
            if episode <= 20:
                clip_ration = 0.2
                updated_prob_tensor = (1 - clip_ration) * mixture_ratio + clip_ration * torch.from_numpy(new_mixture_ratio.astype("float32"))
                mixture_ratio = updated_prob_tensor.float()
                del updated_prob_tensor
                gc.collect()
            else:
                mixture_ratio = torch.from_numpy(new_mixture_ratio.astype("float32"))
                ratio_size = mixture_ratio.size()
                mixture_ratio = mixture_ratio.expand(5,ratio_size[0],ratio_size[1])
 
        
        alpha = means[:,0]
        beta = means[:,1]
        gamma = means[:,2]
            


        #personaはじめは均等
        
                    #環境の設定
        obs = Env(
            agent_num=agent_num,
            edge = edge_sparse[LEARNED_TIME],
            feature = feat_sparse[LEARNED_TIME],
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            persona=mixture_ratio
        )


        
        agents = PPO(obs,agent_num, input_size, action_dim,lr, mu,T,e,r,w,x,s,mixture_ratio,temperature,story_count,data_name)
        episode_reward = 0
        prob_sparse_memory = []
        edge_sparse_memory = []
        feat_sparse_memory = []
        reward_memory = []
        edge_sparse_memory.append(edge_sparse[LEARNED_TIME].clone())
        feat_sparse_memory.append(feat_sparse[LEARNED_TIME].clone())


        #時間をすすめる
        for time in range(story_count):
            #状態を取得
            edge,two_hop_neighbar,feature = obs.state()
  
            #行動を獲得
            edge_probs,edge_action,feat_action= agents.get_actions(edge,two_hop_neighbar,feature,time,action_dim,feat_size)
            
            print("done get actions")

            #属性値を確率分布の出力と考えているので、ベルヌーイ分布で値予測
    
            reward = obs.step(feat_action.coalesce(),edge_action.coalesce(),time)
            print("done reward")
   
            #agents.memory.probs[time]=edge_probs.clone()
            prob_sparse_memory.append(edge_probs.clone())

            #agents.memory.edges[time] = edge.clone()
            edge_sparse_memory.append(edge_action.clone())
            #agents.memory.features[time] = feature.clone()
            feat_sparse_memory.append(feat_action.clone())

            #agents.memory.next_edges[time]=edge_action.clone()
            #next_edge_sparse_memory.apeend(edge_action.clone())
            #agents.memory.next_features[time]=feat_action.clone()
            #next_feat_sparse_memory.apeend(feat_action.clone())

            #agents.memory.reward[time]=reward.clone()
            reward_memory.append(reward.clone())
            episode_reward = episode_reward + torch.sparse.sum(reward)
            del edge,feature,edge_probs,edge_action,feat_action,reward
            gc.collect()
      

        episodes_reward.append(episode_reward)

        T,e,r,w,x,s= agents.train(mu,action_dim,feat_size,prob_sparse_memory,edge_sparse_memory,feat_sparse_memory,reward_memory)
        print("Updated Policy")

        alpha,beta,gamma = agents.update_reward(obs,T,e,r,w,x,s,mixture_ratio,temperature,action_dim,feat_size,edge_sparse[LEARNED_TIME],feat_sparse[LEARNED_TIME],scaler)

        print("Updated Reward")
        
        #ln_before = lx
        #ln = agents.ln
        #ln_sub =abs(ln-ln_before)
        episode += 1
        #sub_ln.append([ln_sub,episode_reward])
        #print("ln_sub---------------------------------",ln_sub)
   
    
     
        if episode % 5 == 0:
            #print(reward)
            print(episodes_reward)
            print(f"episode: {episode}, average reward: {sum(episodes_reward[-10:]) / 10}")

        if episode >=20:
            flag = False

        else:
            print("delete")
            show()
            del episode_reward
            del agents,obs
            gc.collect()
            show()
 


    calc_log = np.zeros((10, 5))
    calc_nll_log = np.zeros((10, 5))
    attr_calc_log = np.zeros((10, 5))
    attr_calc_nll_log = np.zeros((10, 5))

    print("学習終了")
    print("パラメータ",T,e,r,w)

            
    new_mixture_ratio = em_algorithm(
                alpha,
                beta,
                gamma,
                means,
                sigmas,
                mixture_ratio,
                scaler
            )

                      

    # スムージングファクター
    #
    # a = 0.1
    #print("nm",new_mixture_ratio)
    #updated_prob_tensor = (1 - a) * mixture_ratio + a * new_mixture_ratio
    mixture_ratio = torch.from_numpy(new_mixture_ratio.astype("float32"))
    ratio_size = mixture_ratio.size()
    mixture_ratio = mixture_ratio.expand(5,ratio_size[0],ratio_size[1])
    agents = PPO(obs,agent_num, input_size, action_dim,lr, gamma,T,e,r,w,x,s,mixture_ratio,temperature,story_count,data_name)
    path_save = "experiment_data"
    np.save(path_save+"/train_persona_ration",np.concatenate([mixture_ratio.detach().numpy()],axis=0))
    np.save(path_save+"/train_paramerter",np.concatenate([T.detach().numpy(),e.detach().numpy(),r.detach().numpy(),w.detach().numpy()],axis=0))


        
    for count in range(10):

        obs.reset(
            edge_sparse[LEARNED_TIME].clone(),
            feat_sparse[LEARNED_TIME].clone(),
            persona=mixture_ratio
        )
   
        for test_time in range(TOTAL_TIME - GENERATE_TIME):

            edges, two_hop_neighbar,feature = obs.state()
            edge_action,edge_prob ,feat_prob ,feat_action = agents.actor.pred(feature,edges,two_hop_neighbar,test_time,action_dim,feat_size)
            reward = obs.step(feat_action,edge_action,test_time)
            

            #属性値の評価 
            pred_prob = torch.ravel(feat_prob.to_dense()).to("cpu")
            pred_prob = pred_prob.to("cpu").detach().numpy()
     

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

            detach_attr[detach_attr > 0] = 1.0
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
            
      
            del pred_prob,detach_attr
            #エッジの評価

            #予測データ
            target_prob= edge_prob.to_dense()
            #print("pi",pi_test)     
            target_prob = torch.ravel(target_prob)
            target_prob = target_prob.detach().numpy()
            edge_predict_probs = np.concatenate([target_prob], 0)
            
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

            print("エッジの総数")
            print("acction edge sum",edge_action.sum())
            print("target edge sum",edge_test.sum())

            criterion = nn.CrossEntropyLoss()
            error_edge = criterion(
                torch.from_numpy(edge_predict_probs),
                torch.from_numpy(edge_test),
            )
            del target_prob,detach_edge
            gc.collect()

            auc_calc = roc_auc_score(edge_test, edge_predict_probs)  
            print("edge auc, t={}:".format(test_time), auc_calc)

            print(T,e,r,w,x)    
            calc_log[count][test_time] = auc_calc
            calc_nll_log[count][test_time] = error_edge.item()
           

            path_save = "experiment_data"

            np.save(path_save+"/proposed_edge_auc", calc_log)
            np.save(path_save+"/proposed_edge_nll", calc_nll_log)
            np.save(path_save+"/proposed_attr_auc", attr_calc_log)
            np.save(path_save+"/proposed_attr_nll", attr_calc_nll_log)
            #print("t",T,"e",e,"r",r,"w",w)
            #print(mixture_ratio)





if __name__ == "__main__":
    #[5,8,12,16,24,32,64,128]
    #[4,8,12,16]
    s = time.time()
    for i in [5]:
        execute_data(i,"Twitter","complete")
    e = time.time()
    print("time:",s-e)