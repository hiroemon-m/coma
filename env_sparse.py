# Standard Library
import math

# Third Party Library
import networkx as nx
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torchviz import make_dot


import numpy as np
import scipy
import torch

# First Party Library
import config
device = config.select_device




def remove_diagonal(data):
    """2hop先の隣接行列の対角成分を取り除く"""
    filtered_indices = data.indices()
    filtered_values = data.values()
    mask = filtered_indices[0] != filtered_indices[1]  # 行番号と列番号が異なる要素だけを選択
    # マスクを使って新しいインデックスと値をフィルタリング
    new_indices = filtered_indices[:, mask]
    new_values = filtered_values[mask]
    remove_diagonal_sparse = torch.sparse_coo_tensor(new_indices,new_values,data.size()).coalesce()

    return remove_diagonal_sparse
            


def norm_func(x):
    """スパーステンソルのL2ノルムで正規化."""
    x = x.coalesce()
    x_indices = x.indices()
    x_values = x.values()
    x_size = x.size()
    x_row_indices = x_indices[0]            # 行インデックス
    x_squared_values = x_values ** 2                # 各値の平方
    
    # in-place操作を避けるため、新しいテンソルを作��
    x_row_sums = torch.zeros(x_size[0])
    x_row_sums = x_row_sums.scatter_add(0, x_row_indices, x_squared_values)  # in-place操作を避ける
    
    x_l2_norms = torch.sqrt(x_row_sums)                # 平方根を取る
    x_normalized_values = x_values / x_l2_norms[x_row_indices] # 正規化
    
    # スパーステンソルを作成
    normalized_x_feat = torch.sparse_coo_tensor(
        x_indices, 
        x_normalized_values, 
        x_size
    ).coalesce()

    return normalized_x_feat






def sparse_hadamard_product(adj, similarity):
    """
    スパーステンソルのアダマール積を計算。共通インデックスのみを効率的に扱う。
    """
    # スパーステンソルを圧縮
    adj = adj.coalesce()
    similarity = similarity.coalesce()

    # 非ゼロ要素のインデックスと値を取得
    indices_a, values_a = adj.indices(), adj.values()
    indices_b, values_b = similarity.indices(), similarity.values()

    # (行, 列) を結合してユニークなキーとして扱う
    indices_a_flat = indices_a[0] * adj.size(1) + indices_a[1]
    indices_b_flat = indices_b[0] * similarity.size(1) + indices_b[1]

    # 共通インデックスを特定
    common_mask = torch.isin(indices_a_flat, indices_b_flat)

    # 共通インデックスに対応する値を取得
    common_indices = indices_a[:, common_mask]
    common_values_a = values_a[common_mask]

    # `indices_a_flat` と `indices_b_flat` の対応を見つけて、`values_b` を合わせる
    matched_b_indices = torch.searchsorted(indices_b_flat, indices_a_flat[common_mask])
    common_values_b = values_b[matched_b_indices]

    # アダマール積（要素ごとの積）を計算
    common_values = common_values_a * common_values_b

    # スパーステンソルとして返す
    result = torch.sparse_coo_tensor(common_indices, common_values, adj.size())
    result = result.coalesce()
    return result


def adj_sim_self(adj, feat):
    """gammaの計算"""
    adj_sparse = adj.coalesce()
    feat_sparse = feat.coalesce()
    similarity = torch.sparse.mm(feat_sparse, feat_sparse.t())
    neigh_similarity = sparse_hadamard_product(adj_sparse, similarity.to_sparse())


    return neigh_similarity
    
def adj_sim(adj, feat):
    """alphaの計算"""
    adj_sparse = adj.coalesce()
    feat_sparse = feat.coalesce()
    #L２ノルムの計算
    normalized_feat = norm_func(feat)
    #ノードの類似度の計算
    similarity = torch.sparse.mm(normalized_feat, normalized_feat.t())
    neigh_similarity = sparse_hadamard_product(adj_sparse, similarity.to_sparse())


    return neigh_similarity


# 勾配追跡用のデバッグ出力を追加
def debug_grad(name, tensor):
    if tensor.requires_grad:
        print(f"{name} arad_fn={tensor.grad_fn} , requires_grad=True")
        if tensor.grad is not None:
            print(f"{name} grad={tensor.grad}")
    else:
        print(f"{name} requires_grad=False")

class Env(nn.Module):
    def __init__(self, agent_num,edge, feature,alpha,beta,gamma,persona) -> None:
        #t=4での状態に初期��
        super().__init__()
        self.agent_num = agent_num
        self.edges = edge
        self.feature = feature
        self.feature_t = self.feature.t()
        self.alpha = nn.Parameter(alpha.clone().detach().requires_grad_(True))
        self.beta = nn.Parameter(beta.clone().detach().requires_grad_(True))
        self.gamma = nn.Parameter(gamma.clone().detach().requires_grad_(True))
        self.persona = persona.clone().detach().requires_grad_(False)
        
  

    def reset(self, edges, attributes,persona):
        self.edges = edges
        self.feature = attributes
        self.feature_t = self.feature.t()
        self.persona = persona.detach().clone().requires_grad_(True)

        return self.edges,self.feature


    """報酬を計算する"""
    def step(self,next_feature,next_action,time):
        with torch.no_grad():

            persona_num = self.persona[time].size()[1]
            #impactを計算:(L2ノルム)^2
            old_feature = self.feature 
            new_feature = next_feature 
            diff_feature = torch.sub(old_feature,new_feature).coalesce() #属性値の差を求める
            impact_coo = adj_sim_self(next_action,diff_feature)
            impact_norm = impact_coo/(self.feature[0].size()[0])
      
            persona_gamma = torch.mm(self.persona[time],self.gamma.view(persona_num,1)) #gammaを計算

            reward_impact = impact_norm.multiply(persona_gamma).coalesce()


            #simlalityを計算
            normed_next_feature = norm_func(next_feature) #単位ベクトルに変更する
            similality_coo = adj_sim(next_action,normed_next_feature)
            persona_alpha = torch.mm(self.persona[time],self.alpha.view(self.persona[time].size()[1],1))
            reward_sim = similality_coo.multiply(persona_alpha).coalesce()

            #costを計算
            persona_beta = torch.mm(self.persona[time],self.beta.view(self.persona[time].size()[1],1))
            reward_cost = self.edges.multiply(persona_beta).to_sparse().coalesce()
            reward = reward_sim - reward_cost + reward_impact

            #edges,featureを更新
            self.edges = next_action
            self.feature = next_feature

        return reward
    
    def update_reward(self,next_feature,one_hop_action,time,scaler,alpha_all=None,beta_all=None,gamma_all=None):

        if time == 0:
            # alpha, beta, gamma の計算と逆変換
            alpha_all = torch.matmul(self.persona[time], self.alpha).squeeze()
            beta_all = torch.matmul(self.persona[time], self.beta).squeeze()
            gamma_all = torch.matmul(self.persona[time], self.gamma).squeeze()

            # DataFrame 作成と逆変換
            data = pd.DataFrame({"alpha": alpha_all.clone().detach().numpy(), "beta": beta_all.clone().detach().numpy(), "gamma": gamma_all.clone().detach().numpy()})
            original_data = scaler.inverse_transform(data)

            # PyTorch テンソルに戻し requires_grad=True を設定
            alpha_all = torch.unsqueeze(torch.tensor(original_data[:, 0]),1).requires_grad_(True)
            beta_all = torch.unsqueeze(torch.tensor(original_data[:, 1]),1).requires_grad_(True)
            gamma_all = torch.unsqueeze(torch.tensor(original_data[:, 2]),1).requires_grad_(True)
            print(alpha_all.size())
            
        else:
            alpha_all = alpha_all.clone().detach().requires_grad_(True)
            beta_all = beta_all.clone().detach().requires_grad_(True)
            gamma_all = gamma_all.clone().detach().requires_grad_(True)
       
        optimizer = torch.optim.SGD([alpha_all,beta_all,gamma_all],lr=0.001)


        # impact計算の勾配追跡
        old_feature = self.feature.detach().clone()
        new_feature = next_feature.detach().clone()
        edge = self.edges.detach().clone()
        diff_feature = torch.sub(old_feature, new_feature)

        impact_coo = adj_sim_self(one_hop_action, diff_feature).coalesce()
        impact_norm = impact_coo/(self.feature[0].size()[0])

        #node x 属性値数 → node x 1にして、alpha,betaのnxnにたす
        #nnzの行番号を求める
        #row_indices = impact_norm.indices()[0]
        #row_values = impact_norm.values()s
        #tow_indicesの値を使って,行方向に足し合わせる(node x 1)
        #row_sum = torch.zeros(impact_norm.size()[0]).scatter_add(0, row_indices, row_values)
        #impact_norm_sum = torch.sparse_coo_tensor(impact_norm.indices(),row_sum,impact_norm.size())
        reward_impact = impact_norm.multiply(gamma_all).coalesce()
        # similarity計算の勾配追跡
        similality_coo = adj_sim(one_hop_action, next_feature) 
        reward_sim = similality_coo.multiply(alpha_all).coalesce()
        # cost計算の勾配追跡
        reward_cost = edge.multiply(beta_all).to_sparse().coalesce()
        reward = reward_sim - reward_cost + reward_impact
        reward_loss = torch.sparse.sum(reward)


        optimizer.zero_grad()
        reward_loss.backward()
        optimizer.step()
    
       

        return alpha_all,beta_all,gamma_all




    #隣接行列を返す
    def state(self):

        one_hop_neighbar = self.edges#1hop
        two_hop_neighbar = torch.sparse.mm(self.edges,self.edges).coalesce()#2hop

        return one_hop_neighbar, two_hop_neighbar, self.feature
