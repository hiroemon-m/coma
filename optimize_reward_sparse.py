# Standard Library

# Third Party Library
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import time
from torchviz import make_dot

# First Party Library
import config
from init_real_data import init_real_data
import gc

device = config.select_device





def norm_func(x):
    """スパーステンソルのL2ノルムで正規化."""
    x = x.coalesce()
    x_indices = x.indices()
    x_values = x.values()
    x_size = x.size()
    x_row_indices = x_indices[0]            # 行インデックス
    x_squared_values = x_values ** 2                # 各値の平方
    
    # in-place操作を避けるため、新しいテンソルを作��z
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
    スパーステンソルのアダマール積（共通インデックスのみ）。
    勾配を保ったまま効率的に計算。
    """
    adj = adj.coalesce()
    similarity = similarity.coalesce()

    # いずれかが空なら空結果を返す（安全対策）
    if adj._nnz() == 0 or similarity._nnz() == 0:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long, device=adj.device),
            torch.tensor([], dtype=adj.dtype, device=adj.device),
            adj.size()
        ).coalesce()

    indices_a, values_a = adj.indices(), adj.values()
    indices_b, values_b = similarity.indices(), similarity.values()

    index_a = indices_a[0] * adj.size(1) + indices_a[1]
    index_b = indices_b[0] * similarity.size(1) + indices_b[1]

    sorted_b, b_sort_idx = torch.sort(index_b)
    search_idx = torch.searchsorted(sorted_b, index_a)

    # 範囲外アクセス対策
    if sorted_b.size(0) == 0:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long, device=adj.device),
            torch.tensor([], dtype=adj.dtype, device=adj.device),
            adj.size()
        ).coalesce()

    search_idx = torch.clamp(search_idx, max=sorted_b.size(0) - 1)

    matched_mask = sorted_b[search_idx] == index_a

    if matched_mask.sum() == 0:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long, device=adj.device),
            torch.tensor([], dtype=adj.dtype, device=adj.device),
            adj.size()
        ).coalesce()

    matched_indices = indices_a[:, matched_mask]
    matched_values_a = values_a[matched_mask]
    matched_values_b = values_b[b_sort_idx[search_idx[matched_mask]]]
    matched_values = matched_values_a * matched_values_b

    return torch.sparse_coo_tensor(matched_indices, matched_values, adj.size()).coalesce()



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


def is_csc(indices, shape):
    """
    与えられたインデックスがCSC形式に該当するか確認する。
    
    Args:
        indices (torch.Tensor): CSC形式のインデックス（ポインタ）
        shape (tuple): 行列の形状（行数, 列数）

    Returns:
        bool: CSC形式であればTrue、そうでなければFalse
    """
    if indices.dim() != 1:
        # インデックスが1次元でない場合、CSCではない
        return False
    if indices.size(0) != shape[1] + 1:
        # インデックスのサイズが列数+1でなければCSCではない
        return False
    if not torch.all(indices[:-1] <= indices[1:]):
        # ポインタが単調増加でない場合、CSCではない
        return False
    return True




class Model(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super().__init__()

        self.alpha = nn.Parameter(alpha, requires_grad=True).to(device)
        self.beta = nn.Parameter(beta, requires_grad=True).to(device)
        self.gamma = nn.Parameter(gamma, requires_grad=True).to(device)
        return


class Optimizer:
    def __init__(self, edges, feats, model: Model, size: int):
        self.edges = edges
        self.feats = feats
        self.model = model
        self.size = size
        #self.optimizer = optim.SGD(self.model.parameters(), lr=0.05)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.005)

        return






    def optimize_sparse(self,model, t: int):


        next_feature = self.feats[t] 
        next_action = self.edges[t]

        print("*"*100)
        print("対角要素の数")
        
     


        

        #simlalityを計算
        normed_next_feature = norm_func(next_feature) #単位ベクトルに変更する

        similality_coo = adj_sim(next_action,normed_next_feature)
        reward_sim = torch.sparse.mm(similality_coo,self.model.alpha).to_sparse().coalesce()


        #costの計算
        reward_costs = torch.sparse.mm(next_action,self.model.beta).to_sparse().coalesce()
        print("reward_costs",next_action)
        #t=0ではimpact計算不可能
        reward = reward_sim - reward_costs
        print("reward",reward)
        #impactの計算
        if t > 0:
     

            #impactを計算:(L2ノルム)^2
            old_feature = self.feats[t-1] 
            neigboer_feature = torch.sparse.mm(self.edges[t-1],self.feats[t-1])
            new_feature = self.feats[t] 
            new_neigboer_feature = torch.sparse.mm(self.edges[t],self.feats[t])
            diff_feature_sub = torch.sub(neigboer_feature,new_neigboer_feature) #属性値の差を求める
      
    
            diff_feature = torch.sqrt(diff_feature_sub*diff_feature_sub)
           
            softmax_diff_feature = torch.sparse_coo_tensor(diff_feature.indices(),diff_feature.values(),diff_feature.size()).coalesce()
            print("softmax_diff_feature",softmax_diff_feature)

     
            #三宅はこれ、だけどNにスケールするので...
     
            reward_impact = torch.sparse_coo_tensor(softmax_diff_feature.indices(),F.softmax(torch.abs(softmax_diff_feature.values()),dim=0),softmax_diff_feature.size())
            reward_impact = torch.sparse.mm(reward_impact,self.model.gamma).to_sparse().coalesce()
            reward += reward_impact

        #    print(torch.abs(new_feature - old_feature)+1e-4)
        
       
        loss = - reward.sum()
        #dot = make_dot(loss,params=dict(list(model.named_parameters())))

        # グラフを表示
        #dot.render("graph", format="png", cleanup=True)
        self.optimizer.zero_grad()
        loss.backward()

        print("*"*100)
        self.optimizer.step()



       


    def export_param(self,data_type,data_name):
        #gamma/NIPS/
        with open("optimize/{}/{}/model.param.data.fast".format(data_type,data_name), "w") as f:


            for i in range(self.size):
                f.write(
                    "{},{},{}\n".format(
                        self.model.alpha[i].item() ,
                        self.model.beta[i].item() ,
                        self.model.gamma[i].item(),
                    )
                )


if __name__ == "__main__":
    start = time.perf_counter()
    data_name = "NIPS"

    data = init_real_data(data_name)
    data_size = data.adj[0].size()[0]

   


    alpha = torch.ones(data_size,1)
    beta = torch.ones(data_size,1)
    gamma = torch.ones(data_size,1)
    model = Model(alpha, beta, gamma)

    


    optimizer = Optimizer(data.adj, data.feature, model, data_size)
    data_type = "complete"



    for t in range(5):
        #if data_name == "NIPS":
       #     optimizer.optimize(t)
        #else:
        #    optimizer.optimize_sparse(t)
        optimizer.optimize_sparse(model,t)


        optimizer.export_param(data_type,data_name)
    end = time.perf_counter()
    print((end-start)/60)