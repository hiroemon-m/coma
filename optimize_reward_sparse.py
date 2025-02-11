# Standard Library

# Third Party Library
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchviz import make_dot

# First Party Library
import config
from init_real_data import init_real_data
import gc

device = config.select_device

def search_diagonal(data):
    """スパーステンソルの対角成分を削除."""
    filtered_indices = data.coalesce().indices()
    mask = filtered_indices[0] == filtered_indices[1]
    
    print(len(mask))

def norm_func(x):
    x = x.coalesce()
    x_indices = x.indices()
    x_values = x.values()
    x_size = x.size()
    x_row_indices = x_indices[0]            # 行インデックス
    x_squared_values = x_values ** 2                # 各値の平方
    x_row_sums = torch.zeros(x_size[0])                # 行ごとの平方和を格納
    x_row_sums.index_add_(0, x_row_indices, x_squared_values)  # 行ごとに値を加算
    x_l2_norms = torch.sqrt(x_row_sums)                # 平方根を取る
    x_normalized_values = x_values / x_l2_norms[x_row_indices] # 正規化：各非ゼロ要素を行ごとの L2 ノルムで割る
    normalized_x_feat = torch.sparse_coo_tensor(x_indices, x_normalized_values, x_size) # 正規化されたスパーステンソルを作成
 

    return normalized_x_feat


"""隣接行列の類似度計算"""
def adj_sim(adj, feat):
    # 隣接行列
    A_sparse = adj.coalesce()
    row_indices, col_indices = A_sparse.indices()
    values_A = A_sparse.values()
    
    # 属性値行列
    B_sparse = feat.coalesce()
    B_indices = B_sparse.indices()
    B_values = B_sparse.values()

    # 非ゼロインデックスに対応する行列積を計算
    result_indices = []
    result_values = []

    for i, j in zip(row_indices, col_indices):
        # B の行要素と列要素を取得
        row_mask = B_indices[0] == i.item()
        col_mask = B_indices[0] == j.item()
        
        row_b = B_values[row_mask]
        col_b = B_values[col_mask]

        # 行要素と列要素のインデックスを比較して共通部分を計算
        if len(row_b) > 0 and len(col_b) > 0:
            # サイズを合わせるために行列積を取る
            row_indices_b = B_indices[1][row_mask]
            col_indices_b = B_indices[1][col_mask]

            # 共通する列要素を抽出
            common_indices = torch.isin(row_indices_b, col_indices_b)
            filtered_row_b = row_b[common_indices]

            common_indices = torch.isin(col_indices_b, row_indices_b)
            filtered_col_b = col_b[common_indices]

            # 要素が一致した場合のみ積を計算
            if len(filtered_row_b) > 0 and len(filtered_col_b) > 0:
                result_indices.append([i.item(), j.item()])
                result_values.append(torch.sum(filtered_row_b * filtered_col_b).item())

    # スパース結果を作成
    result_indices = torch.tensor(result_indices, dtype=torch.long).t()  # 転置して形状を整える
    result_values = torch.tensor(result_values, dtype=torch.float32)
    result_sparse = torch.sparse_coo_tensor(
        result_indices,
        result_values,
        size=adj.size()
    )

    return result_sparse

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
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.05)

        return






    def optimize_sparse(self,model, t: int):

        next_feature = self.feats[t] 
    
        next_action = self.edges[t]
        print("*"*100)
        print("対角要素の数")
        search_diagonal(next_action)
        

        #simlalityを計算
        normed_next_feature = norm_func(next_feature) #単位ベクトルに変更する
        similality_coo = adj_sim(next_action,normed_next_feature)
        reward_sim = torch.sparse.mm(similality_coo,self.model.alpha).to_sparse().coalesce()

        #costの計算
        reward_costs = torch.sparse.mm(next_action,self.model.beta).to_sparse().coalesce()

        #t=0ではimpact計算不可能
        reward = reward_sim - reward_costs

        #impactの計算
        if t > 0:

            #impactを計算:(L2ノルム)^2
            old_feature = self.feats[t-1] 
            new_feature = self.feats[t] 
            diff_feature = torch.sub(old_feature,new_feature) #属性値の差を求める
            impact_coo = adj_sim(next_action,diff_feature) 
            impact_norm = impact_coo/data.feature[0].size()[1]
            reward_impact = torch.sparse.mm(impact_norm,self.model.gamma).to_sparse().coalesce()
             
            reward += reward_impact

        #    print(torch.abs(new_feature - old_feature)+1e-4)
        
       
        loss = - reward.sum()
        #dot = make_dot(loss,params=dict(list(model.named_parameters())))

        # グラフを表示
        #dot.render("graph", format="png", cleanup=True)
        self.optimizer.zero_grad()
        loss.backward()
        print("*"*100)
        print(self.model.alpha.grad.mean())
        print(self.model.beta.grad.mean())
        if t > 0:
            print(self.model.gamma.grad.mean())
        self.optimizer.step()

       


    def export_param(self,data_type,data_name):
        #gamma/NIPS/
        with open("optimize/{}/{}/model.param.data.fast".format(data_type,data_name), "w") as f:
            max_alpha = 1.0
            max_beta = 1.0
            max_gamma = 1.0
            print("maxmax",max(self.model.alpha))
            print("maxmax",max(self.model.beta))
            print("maxmax",max(self.model.gamma))
            for i in range(self.size):
                f.write(
                    "{},{},{}\n".format(
                        self.model.alpha[i].item() / max_alpha,
                        self.model.beta[i].item() / max_beta,
                        self.model.gamma[i].item() / max_gamma,
                    )
                )


if __name__ == "__main__":
    start = time.perf_counter()
    data_name = "Twitter"

    data = init_real_data(data_name)
    data_size = data.adj[0].size()[0]

   


    alpha = torch.ones(data_size,1)
    beta = torch.ones(data_size,1)
    gamma = torch.ones(data_size,1)
    model = Model(alpha, beta, gamma)

    


    optimizer = Optimizer(data.adj, data.feature, model, data_size)
    data_type = "complete"


    print(len(data.feature),len(data.feature[0]))
    for t in range(5):
        #if data_name == "NIPS":
       #     optimizer.optimize(t)
        #else:
        #    optimizer.optimize_sparse(t)
        optimizer.optimize_sparse(model,t)


        optimizer.export_param(data_type,data_name)
    end = time.perf_counter()
    print((end-start)/60)