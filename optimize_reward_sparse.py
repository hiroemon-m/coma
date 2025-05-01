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

def search_diagonal(data):
    """スパーステンソルの対角成分を削除."""
    filtered_indices = data.coalesce().indices()
    mask = filtered_indices[0] == filtered_indices[1]
    



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
        #self.optimizer = optim.SGD(self.model.parameters(), lr=0.05)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.03)

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
        print("reward",reward)
        #impactの計算
        if t > 0:
     

            #impactを計算:(L2ノルム)^2
            old_feature = self.feats[t-1] 
            neigboer_feature = torch.sparse.mm(self.edges[t-1],self.feats[t-1])
            new_feature = self.feats[t] 
            new_neigboer_feature = torch.sparse.mm(self.edges[t],self.feats[t])
            diff_feature = torch.sub(neigboer_feature,new_neigboer_feature) #属性値の差を求める
      
    
            diff_feature = diff_feature*diff_feature
           
            softmax_diff_feature = torch.sparse_coo_tensor(diff_feature.indices(),(torch.abs(diff_feature.values())),diff_feature.size()).coalesce()
            #min-maxスケーリング
            min_value = torch.min(softmax_diff_feature.values())
            max_value = torch.max(softmax_diff_feature.values())
            softmax_diff_feature_scaling = (softmax_diff_feature.values() - min_value) / (max_value - min_value)
            filltered_indices = softmax_diff_feature.indices()[:,softmax_diff_feature_scaling.nonzero(as_tuple=True)[0]]
            filltered_values = softmax_diff_feature_scaling[softmax_diff_feature_scaling.nonzero(as_tuple=True)[0]]

            softmax_diff_feature = torch.sparse_coo_tensor(filltered_indices,filltered_values,softmax_diff_feature.size()).coalesce()

            
            print("softmax_diff_feature",softmax_diff_feature)
            imapct_indices = softmax_diff_feature.indices()
            imapct_values = softmax_diff_feature.values()
            imapact_size = softmax_diff_feature.size()
            impact_row = imapct_indices[0]
            scale_gamma = self.model.gamma[impact_row,0]
            reward_impact = torch.sparse_coo_tensor(imapct_indices,imapct_values*scale_gamma,imapact_size).coalesce()
            print("reward_impact",reward_impact)

            #sparseのまま和をとる
            # Step 1: 各行に含まれる value を group-by して sum を計算（denseを通さず）
            row_indices = reward_impact.indices()[0]  # 各 value が属する行

            # Step 2: 行ごとに加算（index_addで可能）
            row_sums = torch.zeros(imapact_size[0])
            row_sums = row_sums.index_add(0, row_indices, reward_impact.values())  # shape: (32,)

            # Step 3: 非ゼロ行を抽出して sparse(32 x 1) に変換
            nonzero_mask = row_sums != 0
            nz_rows = torch.where(nonzero_mask)[0]
            nz_values = row_sums[nz_rows]

            # (row, col=0)
            sparse_indices = torch.stack([nz_rows, torch.zeros_like(nz_rows)])

            # Step 4: 再構築（この tensor は sparse かつ勾配付き）
            #sparse_row_sums = torch.sparse_coo_tensor(
            #    sparse_indices, nz_values/imapact_size[1], size=(imapact_size[0], 1)
            #)
            sparse_row_sums = torch.sparse_coo_tensor(
                sparse_indices, nz_values, size=(imapact_size[0], 1)
            )

                    
   
            reward += sparse_row_sums

            #三宅はこれ、だけどNにスケールするので...
            #reward_value = torch.sparse.mm(impact_coo,self.model.gamma).to_sparse().coalesce()
            
            #reward_impact = torch.sparse_coo_tensor(reward_value.indices(),F.softmax(torch.abs(reward_value.values()),dim=0),reward_value.size())
            #impact_coo = torch.sparse.mm(diff_feature,diff_feature.t())
            #impact_norm = impact_coo/data.feature[0].size()[1]
            #reward_impact = torch.sparse.mm(impact_norm,self.model.gamma).to_sparse().coalesce()
            #reward += reward_impact

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