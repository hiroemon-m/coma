"""
一旦sparseにしたmの、不備あり
"""


import torch
import torch.nn as nn
import gc
import time

device = config.select_device

"""tanh"""
def tanh_func(data):

    x = data.values()
    tanhx = (torch.exp(x) - torch.exp(-x))/(torch.exp(x) + torch.exp(-x))
    tanhx_indices = tanhx.nonzero(as_tuple=True)[0]  # 0以外の値のインデックス
    filtered_indices = data.indices()[:, tanhx_indices]  # 0でないインデックスのみ
    filtered_values = tanhx[tanhx_indices]  # 0でない値のみ 
    mask = filtered_indices[0] != filtered_indices[1]  # 行番号と列番号が異なる要素だけを選択
    #マスクを使って新しいインデックスと値をフィルタリング
    new_indices = filtered_indices[:, mask]
    new_values = filtered_values[mask]
    
    #Nips,Dblptmp_coo = torch.sparse_coo_tensor(new_indices,new_values,tmp_x_e.size()).coalesce()
    tanh = torch.sparse_coo_tensor(new_indices,new_values,data.size()).coalesce()

    return tanh

"""sigmoid"""
def sigmoid_func(feat_data):

    feat_data = feat_data.coalesce()
    feat_indcies = feat_data.indices()
    feat_values = feat_data.values()
    feat_size = feat_data.size()
    feat_sigmoid = torch.sigmoid(feat_values)
    feat_sigmoid_prob = torch.sparse_coo_tensor(feat_indcies,feat_sigmoid,feat_size).coalesce()

    return feat_sigmoid_prob

def calcu_l2(feat_data):

    norm_indices = feat_data.indices()
    norm_values = feat_data.values()
    norm_size = feat_data.size()
    row_indices = norm_indices[0]                       # 行インデックス
    squared_values = norm_values ** 2                   # 各値の平方
    row_sums = torch.zeros(norm_size[0])                # 行ごとの平方和を格納
    row_sums.index_add_(0, row_indices, squared_values)  # 行ごとに値を加算
    l2_norms = torch.sqrt(row_sums)                # 平方根を取る
    normalized_values = norm_values / l2_norms[row_indices] # 正規化：各非ゼロ要素を行ごとの L2 ノルムで割る
    normalized_next_feat = torch.sparse_coo_tensor(norm_indices, normalized_values, norm_size) # 正規化されたスパーステンソルを作成

    return normalized_next_feat

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
    ).coalesce()

    return result_sparse


def overwrite_exit(exit_prob, common_prob):
    """common_probのindicesはexit_probに含まれる"""
    indices_exit = exit_prob.indices()
    values_exit = exit_prob.values()

    # common_prob の indices と values を取得
    indices_common = common_prob.indices()
    values_common = common_prob.values()

    # B のインデックスを A のインデックス内で探して対応する位置を特定
    matched_indices = torch.where(
        (indices_exit.T == indices_common.T.unsqueeze(1)).all(dim=2)
    )[1]

    # 勾配を維持しながら更新する
    # scatter_add を使い、既存の values_exit に更新値を追加
    updated_values = values_exit.clone()  # 勾配を維持するため clone
    updated_values = updated_values.index_copy(
        0, matched_indices, values_common
    )

    # 更新されたスパーステンソルを再構築
    updated_exit_prob = torch.sparse_coo_tensor(
        indices=indices_exit,
        values=updated_values,
        size=exit_prob.size()
    ).coalesce()

    return updated_exit_prob

def remove_diagonal(data):

    filtered_indices = data.indices()
    filtered_values = data.values()
    mask = filtered_indices[0] != filtered_indices[1]  # 行番号と列番号が異なる要素だけを選択
    # マスクを使って新しいインデックスと値をフィルタリング
    new_indices = filtered_indices[:, mask]
    new_values = filtered_values[mask]
    remove_diagonal_sparse = torch.sparse_coo_tensor(new_indices,new_values,data.size()).coalesce()

    return remove_diagonal_sparse
            


class Actor(nn.Module):

    def __init__(self, T, e, r, w,persona,agent_num,temperature) -> None:

        super().__init__()
        self.T = nn.Parameter(T.clone().detach().requires_grad_(True))
        self.e = nn.Parameter(e.clone().detach().requires_grad_(True))
        self.r = nn.Parameter(r.clone().detach().to(device).requires_grad_(True))
        self.W = nn.Parameter(w.clone().detach().to(device).requires_grad_(True))
        self.temperature = temperature
        self.persona = persona
        self.agent_num = agent_num
        

    def forward(self, attributes, edges, two_hop_neighbar, times, agent_num, sparse_size):
        
        for i in range(len(self.persona[0][0])):

            #エッジ(時刻t)の情報をもとに属性値を(時刻t+1)進める
            tmp_feat_prob = torch.empty(len(attributes), len(attributes[0]), 2)
            neighbar_feat = torch.sparse.mm(edges, attributes) * self.W[i] 
            next_feat = (self.r[i] * attributes + neighbar_feat * (1 - self.r[i])).coalesce()
            del neighbar_feat

            #sigmoidに通し、属性値の確率を計算
            feat_sigmoid_prob = sigmoid_func(next_feat)

            #エッジの存在確率を計算
            #2hop以内のエッジの存在確率の計算
            normalized_next_feat = calcu_l2(next_feat) #時刻Tで、属性値のL2ノルムを計算

            #時刻Tで、2ホップ以内の隣接行列を用いて類似度の計算
            two_hop_range = (edges + two_hop_neighbar).coalesce() #1hop隣接と2hop先
            one_values = torch.ones_like(two_hop_range.values())
            two_hop_adj = torch.sparse_coo_tensor(two_hop_range.indices(), one_values, two_hop_range.size())
            similality = adj_sim(two_hop_adj,normalized_next_feat)#2hop先の類似度を計算

            #similalityの計算
            exp_input = similality / (self.T[i])
            exp_input_values= exp_input.values()
            exp_output = torch.exp(exp_input_values)
            sparse_exp = torch.sparse_coo_tensor(exp_input.indices(),exp_output,exp_input.size())
            connect_edge = (sparse_exp * self.e[i]).coalesce()

            #確率に変換
            connect_edge_prob = tanh_func(connect_edge)

            
            #1hopのエッジ存在の確率
            #時刻Tで、1ホップ以内の隣接行列における属性値(T+1)の計算
            one_hop_adj = edges #隣接する数はそのまま
            one_hop_feat = next_feat #2ホップ以内の隣接行列の属性値
            one_hop_normalized_next_feat = calcu_l2(one_hop_feat)
            one_hop_similality = adj_sim(one_hop_adj,one_hop_normalized_next_feat)#2hop先の類似度を計算

            #similalityの計算
            one_hop_exp_input = one_hop_similality / (self.T[i])
            one_hop_exp_input_values= one_hop_exp_input.values()
            one_hop_exp_output = torch.exp(one_hop_exp_input_values)
            one_hop_sparse_exp = torch.sparse_coo_tensor(one_hop_exp_input.indices(),one_hop_exp_output,one_hop_exp_input.size())
            one_hop_connect_edge = (one_hop_sparse_exp * self.e[i]).coalesce()
            #tanh
            one_hop_connect_edge_prob = tanh_func(one_hop_connect_edge)


            #dissimilarityの計算
        
            delete_similality_indices = one_hop_similality.indices()
            delete_similality_values = one_hop_similality.values()
            delete_similality_size = one_hop_similality.size()
            dissimilarity = torch.sparse_coo_tensor(delete_similality_indices,1-delete_similality_values,delete_similality_size).coalesce()
            dissim_exp_input = dissimilarity / (self.T[i])
            dissim_exp_input_values= dissim_exp_input.values()
            dissim_exp_output = torch.exp(dissim_exp_input_values)
            
            dissim_sparse_exp = torch.sparse_coo_tensor(dissim_exp_input.indices(),dissim_exp_output,dissim_exp_input.size())
            delete_edge = (dissim_sparse_exp * self.e[i]).coalesce()

            #tanh
            delete_edge_prob = tanh_func(delete_edge)

            #connectとdeleteの共通部分の積
            indices_A = one_hop_connect_edge_prob.indices()
            values_A = one_hop_connect_edge_prob.values()
            size_A = one_hop_connect_edge_prob.size()

            indices_B = delete_edge_prob.indices()
            values_B = delete_edge_prob.values()
            
            if torch.equal(indices_A, indices_B):
                values_result = values_A * values_B
                common_prob = torch.sparse_coo_tensor(indices_B, values_result, size_A).coalesce()

            #coonectに共通部分で上書き
            exit_prob_clone = connect_edge_prob.clone()
            exit_prob = overwrite_exit(exit_prob_clone,common_prob)
                
            if i ==0 :
                edges_prob =  self.persona[times][:, i] * exit_prob
                #print(tmp_coo.values(),self.persona[times][:, i])
            else:
                exit_prob_persona = exit_prob*self.persona[times][:, i]
                                    # 両方のスパーステンソルのインデックスと値を取得
                indices1, values1 = edges_prob.indices(), edges_prob.values()
                indices2, values2 = exit_prob_persona.indices(), exit_prob_persona.values()
                #print("values",values1.max(),values2.max())
                # インデックスを結合
                combined_indices = torch.cat([indices1, indices2], dim=1)
                combined_values = torch.cat([values1, values2])
                
                # 重複するインデックスの値を合算
                edges_prob = torch.sparse_coo_tensor(
                    combined_indices, 
                    combined_values, 
                    edges_prob.size()
                ).coalesce()

            
            if i ==0:
                attr_prob = feat_sigmoid_prob*self.persona[times][:, i].view(-1, 1)
            else:
                indices1, values1 = attr_prob.coalesce().indices(), attr_prob.coalesce().values()
                tmp_feat_scaled = feat_sigmoid_prob * self.persona[times][:, i].view(-1, 1)
                indices2, values2 = tmp_feat_scaled.indices(), tmp_feat_scaled.values()
                
                # インデックスと値を結合
                combined_indices = torch.cat([indices1, indices2], dim=1)
                combined_values = torch.cat([values1, values2])
                
                # 重複するインデックスの値を合算
                attr_prob = torch.sparse_coo_tensor(
                    combined_indices, 
                    combined_values, 
                    attr_prob.size()
                ).coalesce()
            
        return edges_prob, attr_prob


    def train(self,attributes, edges,times,agent_num,sparse_size):
        """"trainで呼び出す"""
        two_hop_neighbar = remove_diagonal(edges@edges)

        edges_prob,_ = self.forward(attributes, edges, two_hop_neighbar, times, agent_num, sparse_size)

  

        return edges_prob



    #@profile    
    def get_action(self, attributes, edges, two_hop_neighbar, times, agent_num, sparse_size):

        a = time.time()

        with torch.no_grad():
            edges_prob,attr_prob = self.forward( attributes, edges, two_hop_neighbar, times, agent_num, sparse_size)

            attr_indces = attr_prob.indices()
            attr_size = attr_prob.size()
            attr_prob_values = attr_prob.values()
            attr_value = torch.bernoulli(torch.sigmoid(attr_prob_values))

            # 0でない値のインデックスをフィルタリング
            non_zero_indices = attr_value.nonzero(as_tuple=True)[0]  # 0以外の値のインデックス
            filtered_indices = attr_indces[:, non_zero_indices]  # 0でないインデックスのみ
            filtered_values = attr_prob_values[non_zero_indices]  # 0でない値のみ 
            attr_action = torch.sparse_coo_tensor(filtered_indices,filtered_values,attr_size)

            #del attr_ber, attr_prob
            gc.collect()

            edges_prob = edges_prob.coalesce()
            edge_indices = edges_prob.indices()
            edge_size = edges_prob.size()
            print("e_prob",edges_prob.values().max(),edges_prob.values().min())
            epsilon = 0.2
            noisy_probabilities = edges_prob.values() + epsilon * torch.rand(edges_prob.values().size(0))
            noisy_probabilities = torch.where(noisy_probabilities>=1, torch.tensor(1),noisy_probabilities)  # 負の値を0にする

            noisy_probabilities_coo = torch.sparse_coo_tensor(edge_indices,noisy_probabilities,edge_size)

            # 閾値でフィルタリング
            threshold = 0.5  # 閾値
            mask = noisy_probabilities > threshold  # 閾値を超えるエントリを保持
            filtered_indices = edge_indices[:, mask]  # フィルタリングされたインデックス
            filtered_values = noisy_probabilities[mask]  # フィルタリングされた
            # フィルタリングされたスパーステンソルの再構築
            filtered_noisy_probabilities_coo = torch.sparse_coo_tensor(filtered_indices, filtered_values, edge_size)

            edge_value = torch.where(edges_prob.values()>=0.5,1.0,0.0)
            #edge_value = torch.bernoulli(edges_prob.values())


            # 0でない値のインデックスをフィルタリング
            non_zero_indices = edge_value.nonzero(as_tuple=True)[0]  # 0以外の値のインデックス
            filtered_indices = edge_indices[:, non_zero_indices]  # 0でないインデックスのみ
            filtered_values = edge_value[non_zero_indices]  # 0でない値のみ 
            edge_action = torch.sparse_coo_tensor(filtered_indices,filtered_values,edge_size).coalesce()
            print("e_action",edge_action)
            
            b = time.time()
            print(attr_action)
            print("pr",b-a,torch.sum(attr_action),edge_size)

            return edges_prob, edge_action, attr_action

    

    def predict(self,attributes,edges, two_hop_neighbar,times,agent_num,sparse_size):

        edges_prob,attr_prob = self.forward( attributes, edges, two_hop_neighbar, times, agent_num, sparse_size)

        attr_indces = attr_prob.indices()
        attr_size = attr_prob.size()
        attr_prob_values = attr_prob.values()
        attr_value = torch.bernoulli(torch.sigmoid(attr_prob_values))

        # 0でない値のインデックスをフィルタリング
        non_zero_indices = attr_value.nonzero(as_tuple=True)[0]  # 0以外の値のインデックス
        filtered_indices = attr_indces[:, non_zero_indices]  # 0でないインデックスのみ
        filtered_values = attr_prob_values[non_zero_indices]  # 0でない値のみ 
        attr_action = torch.sparse_coo_tensor(filtered_indices,filtered_values,attr_size)

        #del attr_ber, attr_prob
        gc.collect()

        edges_prob = edges_prob.coalesce()
        edge_indices = edges_prob.indices()
        edge_size = edges_prob.size()

        epsilon = 0.3
        noisy_probabilities = edges_prob.values() + epsilon * torch.rand(edges_prob.values().size(0))
        noisy_probabilities = torch.where(noisy_probabilities>=1, torch.tensor(1),noisy_probabilities)  # 負の値を0にする

        noisy_probabilities_coo = torch.sparse_coo_tensor(edge_indices,noisy_probabilities,edge_size)

        # 閾値でフィルタリング
        threshold = 0.5  # 閾値
        mask = noisy_probabilities > threshold  # 閾値を超えるエントリを保持
        filtered_indices = edge_indices[:, mask]  # フィルタリングされたインデックス
        filtered_values = noisy_probabilities[mask]  # フィルタリングされた
        # フィルタリングされたスパーステンソルの再構築
        filtered_noisy_probabilities_coo = torch.sparse_coo_tensor(filtered_indices, filtered_values, edge_size)


        edge_value = torch.where(edges_prob.values()>=0.5,1.0,0.0)
        #edge_value = torch.bernoulli(edges_prob.values())

        # 0でない値のインデックスをフィルタリング
        non_zero_indices = edge_value.nonzero(as_tuple=True)[0]  # 0以外の値のインデックス
        filtered_indices = edge_indices[:, non_zero_indices]  # 0でないインデックスのみ
        filtered_values = edge_value[non_zero_indices]  # 0でない値のみ 
        edge_action = torch.sparse_coo_tensor(filtered_indices,filtered_values,edge_size).coalesce()

        gc.collect()

 

        return edge_action,edges_prob,attr_prob,attr_action




