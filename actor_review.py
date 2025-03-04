import torch
import torch.nn as nn
import gc
import time
from torchviz import make_dot
import sys


def remove_zeros_from_sparse(sparse_tensor):
    """
    スパーステンソルから値が0のエントリを削除します。

    Parameters:
    - sparse_tensor (torch.sparse_coo_tensor): スパーステンソル

    Returns:
    - torch.sparse_coo_tensor: 0を含まないスパーステンソル
    """
    # インデックスと値を取得
    sparse_tensor = sparse_tensor.coalesce()
    indices = sparse_tensor.indices()  # 2次元テンソル: 行と列のインデックス
    values = sparse_tensor.values()   # 非ゼロの値

    # 値が0でないインデックスを取得
    nonzero_mask = values != 0
    new_indices = indices[:, nonzero_mask]
    new_values = values[nonzero_mask]

    # 新しいスパーステンソルを作成
    new_sparse_tensor = torch.sparse_coo_tensor(
        new_indices, new_values, sparse_tensor.size()
    )

    return new_sparse_tensor


def Gumbel_Sigmoid(probabilities, tau=1.0, hard=False):
    """
    微分可能なBernoulliサンプリングを実装。

    Parameters:
    - probabilities (torch.Tensor): 各要素の確率（0〜1）。
    - tau (float): 温度パラメータ（シャープネスを制御）。
    - hard (bool): ハード化するかどうか（Trueの場合は0または1に変換）。

    Returns:
    - samples (torch.Tensor): サンプリング結果（連続値または離散値）。
    """
    # Gumbelノイズを加える
    input_prob = probabilities.coalesce().values()
    noise = torch.rand_like(input_prob)
    gumbel_noise = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
    logits = torch.log(input_prob + 1e-20) - torch.log(1 - input_prob + 1e-20)
    y = torch.sigmoid((logits + gumbel_noise) / tau)
    
    if hard:
        # ハード化（0または1に変換）
        y_hard = (y >= 0.5).float()
        # 勾配計算を継続するためのトリック
        y = y_hard.detach() - y.detach() + y
    
    return y

def tanh_func(data,data_name="DBLP"):
    """スパーステンソルのtanh計算."""
    tanhx = torch.tanh(data.values())
    tanhx_indices = tanhx.nonzero(as_tuple=True)[0]
    filtered_indices = data.indices()[:, tanhx_indices]
    filtered_values = tanhx[tanhx_indices]

    if data_name == "Twitter":
        mask = filtered_indices[0] != filtered_indices[1]
        new_indices = filtered_indices[:, mask]
        new_values = filtered_values[mask]
        return torch.sparse_coo_tensor(new_indices, new_values, data.size()).coalesce()

    else:
        return torch.sparse_coo_tensor(filtered_indices, filtered_values, data.size()).coalesce()



def sigmoid_func(feat_data):
    """スパーステンソルのsigmoid計算."""
    feat_data = feat_data.coalesce()
    feat_sigmoid = torch.sigmoid(feat_data.values())
    return torch.sparse_coo_tensor(feat_data.indices(), feat_sigmoid, feat_data.size()).coalesce()

def calcu_l2(feat_data):
    """スパーステンソルのL2ノルムで正規化."""
    feat_data = feat_data.coalesce()
    norm_indices = feat_data.indices()
    norm_values = feat_data.values()
    row_indices = norm_indices[0]
    squared_values = norm_values ** 2
    row_sums = torch.zeros(feat_data.size(0), device=feat_data.device)
    row_sums.index_add_(0, row_indices, squared_values)
    l2_norms = torch.sqrt(row_sums + 1e-10)
    normalized_values = norm_values / l2_norms[row_indices]
    return torch.sparse_coo_tensor(norm_indices, normalized_values, feat_data.size()).coalesce()

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
    #common_indices = indices_a[:, common_mask]
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


def sparse_hadamard_delete_product(adj, similarity):
    """
    スパーステンソルのアダマール積を計算。共通インデックスのみを効率的に扱う。
    """
    # スパーステンソルを圧縮
    adj = adj.coalesce()
    similarity = similarity.coalesce()


    # 非ゼロ要素のインデックスと値を取得
    indices_a, values_a = adj.indices(), adj.values()
    indices_b, values_b = similarity.indices(), similarity.values()

    adj_sim_matrix = torch.zeros(adj._nnz())

    # (行, 列) を結合してユニークなキーとして扱う
    indices_a_flat = indices_a[0] * adj.size(1) + indices_a[1]

    indices_b_flat = indices_b[0] * similarity.size(1) + indices_b[1]

    # 共通インデックスを特定
    common_mask = torch.isin(indices_a_flat, indices_b_flat)

    # 共通インデックスに対応する値を取得
    common_indices = indices_a[:, common_mask]
    #common_indices = indices_a[:, common_mask]
    common_values_a = values_a[common_mask]

    # `indices_a_flat` と `indices_b_flat` の対応を見つけて、`values_b` を合わせる

    matched_b_indices = torch.searchsorted(indices_b_flat, indices_a_flat[common_mask])
    common_values_b = values_b[matched_b_indices]

    # アダマール積（要素ごとの積）を計算
    common_values = common_values_a * common_values_b

    # common_mask が False のインデックスを 0 にする
    zero_indices = indices_a[:, ~common_mask]
    zero_values = torch.zeros(zero_indices.shape[1], dtype=adj.dtype, device=adj.device)

    # すべてのインデックスを統合
    final_indices = torch.cat([common_indices, zero_indices], dim=1)
    final_values = torch.cat([common_values, zero_values])

    # スパーステンソルとして返す
    result = torch.sparse_coo_tensor(final_indices, final_values, adj.shape)




    return result
  


def adj_sim(adj, feat):
    """隣接行列の類似度計算."""
    adj_sparse = adj.coalesce()
    feat_sparse = feat.coalesce()
    normalized_feat = calcu_l2(feat)
    similarity = torch.sparse.mm(normalized_feat, normalized_feat.t()).coalesce()

    neigh_similarity = sparse_hadamard_product(adj_sparse, similarity).coalesce()

    del similarity,normalized_feat,adj_sparse,feat_sparse,feat,adj
    gc.collect()

    return neigh_similarity

def adj_dis_sim(adj, feat):
    """隣接行列の類似度計算."""
    adj_sparse = adj.coalesce()
    feat_sparse = feat.coalesce()
    normalized_feat = calcu_l2(feat)
    similarity = torch.sparse.mm(normalized_feat, normalized_feat.t()).coalesce()
    neigh_similarity = sparse_hadamard_delete_product(adj_sparse, similarity).coalesce()
    del similarity,normalized_feat,adj_sparse,feat_sparse,feat,adj
    gc.collect()

    return neigh_similarity

def create_random_node(matrix_size):

    num_nonzeros = (matrix_size)
    random_num = int(num_nonzeros*(num_nonzeros*0.5))
    # ランダムなインデックスを生成
    indices = torch.randint(0, matrix_size, (2, random_num))  # 2行（行と列）のインデックス

    # ランダムな値を生成
    values = torch.ones(random_num)

    # スパーステンソルを作成
    sparse_tensor = torch.sparse_coo_tensor(indices, values, size=(matrix_size,matrix_size))
    print("sparse_tensor",sparse_tensor)


    return remove_zeros_from_sparse(sparse_tensor)







def remove_diagonal(data):
    """スパーステンソルの対角成分を削除."""
    filtered_indices = data.indices()
    mask = filtered_indices[0] != filtered_indices[1]
    new_indices = filtered_indices[:, mask]
    new_values = data.values()[mask]
    return torch.sparse_coo_tensor(new_indices, new_values, data.size()).coalesce()

def overwrite_exit(exit_prob, common_prob):
    """スパーステンソルの値を上書き."""
    indices_exit = exit_prob.indices()
    values_exit = exit_prob.values()
    indices_common = common_prob.indices()
    values_common = common_prob.values()

    matched_indices = torch.where(
        (indices_exit.T == indices_common.T.unsqueeze(1)).all(dim=2)
    )[1]
    updated_values = values_exit.clone()
    updated_values.index_copy_(0, matched_indices, values_common)

    return torch.sparse_coo_tensor(
        indices_exit, updated_values, exit_prob.size()
    ).coalesce()


class Actor(nn.Module):

    def __init__(self, T, e, r, w, x,s,persona, agent_num, temperature):
        super().__init__()
        size = (len(persona[0][0]),1)
        indices = torch.tensor([[0, 0, 0, 0, 0],[0, 1, 2, 3, 4]])  # 2次元に拡張
        self.T = nn.Parameter(T.clone().detach(),requires_grad=True)  # 密テンソル
        self.e = nn.Parameter(e.clone().detach(),requires_grad=True)  # 密テンソル
        self.r = nn.Parameter(r.clone().detach(),requires_grad=True)  # 密テンソル
        self.W = nn.Parameter(w.clone().detach(),requires_grad=True)  # 密テンソル
        self.x = nn.Parameter(x.clone().detach(),requires_grad=True)  # 密テンソル
        self.s = nn.Parameter(s.clone().detach(),requires_grad=True)  # 密テンソル
        self.temperature = temperature
        self.persona = persona
        self.agent_num = agent_num

        #self.test_optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.test_optimizer = torch.optim.SGD(self.parameters(), lr=0.001)


    def _process_similarity(self, edges, attributes, weight, temp):
        """類似度と確率の計算."""
        
        adj_sim_matrix = adj_sim(edges, attributes)
        exp_input = adj_sim_matrix / temp

        exp_output = torch.exp(torch.where(exp_input.values()>70,torch.tensor(70),exp_input.values()))

        return torch.sparse_coo_tensor(
            exp_input.indices(),
            exp_output * weight,
            exp_input.size()
        ).coalesce()

    

    def _disconect_prob(self, edges, attributes, x):
        """類似度と確率の計算."""
        
        exp_input = edges * x


        #exp_output = torch.exp(torch.where(exp_input.values()>70,torch.tensor(70),exp_input.values()))

        return exp_input.coalesce()
    def _combine_sparse_tensors(self, tensor_a, tensor_b):
        """スパーステンソルの結合."""
        indices = torch.cat([tensor_a.indices(), tensor_b.indices()], dim=1)
        values = torch.cat([tensor_a.values(), tensor_b.values()])
        return torch.sparse_coo_tensor(indices, values, tensor_a.size()).coalesce()

    def _create_two_hop_discconect(self,two_hop,one_hop):
        """2-hopの未接続ノードとのエッジ確率の計算."""
        indices_A = two_hop.indices()
        values_A = two_hop.values()
        size = two_hop.size()
        indices_B = one_hop.indices()
        values_B = one_hop.values()
      
        # A のインデックスをセットに変換
    
        indices_A_set = set(map(tuple, indices_A.T.tolist()))
        # B のインデックスをセットに変換
        indices_B_set = set(map(tuple, indices_B.T.tolist()))

        # A から B のインデックスを引く
        updated_indices_set = indices_A_set - indices_B_set
        

        # 更新後のインデックスをリストに変換
        #print("updated_indices",len(indices_A_set),len(indices_B_set))

        if len(updated_indices_set) == 0:
            updated_indices = torch.tensor([[0],[0]], dtype=torch.long) # 転置して [2, nnz] 形式にする
        else:
            updated_indices = torch.tensor(list(updated_indices_set), dtype=torch.long).T  # 転置して [2, nnz] 形式にする
        #print("updated_indices",updated_indices.size())
        # updated_values のサイズを updated_indices の列数に合わせる
        updated_values = torch.ones(updated_indices.size(1), dtype=torch.float32)
       
        # 新しいスパース隣接行列 C を作成
        create_edge = torch.sparse_coo_tensor(updated_indices, updated_values, size).coalesce()
        create_edge = self._remove_diagonal(create_edge)
        return create_edge

    def _remove_diagonal(self,n_hop):
        """2hop先の隣接行列の対角成分を取り除く"""
        filtered_indices = n_hop.indices()
        filtered_values = n_hop.values()
        mask = filtered_indices[0] != filtered_indices[1]  # 行番号と列番号が異なる要素だけを選択
        # マスクを使って新しいインデックスと値をフィルタリング
        new_indices = filtered_indices[:, mask]
        new_values = filtered_values[mask]
        remove_diagonal_sparse = torch.sparse_coo_tensor(new_indices,new_values,data.size()).coalesce()

        return remove_diagonal_sparse


    def forward(self, attributes, edges, two_hop_neighbar, total_past, times, agent_num, sparse_size):
     
 
        random_node = create_random_node(sparse_size)
        for i in range(len(self.persona[0][0])):
            if times == 0:
                attributes = total_past
            else:
                attributes = (total_past*(times+4) +attributes.coalesce().clone().detach())/(times+5)
            #attributes = attributes.coalesce().clone().detach()
            edges = edges.coalesce().clone().detach()

            two_hop_neighbar = two_hop_neighbar.coalesce().clone().detach()
   

         
            

            # 属性値更新 - sparse.mmの結果に対して直接演算を行う

            influence_attributes = attributes * self.W[i]
        
            neigh_feat = torch.sparse.mm(edges, influence_attributes)
            
      
            # W[i]との乗算 - 密テンソルとして計算
            #neigh_feat = neigh_feat_base * self.W[i]


      
        
            # スケーリング計算を修正
            scaled_attributes = attributes * self.r[i]
            #scaled_attributes = attributes.multiply(self.r_sparse.values()[i])
     

            scaled_neigh_feat = neigh_feat * (1 - self.r[i])
            #scaled_neigh_feat = neigh_feat * (1 - self.r.values()[i])

            #scaled_neigh_feat = neigh_feat.multiply((1 - self.r_sparse.values()[i]))


            # 属性値の更新を結合 - 明示的な加算
            next_feat = scaled_attributes + scaled_neigh_feat


                # スパーステンソルの加算
            #next_feat = self._combine_sparse_tensors(scaled_attributes, scaled_neigh_feat)

            # 属性値の確率
            feat_sigmoid_prob = sigmoid_func(next_feat)
            #確率的に行動
            feat_sigmoid_action_values = Gumbel_Sigmoid(feat_sigmoid_prob, self.temperature, hard=True)
            feat_sigmoid_action = torch.sparse_coo_tensor(feat_sigmoid_prob.indices(), feat_sigmoid_action_values, feat_sigmoid_prob.size())
            feat_sigmoid_action = remove_zeros_from_sparse(feat_sigmoid_action)
            #決定的に行動
            #feat_sigmoid_action = next_feat
     
            # 2-hop以内の未接続ノードとのエッジ確率
            two_hop_edges = two_hop_neighbar.detach().clone()
            two_hop_disconnect_edges = self._create_two_hop_discconect(two_hop_edges,edges.coalesce())
            
            connect_edges_action_space = (two_hop_disconnect_edges + random_node).coalesce()
            #connect_edges_action_space = (two_hop_disconnect_edges).coalesce()

            # 2-hopの未接続ノードとのエッジ確率の計算   
            #print("two_hop_disconnect_edges")
            #print("two_hop_disconnect_edges",two_hop_disconnect_edges._nnz())
            #print("random_node",random_node._nnz())
            #print("connect_edges_action_space",connect_edges_action_space._nnz())
            connect_edge = self._process_similarity(connect_edges_action_space, feat_sigmoid_action, self.e[i], self.T[i])

            # Min-Max スケーリング
            # 1. Min-Max スケーリングのために値を取得
            edge_values = connect_edge.values()

            # 2. 最小値と最大値を取得
            min_value = torch.min(edge_values)
            max_value = torch.max(edge_values)

            # 3. Min-Max スケーリング
            scaled_values = (edge_values - min_value) / (max_value - min_value + 1e-8)  # ゼロ除算防止

            # 4. スケール後のスパーステンソルを再構築
            scaled_connect_edge = torch.sparse_coo_tensor(connect_edge.indices(), scaled_values, connect_edge.size()).coalesce()
            create_edge_prob = tanh_func(connect_edge)

            # エッジ削除確率
            #print("edgedelete")
            one_hop_similality = adj_dis_sim(edges, calcu_l2(next_feat))
            #浮動小数点の誤差を防ぐために.whereを使用
            dissim_values = torch.where((1 - one_hop_similality.values())<0,torch.tensor(0.0),1 - one_hop_similality.values())
            #print("1hop先のノード",edges._nnz())
            #print("one_hop_similality",one_hop_similality._nnz())
            dissim = torch.sparse_coo_tensor(
                one_hop_similality.indices(), dissim_values, one_hop_similality.size()
            )
       
            delete_edge = self._disconect_prob(dissim, feat_sigmoid_action, self.x[i])
            
            # Min-Max スケーリング
            delete_edge_values = delete_edge.values()   
            min_values = torch.min(delete_edge_values)
            max_values = torch.max(delete_edge_values)
            delete_scaled_values = (delete_edge_values - min_values) / ((max_values - min_values) + 1e-8)
            delete_edge = torch.sparse_coo_tensor(delete_edge.indices(), delete_scaled_values, delete_edge.size()).coalesce()
            delete_edge_prob = torch.tanh(delete_edge.values())
            #print("one_hop_similality",one_hop_similality)
            #print("delete_edge_prob",delete_edge_prob)
            delete_edge_exit_prob = torch.sparse_coo_tensor(delete_edge.indices(),(1-delete_edge_prob),delete_edge.size())
 
            #加算
            exit_edge_prob = (create_edge_prob + delete_edge_exit_prob).coalesce()
            print("edge",edges._nnz())
            print("two_hop_neighbar",two_hop_neighbar._nnz())
            print("create_edge_prob",create_edge_prob._nnz())
            print("delete_edge_exit_prob",delete_edge_exit_prob._nnz())
            print("exit_edge_prob",exit_edge_prob._nnz())

            # グラフを表示
            #dot = make_dot(exit_edge_prob,params=dict(list(self.named_parameters())))
            #dot.render("graph_ab", format="png", cleanup=True)
            #break
        
            

            # エッジ確率の結合
            if i == 0:
                edges_prob = self.persona[times][:, i] * exit_edge_prob
            else:
                persona_weighted = self.persona[times][:, i] * exit_edge_prob
                edges_prob = self._combine_sparse_tensors(edges_prob, persona_weighted)

            # 属性確率の結合
            if i == 0:
                attr_prob = feat_sigmoid_prob * self.persona[times][:, i].view(-1, 1)
            else:
                persona_weighted_attr = feat_sigmoid_prob * self.persona[times][:, i].view(-1, 1)
                attr_prob = self._combine_sparse_tensors(attr_prob, persona_weighted_attr)
        
        
        return edges_prob, attr_prob, feat_sigmoid_prob, next_feat, scaled_attributes, scaled_neigh_feat

    def train(self, attributes, edges, total_past, times, agent_num, sparse_size):

        #2hop以内の隣接行列
        two_hop_neighbar = torch.sparse.mm(edges,edges).coalesce()

     


        edges_prob, attr_prob,feat_sigmoid_prob,next_feat,scaled_attributes,scaled_neigh_feat = self.forward(attributes, edges, two_hop_neighbar, total_past, times, agent_num, sparse_size)
        #dot = make_dot(edges_prob,params=dict(list(self.named_parameters())))
   
        # グラフを表示
        #dot.render("graph", format="png", cleanup=True)

        return edges_prob
        
    def get_action(self, attributes, edges, two_hop_neighbar, total_past, times, agent_num, sparse_size):
        with torch.no_grad():
            edges_prob, attr_prob ,_,_,_,_ = self.forward(attributes, edges, two_hop_neighbar, total_past, times, agent_num, sparse_size)

            # 属性アクション
            attr_prob_values = attr_prob.values()
            attr_value = torch.bernoulli(torch.sigmoid(attr_prob_values))
            non_zero_indices = attr_value.nonzero(as_tuple=True)[0]
            filtered_indices = attr_prob.indices()[:, non_zero_indices]
            filtered_values = attr_prob_values[non_zero_indices]
            attr_action = torch.sparse_coo_tensor(filtered_indices, filtered_values, attr_prob.size())

            # エッジアクション
            edge_value = (edges_prob.values() >= 0.5).float()
            non_zero_indices = edge_value.nonzero(as_tuple=True)[0]
            filtered_indices = edges_prob.indices()[:, non_zero_indices]
            filtered_values = edge_value[non_zero_indices]
            edge_action = torch.sparse_coo_tensor(filtered_indices, filtered_values, edges_prob.size()).coalesce()

        return edges_prob, edge_action, attr_action

    def pred(self, attributes, edges, two_hop_neighbar, total_past, times, agent_num, sparse_size):
        with torch.no_grad():
            edges_prob, attr_prob ,_,_,_,_ = self.forward(attributes, edges, two_hop_neighbar, total_past, times, agent_num, sparse_size)

            # 属性アクション
   
            attr_value = Gumbel_Sigmoid(attr_prob, self.temperature, hard=True)
            attr_action = torch.sparse_coo_tensor(attr_prob.indices(), attr_value, attr_prob.size())
            attr_action = remove_zeros_from_sparse(attr_action)


            # エッジアクション
            edge_value = (edges_prob.values() >= 0.5).float()
            non_zero_indices = edge_value.nonzero(as_tuple=True)[0]
            filtered_indices = edges_prob.indices()[:, non_zero_indices]
            filtered_values = edge_value[non_zero_indices]
            edge_action = torch.sparse_coo_tensor(filtered_indices, filtered_values, edges_prob.size()).coalesce()

        return edge_action, edges_prob, attr_prob, attr_action

    


