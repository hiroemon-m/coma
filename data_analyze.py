from init_real_data import init_real_data
import torch
from sklearn.preprocessing import MinMaxScaler,Normalizer
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



def adj_sim(adj, feat):
    """隣接行列の類似度計算."""
    adj_sparse = adj.coalesce()
    feat_sparse = feat.coalesce()
    normalized_feat = calcu_l2(feat)
    similarity = torch.sparse.mm(normalized_feat, normalized_feat.t()).coalesce()
    print("similarity",similarity.values()[:20])
    #neigh_similarity = sparse_hadamard_product(adj_sparse, similarity).coalesce()
    neigh_similarity = similarity


    return neigh_similarity

def process_similarity(edges, attributes, weight, temp):
    """類似度と確率の計算."""
    
    adj_sim_matrix = adj_sim(edges, attributes)
    exp_input = adj_sim_matrix / temp

    exp_output = torch.exp(torch.where(exp_input.values()>70,torch.tensor(70),exp_input.values()))

    return torch.sparse_coo_tensor(
        exp_input.indices(),
        exp_output * weight,
        exp_input.size()
    ).coalesce()


LEARNED_TIME = 4
GENERATE_TIME = 5
TOTAL_TIME = 10
load_data = init_real_data("Twitter")
def remove_diagonal(data):
    """スパーステンソルの対角成分を削除."""
    filtered_indices = data.indices()
    mask = filtered_indices[0] == filtered_indices[1]
    new_indices = filtered_indices[:, mask]
    new_values = data.values()[mask]
    return torch.sparse_coo_tensor(new_indices, new_values, data.size()).coalesce()

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

for time in range(TOTAL_TIME):

    edge = load_data.adj[time].to_dense()
    attr = load_data.feature[time].to_dense()
    s_edge = remove_diagonal(load_data.adj[time].coalesce())



    print("-----------------{}----------------".format(time))
    print("属性値の総数")
    print(torch.sum(attr))

    trend_sum = torch.sparse.sum(load_data.feature[time],dim=0)
    print(trend_sum)
    row_sum_values = trend_sum.values()
    row_sum_indices = trend_sum.indices()
    min_value = torch.min(row_sum_values)
    max_value = torch.max(row_sum_values)
    normalized_values = (row_sum_values - min_value) / (max_value - min_value)
    normalized_trend = torch.sparse_coo_tensor(row_sum_indices, normalized_values, trend_sum.size()).coalesce()
    n_trend = remove_zeros_from_sparse(normalized_trend)
    print(n_trend)





    if time >= 1:
        edge_before = load_data.adj[time-1].to_dense()
        print(attr.size())
        attr_before = load_data.feature[time-1].to_dense()  
        print("一つ前の時刻と同じ属性値")
        sa = (attr>0.0)&(attr_before>0.0)
        print(torch.sum(sa))
        old = 0
        for k in range(time):
      
            old = old + load_data.feature[time].to_dense()
        print("過去興味を持ったものと同じか？",torch.sum((old>0)&(attr>0),dim=1))
      
        print("一つ前の時刻から０属性値")
        print(torch.sum(torch.where((torch.where(attr_before>0.0,1.0,0.0)-torch.where(attr>0.0,1.0,0.0))>0.0,1.0,0.0)))
        print("一つ前の時刻の隣接ノード同じ属性値")
        adj = torch.matmul(edge_before,attr_before)
        aa = (adj>0.0)&(attr>0.0)
        print(torch.sum(aa))
        print("一つ前の時刻の隣接でないノード同じ属性値")
        un_adj = torch.matmul(1-edge_before,attr_before)
        print(torch.sum((un_adj>0.0)&(attr>0.0)))
        print("一つ前の時刻の隣接でないノードと隣接ノードの同じ属性値")
        print(torch.sum((un_adj>0.0)&(adj>0.0)))
        #隣接と隣接でないどちらかでいい？という説の否定。多すぎる少なすぎることが多い

        #仮説　一つ前の時刻で属性値の類似度が高いものの属性値の影響受ける
        print("一つ前の時刻で類似しているノードと同じ属性値")
        norm = attr_before.norm(dim=1)[:, None] + 1e-8
        attr_before_norm = attr_before.div(norm)
        sim = torch.mm(attr_before_norm,torch.t(attr_before_norm))
        sim_index = torch.where(sim>0,1,0)
        sim_index = sim_index.to(torch.float32)
        sim_attr = torch.matmul(sim_index,attr_before)
        a = torch.zeros(15000,15000)
        #print(sim.size(),attr_before.size())
        #print((sim>0).size())
        #print(edge.size())
        #print(edge[sim>0].size())
        attr_now = load_data.feature[time].to_dense()
        norm = attr_now.norm(dim=1)[:, None] + 1e-8
        attr_now_norm = attr_now.div(norm)
        sim_now = torch.mm(attr_now_norm,torch.t(attr_now_norm))
        next_edge = load_data.adj[time+1].to_dense()
        attr_one = torch.zeros(15000,15000)
        attr_one[sim_now>0] = next_edge[sim_now>0]
        print("類似度の高いエッジのこと",torch.sum(torch.where(((attr_one))>0,1,0)))

        ed = process_similarity(load_data.adj[time],attr_now.to_sparse(),1,1).to_dense()
        attr_two = torch.zeros(15000,15000)
        attr_two[ed>0] = next_edge[ed>0]
        print("類似度の高いエッジのこと",torch.sum(torch.where(((attr_two))>0,1,0)))
        print("類似度の高いエッジのこと",torch.sum(torch.where(((ed))>0,1,0)))

        print(torch.sum((attr>0)&(sim_attr>0)))

        
        #全体で流行の属性値の影響を受けてる説
        #属性値を行方向に加算を行う。閾値以上のものとの一致度調べる
        #for n in range(int((torch.max(torch.sum(attr_before,dim=0)).item()))):
            
        #    print("流行の属性値 {}より大きい".format(n),torch.sum(torch.sum(attr_before,dim=0)>n))
        #    trend = (torch.sum(attr_before,dim=0)>n).repeat(15000,1)
        #    trend = torch.where(trend>0,1,0)
        #    print("流行の属性値",torch.sum((attr>0)&(trend)))
           # "0より大きい(少しでも被ってる)時に日隣接の属性値と類似している.
            #さらに、この中で隣接と同じもの非隣接と同じものを探そう
            #隣接
        #    print("流行と隣接の和集合",torch.sum((attr>0)&(trend)&(adj>0)))
            #非隣接
        #    print("流行と非隣接の和集合",torch.sum((attr>0)&(trend)&(un_adj>0)))

        #print("流行の属性値 {}より大きい".format(0),torch.sum(torch.sum(attr_before,dim=0)>0))
        #mm = MinMaxScaler()
        ta = (torch.sum(attr_before,dim=0) - torch.min(torch.sum(attr_before,dim=0)))/(torch.max(torch.sum(attr_before,dim=0)) -torch.min(torch.sum(attr_before,dim=0)))
        print(torch.sum(ta))
        trend = (ta).repeat(15000,1)

        #trend = torch.where(trend>0,1,0)
        ta = (attr>0)&(trend>0)
        
        print("{}流行の属性値".format(time),torch.sum(ta))
        #trend = torch.sum(attr_before,dim=0).repeat(500,1)
        
        print("ysoku",torch.sum(torch.where((attr_before+adj+trend)>0.5,1,0)))
        print("総数",torch.sum(attr))
        #for k in range(10):
        #    print("{}流行の属性値".format(k),torch.sum((attr>0)&(trend>(k**2))))
        #2hop
        twohop_edge = torch.mm(edge_before,edge_before)
        adj = torch.matmul(twohop_edge,attr_before)
        print("{}2ホップ先の属性値".format(time),torch.sum((attr>0)&(adj>0)))
        





        
        #print(sim[0])#高くても0.1程度 → 閾値で分ける


        #仮説2 全体の属性値の流行の影響を受ける

