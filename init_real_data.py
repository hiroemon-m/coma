import torch
# First Party Library
from data_loader import (
    attr_graph_dynamic_spmat_DBLP,
    attr_graph_dynamic_spmat_NIPS,
    attr_graph_dynamic_spmat_twitter,
    attr_graph_dynamic_spmat_Reddit,
    spmat2sptensor,
    spmat2tensor,
)

TOTAL_TIME = 10
# input_graph = attr_graph_dynamic_spmat_twitter(T=TOTAL_TIME)
#input_graph = attr_graph_dynamic_spmat_NIPS(T=TOTAL_TIME)
#input_graph = attr_graph_dynamic_spmat_DBLP(T=TOTAL_TIME)
#input_graph = attr_graph_dynamic_spmat_Reddit(T=15)

class LoadDataset:
    adj = []
    feature = []

    def __init__(self, adj, feature):
        self.adj = adj
        self.feature = feature


def init_real_data(data_neam) -> LoadDataset:
    if data_neam == "NIPS":
        input_graph = attr_graph_dynamic_spmat_NIPS(T=TOTAL_TIME)
        adj = input_graph.Gmat_list
        feature = input_graph.Amat_list


    elif data_neam == "DBLP":
        input_graph = attr_graph_dynamic_spmat_DBLP(T=TOTAL_TIME) 
        adj = input_graph.Gmat_list
        feature = input_graph.Amat_list

    elif data_neam == "Twitter":
        input_graph = attr_graph_dynamic_spmat_twitter(T=TOTAL_TIME) 
        adj = input_graph.Gmat_list
        feature = input_graph.Amat_list

    elif data_neam == "Reddit":
        input_graph = attr_graph_dynamic_spmat_Reddit(T=15)




    else:
        adj = input_graph.Gmat_list
        feature = input_graph.Amat_list

        for t in range(input_graph.T):
            adj[t] = adj[t]
            _ = spmat2tensor(adj[t])

            adj[t] = _
            feature_ = input_graph.Amat_list[t]

            _ = spmat2tensor(feature_)
            feature[t] = _



    return LoadDataset(
        adj=adj,
        feature=feature,
    )


if "__name__" == "__main__":
    
    init_real_data()