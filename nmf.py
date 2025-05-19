import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import NMF
import pandas as pd
import torch
import joblib


def tolist(data):
    """データをリスト形式で読み込む"""
    alpha, beta, gamma = [], [], []
    with open(data, "r") as f:
        for line in f:
            a, b, g = map(float, line.strip().split(","))
            alpha.append(a)
            beta.append(b)
            gamma.append(g)
    return alpha, beta, gamma


if __name__ == "__main__":
    data_name = "Twitter"
    persona_list = {"DBLP": [5, 25, 50], "NIPS": [3,5,8,12,16], "Twitter": [5,20,50,100], "Reddit": [5, 20, 50, 100, 200]}


    for k in persona_list[data_name]:
        path = f"optimize/complete/{data_name}/model.param.data.fast"
        alpha, beta, gamma = tolist(path)

        # データフレーム化と標準化
        data = pd.DataFrame({"alpha": alpha, "beta": beta, "gamma": gamma})
        scaler = MinMaxScaler().fit(data)
        #scaler = StandardScaler().fit(data)
        norm_data = scaler.transform(data)
        norm_df = pd.DataFrame(norm_data, columns=["alpha", "beta", "gamma"])

        

    
        X = norm_df.to_numpy()      # shape [N, F]
        print(k)
        nmf = NMF(n_components=k, init='random', random_state=0)
        W = nmf.fit_transform(X)      # ← これが gamma に相当
        print(W)
        H = nmf.components_           # ← これが means/sigmas の代わりに“ペルソナの特徴”
        print(H)

        # 保存
        joblib.dump(scaler, f"optimize/complete/{data_name}/persona={k}/scaler.pkl")
        np.save(f"optimize/complete/{data_name}/persona={k}/gamma.npy", W)
        np.save(f"optimize/complete/{data_name}/persona={k}/H.npy", H)




 

