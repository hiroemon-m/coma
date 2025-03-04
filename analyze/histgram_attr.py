import matplotlib
matplotlib.use("Agg")  # GUI 非対応のバックエンドに切り替え
import matplotlib.pyplot as plt

# 以下は通常通りのコード
import torch
for i in range(5):
    path = "../edge_value_{}.pt".format(i)
    data = torch.load(path)

    if data.is_sparse:
        data = data.to_dense()

    bins = torch.histc(data.float(), bins=20, min=0, max=1)

    x_positions = [i + 0.5 for i in range(1,20)]
    plt.bar(x_positions, bins.tolist()[1:], width=1.0, edgecolor="black", align="center")
    plt.xticks(range(1,20), [f"{i/20}-{(i+1)/20}" for i in range(1,20)], rotation=45)
    plt.title("Histogram of Tensor Values")
    plt.xlabel("Range")
    plt.ylabel("Frequency")
    plt.savefig("histogram_{}.png".format(i))  # GUIを使わずファイルに保存 
    plt.close()

