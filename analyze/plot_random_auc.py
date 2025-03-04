"""
ランダムの探索範囲を変更したときのAUCをプロットする
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def load_data(path):
    """データを読み込み、平均と標準偏差を返す"""
    data = np.load(path)
    return data.mean(axis=0), data.std(axis=0)

def setup_plot():
    """プロットの基本設定"""
    matplotlib.use('Agg')
    fig, ax = plt.subplots(figsize=(18, 12))
    plt.rcParams["font.size"] = 30
    plt.tick_params(labelsize=24)
    return fig, ax

def get_model_paths(data_name, plot_type):
    """各モデルのデータパスを返す"""
    base_path = "../"
    return {
        #"2hop": f"{base_path}experiment_data/action_space/n_hop/2hop/10%/proposed_{plot_type}_auc.npy",
        "0.1": f"{base_path}experiment_data/action_space/variable/space=0.1%/proposed_{plot_type}_auc.npy",
        "1": f"{base_path}experiment_data/action_space/variable/space=1%/proposed_{plot_type}_auc.npy",
        #"10": f"{base_path}experiment_data/action_space/variable/space=10%/proposed_{plot_type}_auc.npy",
        "50": f"{base_path}experiment_data/action_space/variable/space=50%/proposed_{plot_type}_auc.npy",
        "100": f"{base_path}experiment_data/action_space/variable/space=100%/proposed_{plot_type}_auc.npy",
    }
     

def plot_auc(data_name, plot_type):
    """AUCプロット作成
    Args:
        data_name: データセット名 (DBLP, NIPS, Twitter)
        plot_type: プロット種類 (attr, edge)
        data_type: データ種類 (complete, incomplete, original)
    """
    # プロット初期設定
    fig, ax = setup_plot()
    time_steps = np.array([1, 2, 3, 4, 5])

    # タイトルと軸の設定
    plt.title(f"{plot_type} AUC ({data_name})", fontsize=30)
    ax.set_xlabel("Time segment (year)", fontsize=24)
    ax.set_ylabel(f"{plot_type} AUC", fontsize=24)
    ax.set_xlim(1, 5)

    # 各モデルのプロット
    paths = get_model_paths(data_name, plot_type)
    for model, path in paths.items():
        mean, std = load_data(path)
        is_main_model = model in ["0.1", "1", "10", "50", "100"]
        alpha = 0.5 if is_main_model else 0.3
        lw = 5 if is_main_model else 2
        
        plt.fill_between(time_steps, mean + std, mean - std, alpha=alpha)
        plt.plot(time_steps, mean, label=model, lw=lw)

    # グラフの体裁設定
    plt.xticks(np.arange(1, 6, 1))
    plt.yticks(np.arange(0.5, 0.9, 0.05))
    plt.grid(alpha=0.8, linestyle="--")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0, fontsize=18)
    fig.tight_layout()

    # 保存
    save_path = f"../result/img/{data_name}/variable/{plot_type}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved {data_name} {plot_type}")

def main():
    """メイン実行関数"""
    data_names = ["Twitter"]
    plot_types = ["attr", "edge"]
    
    for data_name in data_names:
        for plot_type in plot_types:
            plot_auc(data_name, plot_type)

if __name__ == "__main__":
    main()