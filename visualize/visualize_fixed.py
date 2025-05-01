#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import torch
import os
import plotly.graph_objects as go
import webbrowser

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

def main(data_name):
    # 結果ディレクトリの作成
    os.makedirs("results", exist_ok=True)
    
    path = f"../optimize/complete/{data_name}/model.param.data.fast"
    alpha, beta, gamma = tolist(path)

    # データフレーム化
    data = pd.DataFrame({"alpha": alpha, "beta": beta, "gamma": gamma})

    # 平均と共分散
    mean = data.mean().values
    cov = np.cov(data.T)

    # グリッド生成
    x = np.linspace(mean[0] - 0.1, mean[0] + 0.1, 30)
    y = np.linspace(mean[1] - 0.1, mean[1] + 0.1, 30)
    z = np.linspace(mean[2] - 0.1, mean[2] + 0.1, 30)
    X, Y, Z = np.meshgrid(x, y, z)
    pos = np.stack([X, Y, Z], axis=-1)

    # PDF計算
    rv = multivariate_normal(mean, cov)
    pdf = rv.pdf(pos)

    # Plotlyで等密度面（isosurface）を描画
    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=pdf.flatten(),
        isomin=np.max(pdf) * 0.1,
        isomax=np.max(pdf),
        surface_count=3,
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorscale='Viridis',
        opacity=0.6
    ))

    # データ点の描画
    fig.add_trace(go.Scatter3d(
        x=data["alpha"], y=data["beta"], z=data["gamma"],
        mode='markers',
        marker=dict(size=4, color='black'),
        name='データ点'
    ))

    fig.update_layout(
        title='3次元ガウス分布の等密度曲面とデータ点',
        scene=dict(
            xaxis_title='Alpha',
            yaxis_title='Beta',
            zaxis_title='Gamma'
        )
    )
    
    # HTMLファイルとして保存（nbformatエラーを回避）
    output_file = f"results/{data_name}_3d_gaussian.html"
    fig.write_html(output_file)
    print(f"3Dプロットを保存しました: {output_file}")
    
    # ブラウザで開く（オプション）
    try:
        webbrowser.open(output_file)
        print(f"ブラウザで{output_file}を開きました")
    except:
        print(f"ブラウザで開くことができませんでした。{output_file}を手動で開いてください。")
    
    # 2D投影も作成
    plot_2d_projections(data, data_name)
    
    # ペルソナごとのプロット（複数のペルソナがある場合）
    plot_personas(data_name)

def plot_2d_projections(data, data_name):
    """アルファ、ベータ、ガンマの2次元投影をプロット"""
    # 結果ディレクトリの作成
    os.makedirs("results", exist_ok=True)
    
    # MatplotlibでPlot
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Alpha vs Beta
    axs[0].scatter(data["alpha"], data["beta"], alpha=0.5)
    axs[0].set_title('Alpha vs Beta')
    axs[0].set_xlabel('Alpha')
    axs[0].set_ylabel('Beta')
    
    # Alpha vs Gamma
    axs[1].scatter(data["alpha"], data["gamma"], alpha=0.5)
    axs[1].set_title('Alpha vs Gamma')
    axs[1].set_xlabel('Alpha')
    axs[1].set_ylabel('Gamma')
    
    # Beta vs Gamma
    axs[2].scatter(data["beta"], data["gamma"], alpha=0.5)
    axs[2].set_title('Beta vs Gamma')
    axs[2].set_xlabel('Beta')
    axs[2].set_ylabel('Gamma')
    
    plt.tight_layout()
    plt.savefig(f"results/{data_name}_2d_projections.png")
    print(f"2D投影を保存しました: results/{data_name}_2d_projections.png")
    plt.close()
    
    # 標準化バージョンも作成
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(
        scaler.fit_transform(data),
        columns=["alpha", "beta", "gamma"]
    )
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Alpha vs Beta (標準化済み)
    axs[0].scatter(scaled_data["alpha"], scaled_data["beta"], alpha=0.5)
    axs[0].set_title('Alpha vs Beta (標準化済み)')
    axs[0].set_xlabel('Alpha')
    axs[0].set_ylabel('Beta')
    
    # Alpha vs Gamma (標準化済み)
    axs[1].scatter(scaled_data["alpha"], scaled_data["gamma"], alpha=0.5)
    axs[1].set_title('Alpha vs Gamma (標準化済み)')
    axs[1].set_xlabel('Alpha')
    axs[1].set_ylabel('Gamma')
    
    # Beta vs Gamma (標準化済み)
    axs[2].scatter(scaled_data["beta"], scaled_data["gamma"], alpha=0.5)
    axs[2].set_title('Beta vs Gamma (標準化済み)')
    axs[2].set_xlabel('Beta')
    axs[2].set_ylabel('Gamma')
    
    plt.tight_layout()
    plt.savefig(f"results/{data_name}_2d_projections_standardized.png")
    print(f"標準化2D投影を保存しました: results/{data_name}_2d_projections_standardized.png")
    plt.close()

def plot_personas(data_name):
    """ペルソナごとの特徴量をプロット"""
    # 結果ディレクトリの作成
    os.makedirs("results", exist_ok=True)
    
    # 使用するペルソナ数のリスト
    persona_nums = [5, 8, 12] if data_name == "NIPS" else [5, 25, 50]
    
    for persona_num in persona_nums:
        try:
            # 各ペルソナのデータを読み込む
            path_n = f"../optimize/complete/{data_name}/"
            
            # means（平均値）の読み込み
            path = path_n + f"persona={persona_num}/means.npy"
            means = np.load(path)
            
            # sigma（共分散）の読み込み
            path = path_n + f"persona={persona_num}/sigma.npy"
            sigmas = np.load(path)
            
            # 標準化
            scaler = StandardScaler()
            standardized_means = scaler.fit_transform(means)
            
            # 2Dプロット（標準化済み）
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            
            # 色のリスト
            colors = ["#FF4B00", "#03AF7A", "#F6AA00", "#4DC4FF", "#000000", "#FFF100"]
            
            # Alpha vs Beta
            for i in range(persona_num):
                color_idx = i % len(colors)
                axs[0].scatter(standardized_means[i, 0], standardized_means[i, 1], 
                               color=colors[color_idx], s=100, marker='o', 
                               label=f'ペルソナ {i+1}')
            
            axs[0].set_title(f'Alpha vs Beta - ペルソナ数 {persona_num}')
            axs[0].set_xlabel('Alpha (標準化済み)')
            axs[0].set_ylabel('Beta (標準化済み)')
            axs[0].legend()
            
            # Alpha vs Gamma
            for i in range(persona_num):
                color_idx = i % len(colors)
                axs[1].scatter(standardized_means[i, 0], standardized_means[i, 2], 
                               color=colors[color_idx], s=100, marker='o', 
                               label=f'ペルソナ {i+1}')
            
            axs[1].set_title(f'Alpha vs Gamma - ペルソナ数 {persona_num}')
            axs[1].set_xlabel('Alpha (標準化済み)')
            axs[1].set_ylabel('Gamma (標準化済み)')
            
            # Beta vs Gamma
            for i in range(persona_num):
                color_idx = i % len(colors)
                axs[2].scatter(standardized_means[i, 1], standardized_means[i, 2], 
                               color=colors[color_idx], s=100, marker='o', 
                               label=f'ペルソナ {i+1}')
            
            axs[2].set_title(f'Beta vs Gamma - ペルソナ数 {persona_num}')
            axs[2].set_xlabel('Beta (標準化済み)')
            axs[2].set_ylabel('Gamma (標準化済み)')
            
            plt.tight_layout()
            plt.savefig(f"results/{data_name}_persona{persona_num}_2d_standardized.png")
            print(f"ペルソナ{persona_num}の2Dプロットを保存しました: results/{data_name}_persona{persona_num}_2d_standardized.png")
            plt.close()
            
            # 3Dプロット
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # 各クラスタの中心点をプロット
            for i in range(persona_num):
                color_idx = i % len(colors)
                ax.scatter(means[i, 0], means[i, 1], means[i, 2], 
                           color=colors[color_idx], s=100, marker='o', 
                           label=f'ペルソナ {i+1}')
            
            ax.set_xlabel('Alpha')
            ax.set_ylabel('Beta')
            ax.set_zlabel('Gamma')
            
            ax.set_title(f'{data_name} - ペルソナ数={persona_num} - 3次元特徴量')
            ax.legend()
            plt.tight_layout()
            
            plt.savefig(f"results/{data_name}_persona{persona_num}_3d.png")
            print(f"ペルソナ{persona_num}の3Dプロットを保存しました: results/{data_name}_persona{persona_num}_3d.png")
            plt.close()
        
        except Exception as e:
            print(f"ペルソナ{persona_num}の処理でエラーが発生しました: {str(e)}")

if __name__ == "__main__":
    # データセット名を指定して実行
    data_name = "NIPS"
    print(f"{data_name}データセットの可視化を開始します...")
    main(data_name)
    print("可視化が完了しました。resultsフォルダ内のファイルを確認してください。") 