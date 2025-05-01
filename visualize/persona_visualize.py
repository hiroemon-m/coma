#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import torch
import os

# データを読み込む関数
def load_data(data_name, persona_num):
    path_n = f"../optimize/complete/{data_name}/"
    
    # means（平均値）の読み込み
    path = path_n + f"persona={persona_num}/means.npy"
    means = np.load(path)
    
    # sigma（共分散）の読み込み
    path = path_n + f"persona={persona_num}/sigma.npy"
    sigmas = np.load(path)
    
    # pi（混合比率）の読み込み
    path = path_n + f"persona={persona_num}/pi.npy"
    pi = np.load(path)
    
    # スケーラーの読み込み（あれば）
    try:
        path = path_n + f"persona={persona_num}/scaler.pkl"
        scaler = joblib.load(path)
    except:
        scaler = None
    
    return means, sigmas, pi, scaler

# 2Dガウス分布を可視化する関数
def plot_2d_gaussian(means, sigmas, persona_num, data_name, feature_indices=(0, 1)):
    """
    2次元ガウス分布の可視化
    feature_indices: 可視化する特徴量のインデックス（例：(0,1)はalpha,betaを示す）
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#FF4B00", "#03AF7A", "#F6AA00", "#4DC4FF", "#000000", "#FFF100"]
    
    # 表示範囲の設定
    all_means = means[:, feature_indices]
    x_min, x_max = np.min(all_means[:, 0]) - 1, np.max(all_means[:, 0]) + 1
    y_min, y_max = np.min(all_means[:, 1]) - 1, np.max(all_means[:, 1]) + 1
    
    # メッシュグリッドの作成
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    
    # 各クラスタのガウス分布を描画
    for i in range(persona_num):
        mu = means[i, feature_indices]
        sigma = np.array([
            [sigmas[i, feature_indices[0], feature_indices[0]], sigmas[i, feature_indices[0], feature_indices[1]]],
            [sigmas[i, feature_indices[1], feature_indices[0]], sigmas[i, feature_indices[1], feature_indices[1]]]
        ])
        
        # 確率密度関数を計算
        rv = multivariate_normal(mu, sigma)
        pos = np.dstack((X, Y))
        Z = rv.pdf(pos)
        
        # 等高線で描画
        color_idx = i % len(colors)
        ax.contour(X, Y, Z, levels=5, colors=colors[color_idx], alpha=0.6)
        
        # 平均値をプロット
        ax.scatter(mu[0], mu[1], marker='x', color=colors[color_idx], s=100, label=f'ペルソナ {i+1}')
    
    # 軸ラベルの設定
    feature_names = ['Alpha', 'Beta', 'Gamma']
    ax.set_xlabel(feature_names[feature_indices[0]])
    ax.set_ylabel(feature_names[feature_indices[1]])
    
    ax.set_title(f'{data_name} - ペルソナ数={persona_num} - 2次元ガウス分布')
    ax.legend()
    plt.tight_layout()
    
    # 画像を保存
    os.makedirs("../visualize/results", exist_ok=True)
    plt.savefig(f"../visualize/results/{data_name}_persona{persona_num}_{feature_names[feature_indices[0]]}_{feature_names[feature_indices[1]]}.png")
    plt.show()

# 標準化したデータでガウス分布を可視化
def plot_standardized_gaussian(data_name, persona_num):
    # データの読み込み
    means, sigmas, pi, scaler = load_data(data_name, persona_num)
    
    # 標準化処理
    standardized_means = means.copy()
    standardized_sigmas = sigmas.copy()
    
    # StandardScalerを適用
    scaler = StandardScaler()
    scaler.fit(means)
    standardized_means = scaler.transform(means)
    
    print(f"= {data_name} ペルソナ数 {persona_num} の可視化（StandardScaler適用後） =\n")
    
    # Alpha vs Beta
    print("Alpha vs Beta:")
    plot_2d_gaussian(standardized_means, standardized_sigmas, persona_num, f"{data_name} (標準化済み)", (0, 1))
    
    # Alpha vs Gamma
    print("Alpha vs Gamma:")
    plot_2d_gaussian(standardized_means, standardized_sigmas, persona_num, f"{data_name} (標準化済み)", (0, 2))
    
    # Beta vs Gamma
    print("Beta vs Gamma:")
    plot_2d_gaussian(standardized_means, standardized_sigmas, persona_num, f"{data_name} (標準化済み)", (1, 2))

# 3Dプロットでガウス分布と特徴量を可視化
def plot_3d_gaussian(data_name, persona_num):
    # データの読み込み
    means, sigmas, pi, scaler = load_data(data_name, persona_num)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ["#FF4B00", "#03AF7A", "#F6AA00", "#4DC4FF", "#000000", "#FFF100"]
    
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
    
    # 画像を保存
    os.makedirs("../visualize/results", exist_ok=True)
    plt.savefig(f"../visualize/results/{data_name}_persona{persona_num}_3D.png")
    plt.show()

def main():
    # 実行例
    data_name = "NIPS"  # または "DBLP"
    persona_nums = [5, 8, 12]  # 可視化したいペルソナ数
    
    for persona_num in persona_nums:
        print(f"\n{data_name} ペルソナ数 {persona_num} の可視化")
        
        # 2Dガウス分布の可視化（標準化あり）
        plot_standardized_gaussian(data_name, persona_num)
        
        # 3D特徴量の可視化
        print(f"\n3次元特徴量:")
        plot_3d_gaussian(data_name, persona_num)

if __name__ == "__main__":
    main() 