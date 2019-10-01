import pandas as pd
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

'''
Script that applies dimensionality reduction techniques to visualize dependencies between different states
'''

path = "data/markets_new.csv"
df = pd.read_csv(path)

# Drop rows with at least one missing value
df_cut_nan = df.dropna()

df_cut_nan = df_cut_nan.drop("Date", axis=1)

# Compute returns
returns = df_cut_nan.pct_change().dropna()

# scale value between zero and one
scaler = MinMaxScaler()
df_cut_nan_min_max = scaler.fit_transform(df_cut_nan)

scaler = StandardScaler()
df_cut_nan_std = scaler.fit_transform(df_cut_nan)


def compute_reduction_and_plt(data, reduction):
    pca = reduction(n_components=3)
    transformed = pca.fit_transform(data)

    plt.plot(transformed, 'o', markersize=2)
    plt.show()

    plt.scatter(transformed[:, 0], transformed[:, 1], c=np.arange(0, data.shape[0]), s=2)
    plt.show()

    plt.scatter(transformed[:, 1], transformed[:, 2], c=np.arange(0, data.shape[0]), s=2)
    plt.show()

    plt.scatter(transformed[:, 0], transformed[:, 2], c=np.arange(0, data.shape[0]), s=2)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], c=np.arange(0, data.shape[0]), s=2)
    plt.show()


compute_reduction_and_plt(df_cut_nan_min_max, NMF)
compute_reduction_and_plt(df_cut_nan_std, PCA)
compute_reduction_and_plt(df_cut_nan_std, TSNE)
