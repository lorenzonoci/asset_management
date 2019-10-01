from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
Script for weights analysis using clustering method based on non-negative matrix factorization (NNMF)
'''

path = "data/markets_new.csv"
weights_paths = ["results_1/all_weights_5.npy"]


df = pd.read_csv(path)

# Drop rows with at least one missing value
df_cut_nan = df.dropna()
df_cut_nan = df_cut_nan.drop("Date", axis=1)
scaler = MinMaxScaler()
df_cut_nan_min_max = scaler.fit_transform(df_cut_nan)

nmf = NMF(n_components=3, l1_ratio=1)
transformed = nmf.fit_transform(df_cut_nan_min_max)
components = nmf.components_
assign_trend = np.argmax(components, axis=0)

for weights_path in weights_paths:
    all_weights = np.load(weights_path)
    all_trends = []
    for weights in all_weights:
        trends = [0, 0, 0]
        for i, weight in enumerate(weights):
            trend = assign_trend[i]
            trends[trend] += weight
        all_trends.append(trends)

    df = pd.DataFrame(all_trends)
    df.columns = ["Trend 1", "Trend 2", "Trend 3"]
    ax = df.plot.bar(rot=0)
    ax.set_xlabel("time")
    ax.set_ylabel("weights")
    pd.DataFrame(all_weights).plot.bar()
    plt.show()