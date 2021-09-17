import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

genes = ['gene' + str(i) for i in range(1, 101)]
# print(genes)

wt = ['wt' + str(i) for i in range(1, 6)]     # wt = wild type samples
ko = ['ko' + str(i) for i in range(1, 6)]     # ko = knock out samples

data = pd.DataFrame(columns=[*wt, *ko], index=genes)
# here * unpacks the wt and ko arrays so that there is single array with all 12 column names
# without * there will be a single array made of 2 arrays
# print(data)

for gene in data.index:
    data.loc[gene, 'wt1': 'wt5'] = np.random.poisson(lam=rd.randrange(10, 1000), size=5)
    data.loc[gene, 'ko1': 'ko5'] = np.random.poisson(lam=rd.randrange(10, 1000), size=5)
# print(data)

scaled_data = preprocessing.scale(data.T)  # another method
# scaled_data = StandardScaler.fit_transform(data.T)
# NOTE: we have passed data.T becoz the scaling function expects the samples to be row not columns

pca = PCA()
pca.fit(scaled_data)  # this is where all the pca maths is done[i.e finding loading scores and variation for pc1,2 etc]
pca_data = pca.transform(scaled_data)  # here we generate coordinates for pca graph based on loading scores and
                                       # scaled data
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = ['PC' + str(x) for x in range(1 , len(per_var) + 1)]

plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of explained variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
# plt.show()


pca_df = pd.DataFrame(pca_data, index=[*wt, *ko], columns=labels)
# print(pca_df)

plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('My PCA graph')
plt.xlabel('PC1 - {0}'.format(per_var[0]))
plt.ylabel('PC2 - {0}'.format(per_var[1]))

for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))

# plt.show()

loading_scores = pd.Series(pca.components_[0], index=genes)
# print(loading_scores)
# [0] means PC1 thus above line means loading scores of PC1
sorted_loading_sccores = loading_scores.sort_values(ascending=False)
# print(sorted_loading_sccores)

top10_genes = sorted_loading_sccores[0:10].index.values
# print(top10_genes)

print(loading_scores[top10_genes])