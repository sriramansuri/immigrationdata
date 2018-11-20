import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt0
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import matplotlib.pyplot as plt4
import matplotlib.pyplot as plt5
import matplotlib.pyplot as plt6
import matplotlib.pyplot as plt7
import matplotlib.pyplot as plt8
import matplotlib.pyplot as plt9
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing

# Importing the data
df = pd.read_csv("h1b_kaggle.csv")

#Converted the categorical varibales to category datatype
df["SOC_NAME"] = df["SOC_NAME"].astype('category')
df["EMPLOYER_NAME"] = df["EMPLOYER_NAME"].astype('category')
df["JOB_TITLE"] = df["JOB_TITLE"].astype('category')
df["WORKSITE"] = df["WORKSITE"].astype('category')

# Created a copy of the data
obj_df = df[["SOC_NAME","EMPLOYER_NAME","JOB_TITLE","WORKSITE","FULL_TIME_POSITION","lon","lat", "PREVAILING_WAGE", "YEAR"]].copy()

# Removed all null values
obj_df = obj_df.dropna()

# One Hot Encoding
obj_df = pd.get_dummies(obj_df, columns=["FULL_TIME_POSITION"])

# Lable Encoding
obj_df["SOC_NAME"] = obj_df["SOC_NAME"].cat.codes
obj_df["EMPLOYER_NAME"] = obj_df["EMPLOYER_NAME"].cat.codes
obj_df["JOB_TITLE"] = obj_df["JOB_TITLE"].cat.codes
obj_df["WORKSITE"] = obj_df["WORKSITE"].cat.codes

# Normalization
min_max_scalar = preprocessing.MinMaxScaler()
x_scaled = min_max_scalar.fit_transform(obj_df)

# Variance Plot for PCA
pca = PCA().fit(x_scaled)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

# Running PCA to obtain 5 dimensions
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(x_scaled)
pcaDf = pd.DataFrame(data = principalComponents, columns = ['pc 1', 'pc 2','pc 3','pc 4','pc 5'])

# Elbow method analysis for K-Means
X = pcaDf
# k means determine k
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
 
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# Running K-Means Algorithm to obtain 3 Clusters
kmeans = KMeans(n_clusters=3).fit(pcaDf)
labels = kmeans.predict(pcaDf)
C = kmeans.cluster_centers_


# K-Means Visualization
plt0.scatter(pcaDf['pc 1'], pcaDf['pc 2'], c=labels, s=50, cmap='viridis')
plt0.xlabel('pc 1')
plt0.ylabel('pc 2')
plt0.title('K Means Clustering H1b Dataset PC1 & PC2')
plt0.savefig('pc12.png')