#importing datasets
import numpy as np
import matplotlib.pyplot as plot
from sklearn.cluster import KMeans
import pandas as pd
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,[8,12]].values
#find the optional number of cluster
wess = []
for i in range(1,16):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(X)
    wess.append(kmeans.inertia_)
plot.plot(range(1,16),wess)
plot.title('Elbow Method')
plot.xlabel('Number Of Clusters')
plot.ylabel('wess')
plot.show()
#kmeans clustering

kmeans = KMeans(n_clusters=4,init='k-means++',random_state=0)
y = kmeans.fit_predict(X)
plot.scatter(X[y==0,0],X[y==0,1],s=25,c='red',label='cluster1')
plot.scatter(X[y==1,0],X[y==1,1],s=25,c='blue',label='cluster2')
plot.scatter(X[y==2,0],X[y==2,1],s=25,c='green',label='cluster3')
plot.scatter(X[y==3,0],X[y==3,1],s=25,c='pink',label='cluster4')

plot.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=25,c='yellow',label='centroid')
plot.title('KMeans Clustering')
plot.xlabel('Balance')
plot.ylabel('Estimated Salary in $')
plot.legend()
plot.show()