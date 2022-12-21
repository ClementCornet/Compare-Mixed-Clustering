from pretopo.pretopo_base_clustering import Pretopocluster
import numpy as np
import pandas as pd
import umap.umap_ as umap

def process(df, **kwargs):
    n_components=3
    numerical = df.select_dtypes(exclude='object')
    for c in numerical.columns:
        numerical[c] = (numerical[c] - numerical[c].mean())/numerical[c].std(ddof=0)
        
    ##preprocessing categorical
    categorical = df.select_dtypes(include='object')
    categorical = pd.get_dummies(categorical)
    #Embedding numerical & categorical
    fit1 = umap.UMAP(random_state=12,
                    n_components=n_components).fit(numerical)
    fit2 = umap.UMAP(metric='dice', 
                    #n_neighbors=15,
                    n_components=n_components).fit(categorical)
    # intersection will resemble the numerical embedding more.

    # union will resemble the categorical embedding more.
    gamma = np.mean(np.std(numerical))/2

    #embedding = fit1 + fit2*gamma
    embedding = fit1.embedding_ + gamma * fit2.embedding_

    #um = pd.DataFrame(embedding.embedding_) # Each points' UMAP coordinate 
    um = pd.DataFrame(embedding)
    pretopo_clusters = Pretopocluster().fit_predict(np.asarray(um))

    # return clusters as a list
    clusters = pretopo_clusters.astype(str)
    return clusters