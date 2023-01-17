from pretopo.pretopo_base_clustering import Pretopocluster
import numpy as np
import pandas as pd
import pacmap
from prince import FAMD

def process(df, **kwargs):
    #n_components=int(np.sqrt(df.shape[1]))
    #n_components = min([df.shape[0],df.shape[1],10])
    numerical = df.select_dtypes(exclude='object')
    for c in numerical.columns:
        numerical[c] = (numerical[c] - numerical[c].mean())/numerical[c].std(ddof=0)
        
    ##preprocessing categorical
    categorical = df.select_dtypes(include='object')
    categorical = pd.get_dummies(categorical)

    n_components=min([categorical.shape[1],numerical.shape[1],100,len(df),categorical.shape[0],numerical.shape[0]])

    #Embedding numerical & categorical
    fit1 = pacmap.PaCMAP(#random_state=12,
                n_neighbors=10,
                n_components=n_components).fit_transform(numerical)

    fit2 = pacmap.PaCMAP(distance='hamming', 
                    n_neighbors=10,
                    n_components=n_components).fit_transform(categorical)
    # intersection will resemble the numerical embedding more.

    # union will resemble the categorical embedding more.
    gamma = np.mean(np.std(numerical))/2

    #embedding = fit1 + fit2*gamma
    embedding = np.square(fit1) + gamma * fit2

    #um = pd.DataFrame(embedding.embedding_) # Each points' UMAP coordinate 
    um = pd.DataFrame(embedding)
    pretopo_clusters = Pretopocluster().fit_predict(np.asarray(um))

    # return clusters as a list
    clusters = pretopo_clusters.astype(str)

    import streamlit as st
    st.write("pacmap")
    st.write(pd.Series(clusters).value_counts())
    return clusters

def process2(df, **kwargs):
    famd = FAMD(n_components=len(df.columns)).fit_transform(df)
    fit = pacmap.PaCMAP(n_components=len(df.columns),apply_pca=False).fit_transform(famd)
    pm = pd.DataFrame(fit)
    pretopo_clusters = Pretopocluster().fit_predict(np.asarray(pm))

    # return clusters as a list
    clusters = pretopo_clusters.astype(str)

    import streamlit as st
    st.write("pacmapFAMD")
    st.write(pd.Series(clusters).value_counts())
    return clusters