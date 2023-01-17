from pretopo.pretopo_base_clustering import Pretopocluster
import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.manifold import MDS
import plotly.express as px

def process(df, **kwargs):
    
    numerical = df.select_dtypes(exclude='object')
    for c in numerical.columns:
        numerical[c] = (numerical[c] - numerical[c].mean())/numerical[c].std(ddof=0)
        
    ##preprocessing categorical
    categorical = df.select_dtypes(include='object')
    categorical = pd.get_dummies(categorical)

    n_components=min([categorical.shape[1],numerical.shape[1],len(df),numerical.shape[0], categorical.shape[0]])
    #n_components=3
    #Embedding numerical & categorical
    fit1 = umap.UMAP(#random_state=12,
                    #n_neighbors=int(np.log2(len(df))),
                    n_components=n_components).fit_transform(numerical)
    fit2 = umap.UMAP(metric='hamming', 
                    #n_neighbors=int(np.log2(len(df))),
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
    st.write("umap")
    st.write(pd.Series(clusters).value_counts())
    return clusters

from scipy.spatial.distance import cdist

def process2(df, **kwargs):
    numerical = df.select_dtypes(exclude='object')
    for c in numerical.columns:
        numerical[c] = (numerical[c] - numerical[c].mean())/numerical[c].std(ddof=0)
        
    ##preprocessing categorical
    categorical = df.select_dtypes(include='object')
    categorical = pd.get_dummies(categorical)

    n_components=min([categorical.shape[1],numerical.shape[1],len(df),numerical.shape[0], categorical.shape[0]])
    #n_components=3
    #Embedding numerical & categorical

    gamma = np.mean(np.std(numerical))/2
    distances = (cdist(numerical,numerical,'sqeuclidean')) + cdist(categorical,categorical,'hamming')*gamma

    fit1 = umap.UMAP(n_components=3,metric='precomputed').fit_transform(distances)

    pretopo_clusters = Pretopocluster().fit_predict(np.asarray(fit1))
    # return clusters as a list
    clusters = pretopo_clusters.astype(str)
    import streamlit as st
    st.write("umap2")
    st.write(pd.Series(clusters).value_counts())

    um = pd.DataFrame(fit1)
    um.columns = ['X','Y','Z']
    import plotly.express as px
    import streamlit as st
    fig = px.scatter_3d(um, 'X', 'Y', 'Z', color=clusters)
    st.plotly_chart(fig)
    return clusters


def process3cityblock(df, **kwargs):
    numerical = df.select_dtypes(exclude='object')
    for c in numerical.columns:
        numerical[c] = (numerical[c] - numerical[c].mean())/numerical[c].std(ddof=0)
        
    ##preprocessing categorical
    categorical = df.select_dtypes(include='object')
    categorical = pd.get_dummies(categorical)

    n_components=min([categorical.shape[1],numerical.shape[1],len(df),numerical.shape[0], categorical.shape[0]])
    #n_components=3
    #Embedding numerical & categorical

    gamma = np.mean(np.std(numerical))/2
    distances = (cdist(numerical,numerical,'braycurtis')) + cdist(categorical,categorical,'jaccard')

    fit1 = umap.UMAP(n_components=n_components,metric='precomputed').fit_transform(distances)

    pretopo_clusters = Pretopocluster().fit_predict(np.asarray(fit1))
    # return clusters as a list
    clusters = pretopo_clusters.astype(str)
    import streamlit as st
    st.write("umap3")
    st.write(pd.Series(clusters).value_counts())

    

    return clusters

from algos.distance_matrix import distance_matrix as d_matrix

def process_huang(df, **kwargs):
    distances = d_matrix(df, 'huang')
    n_components = min([100,df.shape[0], df.shape[1], len(df)])
    fit1 = umap.UMAP(n_components=n_components,metric='precomputed').fit_transform(distances)

    pretopo_clusters = Pretopocluster().fit_predict(np.asarray(fit1))
    # return clusters as a list
    clusters = pretopo_clusters.astype(str)
    import streamlit as st
    st.write("umap-huang")
    st.write(pd.Series(clusters).value_counts())


    model = MDS(n_components=3, dissimilarity='precomputed')
    emb = model.fit(distances).embedding_
    emb=pd.DataFrame(emb)
    emb.columns = ['X','Y','Z']
    fig = px.scatter_3d(emb, x="X",y="Y",z="Z", color=clusters)
    st.plotly_chart(fig)
    mds_dist = cdist(emb, emb)
    st.write(f"Avg Ratio: {np.nanmean(distances/mds_dist)}, Std: {np.nanstd(distances/mds_dist)}")
    

    return clusters