import prince
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from pretopo.pretopo_base_clustering import Pretopocluster
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd


### TEST
import gower

def process(df, **kwargs):
    df2 = df.copy()
    if 'cluster' in df2.columns:
        df2.pop('cluster')
    numerical = df2.select_dtypes('number')
    categorical = df2.select_dtypes('object')

    # Scaling
    scaler = StandardScaler()
    numerical = scaler.fit_transform(numerical)
    categorical = categorical.apply(lambda x: x.replace(x.unique(),list(range(1,1+len(x.unique())))))

    # Gamma parameter to compute pairwise distances
    gamma = np.mean(np.std(numerical))/2

    # Compute pairwise distance matrix
    distances = (cdist(numerical,numerical,'sqeuclidean')) + cdist(categorical,categorical,'hamming')*gamma

    #distances = np.nan_to_num(distances)
    #for i in range(len(distances[0])):
    #    for j in range(len(distances)):
    #        distances[i][j] = 1-distances[i][j]
    #distances = np.nan_to_num(distances)

    #g_mat = 1-gower.gower_matrix(df)

    kernel = pd.DataFrame(distances).apply(lambda x: np.exp(-x/1))
    kernel[kernel < np.quantile(kernel, .50)] = 0


    #kernel = np.exp(-(distances**2))

    

    ###### LAPLACIAN EMBEDDINGS
    #lap = SpectralEmbedding(df.shape[1],affinity="precomputed").fit_transform(np.interp(distances, (distances.min(), distances.max()), (0, +1)))
    lap = SpectralEmbedding(df.shape[1],affinity="precomputed").fit_transform(kernel)


    # TEST
    #lap = SpectralEmbedding(df.shape[1],affinity="precomputed").fit_transform(g_mat)
    
    pretopo_clusters = Pretopocluster().fit_predict(np.asarray(lap))

    # return clusters as a list
    clusters = pretopo_clusters.astype(str)
    import streamlit as st
    st.write("Laplacian")
    st.write(pd.Series(clusters).value_counts())
    #import streamlit as st
    #import plotly.express as px
#
    #g = pd.DataFrame(lap)
    #g.columns = ['X','Y','Z']
    #
    #fig = px.scatter_3d(g, 
    #            x='X',y='Y',z='Z',color=clusters)
    #fig.update_layout(showlegend=False)
    #fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    #st.plotly_chart(fig)

    #st.write(pd.Series(clusters).nunique())
#
    #st.write()
    return clusters