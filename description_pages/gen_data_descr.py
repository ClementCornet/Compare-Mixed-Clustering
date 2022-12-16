import streamlit as st
from data_generation.gen_data import generate_data
from prince import FAMD
import plotly.express as px

def gen_data_descr_page():
    st.title("How do we generate datasets?")

    st.markdown("""
    To generate our data, we first generate a numerical dataset using `sklearn.datasets.make_blobs()`
    It generates isotropic Gaussian blobs for clustering. Then, to generate categorical features too,
    we discretize some variables, using the 100%/c quantile as a treshold.  
    Data is generated with a given number of clusters (5 by default), with the cluster centers evenly
    spaced. Typically, the distance between two cluster centers is 1.  

    ```python
    def generate_data(
        n_clusters=5,
        clust_std=0.08,
        n_num=15,
        n_cat=15,
        cat_unique=3,
        n_indiv=250
    )
    ```
    An other interresting parameter used to generate data is the Cluster Standard Deviation, i.e. the
    dispersion of different variables around cluster centers.  
    We also generate data with different number of numerical and categorical features, and number of
    unique features for categorical features.  
    Note that this approach is quite similar to the one used by Costa, Papatsouma and Markos in their
    `Benchmarking distance-based partitioning methods for mixed-type data` article.
    """
    )

    cols = st.columns([1,1,1])
    with cols[0].container():
        st.write("Generate Data :")
        n_clust = st.slider('Number of Clusters',2,10,5)
    with cols[1].container():
        n_num = st.slider('Numerical Features',2,10,5)
        n_cat = st.slider('Categorical Features',2,10,5)
    with cols[2].container():
        clust_std = st.select_slider('Cluster Dispersion',[0.001,0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.5],0.05)
        n_indiv = st.select_slider('Number of individuals',[50,250,500,1000,1750,2500],250)

    df = generate_data(
        n_clusters=n_clust,
        n_num=n_num,
        n_cat=n_cat,
        n_indiv=n_indiv,
        clust_std=clust_std
        )

    famd = FAMD(n_components=3)
    rr = famd.fit_transform(df)
    rr.columns = ['X','Y','Z']

    fig = px.scatter_3d(rr,'X','Y','Z')
    fig.update_layout(margin=dict(r=0,l=0,b=0,t=0))
    st.plotly_chart(fig)