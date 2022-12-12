import streamlit as st
import streamlit.components.v1 as components

import plotly.express as px
import plotly.graph_objects as go



def methodo_page():
    st.title('Compare Mixed Clustering Algorithms')     

    st.markdown(
        """
        The aim of this app is to compare various clustering algorithms suited for mixed data, using multiple internal
        indices.
        The algorithms we compare (from `Clustering-Mixed-Data`) are:
        - K-Prototype
        - FAMD-KMeans
        - Hierarchical clustering with Gower's Distance
        - Kamila
        - MixtComp
        - Modha Spangler
        - Spectral Clustering with K-Prototype's Distance
        - UMAP-HDBSCAN
        - And of course Pretopological Clustering
        """)

    # EXAMPLE GRAPH
    algo_1,algo_2,algo_3 = [1,2,3,4],[4,2,1,3],[2,4,2,1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(2,6)),y=algo_1,name="Algo X"))
    fig.add_trace(go.Scatter(x=list(range(2,6)),y=algo_2,name="Algo Y"))
    fig.add_trace(go.Scatter(x=list(range(2,6)),y=algo_3,name="Algo Z"))
    fig.update_layout(title="Index AAA for different numbers of clusters")
    fig.update_xaxes(title="Varying Parameter")
    fig.update_yaxes(title="Index AAA")
    st.plotly_chart(fig)


    st.markdown("""
            As we want to develop a mixed-clustering algorithm using Pretopology, variations of pretopological clustering
            are also compared here (FAMD, Laplacian Eigenmaps, K-Prototype's Distance...).

            We use this app to determine which algorithm performs the best on which kind of datasets, using generated datasets.

            We then use internal indices to compare the different algorithms.  

            The parameter that are varying on our different datasets are :
            - Number of Clusters
            - Clusters Dispersion
            - Number of numerical and categorical features
            - Number of different values for each categorical feature
            - Number of individuals in the dataset

            """
        )