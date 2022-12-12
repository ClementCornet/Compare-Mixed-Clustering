import streamlit as st

def indices_descr_page():
    st.title('Evaluation Indices')

    st.write("""
    As we are using mixed data (with both categorical and numerical features), we cannot directly use classical
    indices. Although, we adapt some to mixed data, to stay able to evaluate how our algorithms perform.  

    ## Calinski-Harabasz Index

    Also known as the Variance Ratio Criterion. The index is the ratio of the sum of between-clusters dispersion
    and of within-cluster dispersion for all clusters (where dispersion is defined as the sum of distances squared)
    A higher Calinski-Harabasz score relates to a model with better defined clusters.  
    To compute the Calinski-Harabasz index, we use dimensionality reduction techniques such as FAMD or Laplacian
    Eigenmaps, to translate mixed data into numerical data.  
    Advantages :  
    - The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
    - The score is fast to compute.
    Drawbacks :
    - The Calinski-Harabasz index is generally higher for convex clusters than other concepts of clusters

    ## Silouhette Score

    The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster
    distance (b) for each sample. The Silhouette Coefficient for a sample is (b - a) / max(a, b). The best value is
    1 and the worst value is -1. Values near 0 indicate overlapping clusters. We use FAMD and Laplacian Eigenmaps
    to compute Silouhette Score, but also distances suited for mixed data such as K-Prototype's and Gower's Distances.

    ## Davies-Bouldin Score

    This index signifies the average 'similarity' between clusters, where the similarity is a measure that
    compares the distance between clusters with the size of the clusters themselves. Zero is the lowest possible score.
    Values closer to zero indicate a better partition. We also use FAMD and Laplacian Eigenmaps to compute this index.

    Advantages :
    - The computation of Davies-Bouldin is simpler than that of Silhouette scores.
    - The index is solely based on quantities and features inherent to the dataset as its computation only
    uses point-wise distances.
    Drawbacks :
    - Like Calinski-Harabasz index, Davies-Bouldin score is higher for convex clusters.

    """)