import streamlit as st

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
        clust_std=2,
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