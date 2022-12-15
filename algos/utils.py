import gower
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

# DEBUG
#import matplotlib.pyplot as plt
#import streamlit as st

def elbow_method(df):
    """
    Performs Elbow Method to determine the optimal number of clusters in a dataframe. To fit mixed data,
    Gower's distance matrix ise used. The metric used as scoring parameter is the Calinski-Harabasz index.

    Parameters:
        df (pandas.DataFrame): DataFrame to perform the Elbow Method on

    Returns:
        k (int): Optimal number of clusters in df
    
    """
    g_mat = gower.gower_matrix(df)
    #st.write(g_mat.shape)

    model = KElbowVisualizer(KMeans(), k=10, distance_metric='precomputed',metric='calinski_harabasz')

    model.fit(g_mat)
    model.show()

    #fig, ax = plt.subplots()
    
    #st.pyplot(fig)
    #st.write(model.elbow_value_)
    #st.write(model.elbow_score_)
    #st.write(model.locate_elbow)

    return model.elbow_value_
    