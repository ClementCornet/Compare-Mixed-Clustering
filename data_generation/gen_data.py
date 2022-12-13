from sklearn.datasets import make_blobs
import pandas as pd

from sklearn.preprocessing import StandardScaler

### DEBUG
import streamlit as st
from scipy.spatial.distance import cdist


def clust_centers(n_dim, n_centers):
    """
    Get cluster centers for data generation. Counts in binary from 0 to 2**n_dim-1 (biggest number represented on
    n_dim bits) and takes the binary representation as coordinates. If additional centers are needed, just double the
    coordinates from the previously calculated ones.

    Parameters:
        n_dim (int) : number of dimensions of data
        n_centers (int) : number of centers to generate
    Returns:
        grid (2D list) : list of centers
    """

    # Count in Binary
    unary_grid = []
    for i in range(2**n_dim):
        unary_grid.append(
            [eval(digit) for digit in  [*format(i, f'#0{n_dim+2}b')[2:]]] # Get Binary representation and convert to list
            )
    if n_centers <= 2**n_dim:
        return unary_grid[:n_centers]

    # Double coordinates to get additional centers
    grid = unary_grid
    double_grid = [(pd.Series(elem) * 2).tolist() for elem in unary_grid[1:]]
    grid.extend(double_grid[:n_centers-2**n_dim])
    return grid


def generate_data(n_clusters=5,clust_std=0.2,n_num=2,n_cat=2,cat_unique=3,n_indiv=100):
    """
    Generates a Dataset to benchmark clustering algorithms

    Parameters:
        n_clusters (int): Number of clusters to generate
        clust_std (float): Standard deviation of data around cluster centers
        n_num (int): Number of Numerical features
        n_cat (int): Number of Categorical features
        cat_unique (int): Number of unique values taken by each categorical feature
        n_indiv (int): Number of individuals (rows) to generate
    
    Returns:
        df (pandas.DataFrame): Generated Dataset
    """
    # Compute Centers
    c_centers = clust_centers((n_num+n_cat), n_clusters)
    print(c_centers)

    # Generate Numerical Data with computed centers

    blobs = make_blobs(n_samples=n_indiv,
                      n_features=n_num+n_cat,
                      cluster_std=clust_std,
                      centers=n_clusters,
                      return_centers=True)
    df = pd.DataFrame(blobs[0])

    #print(blobs[1])
    #print("now centers:")
    print(cdist(blobs[2],blobs[2]))
    print(blobs[2])

    #for col in df.columns:
    #    df[col] = StandardScaler().fit_transform(df[[col]])

    #print(df.describe())

    # Discretize categorical features
    for i in range(n_cat):
        df.iloc[:,-i] = pd.qcut(df.iloc[:,i],cat_unique,labels=False, duplicates='drop').astype(str)

    return df