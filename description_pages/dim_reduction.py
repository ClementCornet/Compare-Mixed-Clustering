import streamlit as st
import extra_streamlit_components as stx
from data_generation.gen_data import generate_data
import pandas as pd
from prince import FAMD
import plotly.express as px

from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import numpy as np
import umap.umap_ as umap
import prince
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

def dim_reduction_page():
    tech = stx.tab_bar(data=[
        stx.TabBarItemData(id='famd',title='FAMD',description=''),
        stx.TabBarItemData(id='lap',title='Laplacian Eigenmaps',description=''),
        stx.TabBarItemData(id='umap',title='UMAP',description=''),
        stx.TabBarItemData(id='pacmap',title='PaCMAP',description=''),
        stx.TabBarItemData(id='comp',title='Compare',description=''),
    ])

    if tech == 'famd':
        explain_famd()
    elif tech == 'lap':
        explain_laplacian()
    elif tech == 'umap':
        explain_umap()
    elif tech == 'pacmap':
        explain_pacmap()
    elif tech == 'comp':
        compare_dimr()

def explain_famd():
    st.write('Factorial Analysis of Mixed Data')
    st.write("""
        Factorial method
        Combination of a PCA on numerical features and MCA on categorical data.
        Objective function : sum of objectives of PCA and MCA
    """)
    tabs = st.tabs(['Penguins', 'Heart Failure', 'Generated'])
    with tabs[0]:
        df = pd.read_csv('data_generation/penguins.csv').dropna()
        st.write("344 individuals, 3 categorical features + 5 numerical")
        FAMD_plot(df)
    with tabs[1]:
        df = pd.read_csv('data_generation/heart_failure_short.csv').dropna()
        df['DEATH_EVENT'] = df['DEATH_EVENT'].astype(str)
        st.write('299 individuals, 3 categorical features + 3 numerical')
        FAMD_plot(df)
    with tabs[2]:
        df = generate_data(
            n_clusters=3,
            clust_std=0.1,
            n_num=50,
            n_cat=50,
            cat_unique=5,
            n_indiv=800
        )
        st.write('800 individuals, 50 categorical features + 50 numerical')
        st.write('3 generated clusters, 0.1 clust_std, 5 cat_unique')
        FAMD_plot(df)

def FAMD_plot(df):
    famd = FAMD(3)
    famd = famd.fit(df)
    reduced = famd.row_coordinates(df)
    reduced.columns = ['X', 'Y', 'Z']
    tot_inertia = f"{round(100*famd.explained_inertia_.sum(),2)}"
    st.write(f'FAMD Visualization of Clusters ({tot_inertia}%) :')
    labs = {
        "X" : f"Component 0 - ({round(100*famd.explained_inertia_[0],2)}% inertia)",
        "Y" : f"Component 1 - ({round(100*famd.explained_inertia_[1],2)}% inertia)",
        "Z" : f"Component 2 - ({round(100*famd.explained_inertia_[2],2)}% inertia)",
    }
    fig = px.scatter_3d(reduced, 
                    x='X',y='Y',z='Z',
                    labels=labs)
    fig.update_layout(showlegend=False)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    st.plotly_chart(fig)

def explain_laplacian():
    st.write('Spectral Embedding technique')
    st.markdown("""
    - Get pairwise distance matrix : Huang's Distance
        * $$Huang = sqeuclidean_{num} + \gamma \  hamming_{cat} $$  
        where $\gamma$ is a proportion of the average std of numerical features, usually 1/2
    - Build adjacency matrix : multiple solutions, often heat kernel  
        * $$affinity = exp(-distance/t) $$  
          where $t$ is user defined  
        * This matrix $W$ represents the weights of a graph
    - Get graph Laplacian
        * $D$ = sum of weight by node
        * Graph Laplacian $$ L = D - W $$
    - Eigenvalue decomposition of $L$
    """)
    tabs = st.tabs(['Penguins', 'Heart Failure', 'Generated'])
    with tabs[0]:
        t_penguins = st.number_input('t used in heat kernel (1) :',value=1.0)
        df = pd.read_csv('data_generation/penguins.csv').dropna()
        st.write("344 individuals, 3 categorical features + 5 numerical")
        laplacian_plot(df, t_penguins)
    with tabs[1]:
        t_hf = st.number_input('t used in heat kernel (2) :',value=1.0)
        df = pd.read_csv('data_generation/heart_failure_short.csv').dropna()
        df['DEATH_EVENT'] = df['DEATH_EVENT'].astype(str)
        st.write('299 individuals, 3 categorical features + 3 numerical')
        laplacian_plot(df, t_hf)
    with tabs[2]:
        t_gen = st.number_input('t used in heat kernel (3) :',value=1.0)
        df = generate_data(
            n_clusters=3,
            clust_std=0.1,
            n_num=50,
            n_cat=50,
            cat_unique=5,
            n_indiv=800
        )
        st.write('800 individuals, 50 categorical features + 50 numerical')
        st.write('3 generated clusters, 0.1 clust_std, 5 cat_unique')
        laplacian_plot(df, t_gen)

def laplacian_plot(df, t):
    
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
    distances = np.nan_to_num(distances)
    kernel = pd.DataFrame(distances).apply(lambda x: np.exp(-x/t))
    lap = SpectralEmbedding(3,affinity="precomputed").fit_transform(kernel)
    lap = pd.DataFrame(lap)
    lap.columns = ['X', 'Y', 'Z']
    fig = px.scatter_3d(lap, 'X', 'Y', 'Z')
    st.plotly_chart(fig)

def explain_umap():
    st.markdown("""
    Idea : Initial Embedding + Optimization by moving datapoints iteratively. Preserve Local Structure
    - One important hyperparameter $k$
    - Get pairwise Distance Matrix (Huang)
    - Get Affinity Matrix
        * Variation of heat kernel, $$similarity_{AB} = exp(-(rawdist-nearestneighbor)/\sigma)$$
        where $\sigma$ is adjusted for each node, to make sure its total weight = $log_2(k)$
        * $AB \\neq BA$ : $edge_{AB} = (AB+BA) - AB \\times BA$
    - Spectral Embedding (could be random, but spectral embedding gives faster convergence)
    - Optimization, for a datapoint:
        * The $k$ nearest points are considered as neighbors
        * Randomly select 1 neighbor and one not-neighbor points
        * For those 2 points caluculate low-dimensional score  
            $$score = \\frac{1}{1+\\alpha d^{2\\beta}}$$  
            usually, $\\alpha = 1.577$ and $\\beta = 0.8951$
        * Compute Cost  
        $$ Cost = log(\\frac{1}{neighbor}) - log(\\frac{1}{1-notneighbor})$$
        * Move our point using Stochastic Gradient Descent to find optimal position in the low-dimensional space
    """)
    tabs = st.tabs(['Penguins', 'Heart Failure', 'Generated'])
    with tabs[0]:
        k_penguins = st.number_input('k (1) :',value=15)
        df = pd.read_csv('data_generation/penguins.csv').dropna()
        st.write("344 individuals, 3 categorical features + 5 numerical")
        UMAP_plot(df, k_penguins)
    with tabs[1]:
        k_hf = st.number_input('k (2) :',value=15)
        df = pd.read_csv('data_generation/heart_failure_short.csv').dropna()
        df['DEATH_EVENT'] = df['DEATH_EVENT'].astype(str)
        st.write('299 individuals, 3 categorical features + 3 numerical')
        UMAP_plot(df, k_hf)
    with tabs[2]:
        k_gen = st.number_input('k (3) :',value=15)
        df = generate_data(
            n_clusters=3,
            clust_std=0.1,
            n_num=50,
            n_cat=50,
            cat_unique=5,
            n_indiv=800
        )
        st.write('800 individuals, 50 categorical features + 50 numerical')
        st.write('3 generated clusters, 0.1 clust_std, 5 cat_unique')
        UMAP_plot(df, k_gen)

def UMAP_plot(df, k):
    numerical = df.select_dtypes(exclude='object')
    for c in numerical.columns:
        numerical[c] = (numerical[c] - numerical[c].mean())/numerical[c].std(ddof=0)
        
    ##preprocessing categorical
    categorical = df.select_dtypes(include='object')
    categorical = pd.get_dummies(categorical)

    gamma = np.mean(np.std(numerical))/2
    distances = (cdist(numerical,numerical,'sqeuclidean')) + cdist(categorical,categorical,'hamming')*gamma

    fit1 = umap.UMAP(n_components=3,n_neighbors=k,metric='precomputed').fit_transform(distances)

    um = pd.DataFrame(fit1)
    um.columns = ['X','Y','Z']
    fig = px.scatter_3d(um, 'X', 'Y', 'Z')
    st.plotly_chart(fig)


def explain_pacmap():
    st.markdown("""
    Idea : preserve both global and local structures.  
    Similar to UMAP, but some changes:
    - Initialize with PCA (more important that UMAP's initialization, changes results)
    - Define Near-Pairs and Further Pairs (like neighbors / not-neighbors in UMAP)
    - Define Mid-Near Pairs : Sample 6 obervations, chose the 2nd closest
    - Computes graph weights differently for different pairs, at different stages of the optimization (nb of iterations)
        * 1st stage (100 iterations) : $w_{NB}=2$, $w_{MN}: 1000 $ to $ 3$, $w_{FP}=1$
        * 2nd stage (100 iterations) : $w_{NB}=3$, $w_{MN}=3$, $w_{FP}=1$
        * 3rd stage (250 iterations) : $w_{NB}=1$, $w_{MN}=0$, $w_{FP}=1$
    - Cost :  
        $$w_{NB} \\times \\frac{d_{NB}}{10+d_{NB}}+ w_{MN} \\times \\frac{d_{MN}}{10000+d_{MN}}+ w_{FP} \\times \\frac{d_{FP}}{1+d_{FP}}$$
    - Stochastic Gradient Descent
    - Hyperparameters : n_neighbors, MN_ratio, FP_ratio
    """)
    tabs = st.tabs(['Penguins', 'Heart Failure', 'Generated'])
    with tabs[0]:
        df = pd.read_csv('data_generation/penguins.csv').dropna()
        st.write("344 individuals, 3 categorical features + 5 numerical")
        pacmap_plot(df)
    with tabs[1]:
        df = pd.read_csv('data_generation/heart_failure_short.csv').dropna()
        df['DEATH_EVENT'] = df['DEATH_EVENT'].astype(str)
        st.write('299 individuals, 3 categorical features + 3 numerical')
        pacmap_plot(df)
    with tabs[2]:
        df = generate_data(
            n_clusters=3,
            clust_std=0.1,
            n_num=50,
            n_cat=50,
            cat_unique=5,
            n_indiv=800
        )
        st.write('800 individuals, 50 categorical features + 50 numerical')
        st.write('3 generated clusters, 0.1 clust_std, 5 cat_unique')
        pacmap_plot(df)

import pacmap
def pacmap_plot(df):
    MN_ratio = 0.5
    df2 = df.copy()
    numerical = df2.select_dtypes(exclude='object')
    for c in numerical.columns:
        numerical[c] = (numerical[c] - numerical[c].mean())/numerical[c].std(ddof=0)
        
    ##preprocessing categorical
    categorical = df2.select_dtypes(include='object')
    categorical = pd.get_dummies(categorical)
    n_components=3
    #Embedding numerical & categorical
    fit1 = pacmap.PaCMAP(#random_state=12,
                    n_neighbors=10,
                    MN_ratio=MN_ratio,
                    n_components=n_components).fit_transform(numerical)

    fit2 = pacmap.PaCMAP(distance='hamming', 
                    n_neighbors=10,
                    MN_ratio=MN_ratio,
                    n_components=n_components).fit_transform(categorical)

    gamma = np.mean(np.std(numerical))/2
    embedding = np.square(fit1)+fit2*gamma

    um = pd.DataFrame(embedding) # Each points' PaCMAP coordinate 

    # Actual Plotting
    um.columns = ['X','Y','Z']
    #um['cluster'] = df['cluster'].astype(str)
    fig = px.scatter_3d(um, 
                    x='X',y='Y',z='Z')
    fig.update_layout(showlegend=False)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def compare_dimr():
    
    st.write("Palmer Penguins")
    df = pd.read_csv('data_generation/penguins.csv').dropna()
    comp_one_dataframe(df)

    st.write("Heart Failure")
    df = pd.read_csv('data_generation/heart_failure_short.csv').dropna()
    comp_one_dataframe(df)

    st.write("Generated Data")
    df = generate_data(
            n_clusters=25,
            clust_std=0.1,
            n_num=150,
            n_cat=150,
            cat_unique=37,
            n_indiv=500
        )
    comp_one_dataframe(df)
    
def comp_one_dataframe(df):
    ### UMAP
    numerical = df.select_dtypes('number')
    categorical = df.select_dtypes('object')
    scaler = StandardScaler()
    numerical = scaler.fit_transform(numerical)
    categorical = categorical.apply(lambda x: x.replace(x.unique(),list(range(1,1+len(x.unique())))))
    gamma = np.mean(np.std(numerical))/2
    distances = (cdist(numerical,numerical,'sqeuclidean')) + cdist(categorical,categorical,'hamming')*gamma
    umap_embedding = umap.UMAP(n_components=len(df.columns),metric="precomputed").fit_transform(distances)

    ### FAMD
    famd = prince.FAMD(len(df.columns))
    famd_embeddings = famd.fit_transform(df)

    ### Split PaCMAP
    #n_components = np.min([np.min(categorical.shape), np.min(numerical.shape)])
    #fit1 = pacmap.PaCMAP(n_components=n_components).fit_transform(numerical)
    #fit2 = pacmap.PaCMAP(distance='hamming',n_components=n_components).fit_transform(categorical)
    #pacmap_embeddings = np.square(fit1)+fit2*gamma

    ### FAMD PaCMAP
    famd_pacmap_embeddings=pacmap.PaCMAP(n_components=len(df.columns), apply_pca=False).fit_transform(famd_embeddings)

    # Evaluate
    results = pd.DataFrame()
    umap_results = []
    famd_results = []
    pacmap_results = []
    #mds_pacmap_results = []
    for k in range(2,20):
        model = KMeans(n_clusters=k)
        clusters = model.fit_predict(umap_embedding)
        umap_results.append(davies_bouldin_score(umap_embedding, clusters))
        famd_results.append(davies_bouldin_score(famd_embeddings, clusters))
        pacmap_results.append(davies_bouldin_score(famd_pacmap_embeddings, clusters))
        #mds_pacmap_results.append(davies_bouldin_score(mds_pacmap_embeddings, clusters))
    results['UMAP'] = umap_results
    results['FAMD'] = famd_results
    results['PaCMAP'] = pacmap_results
    #results['MDS-PaCMAP'] = mds_pacmap_results
    results.index += 2
    st.dataframe(results)