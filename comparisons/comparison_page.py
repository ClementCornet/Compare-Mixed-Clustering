import streamlit as st
import extra_streamlit_components as stx
import pandas as pd
import numpy as np

from data_generation.gen_data import generate_data

import algos.kproto
import algos.famd_kmeans
import algos.kamila
import pretopo.FAMD_Pretopo
import pretopo.UMAP_pretopo
import pretopo.PaCMAP_preotopo
import pretopo.Laplacian_pretopo

import prince
import umap.umap_ as umap
import gower
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from scipy.spatial.distance import cdist
from sklearn.manifold import MDS

import plotly.express as px
import plotly.graph_objects as go

def comp_page_range():
    """
    The page that generates data, and evaluates performances of different algorithms on each
    clustering result.
    """


    # Let the User choose which parameter may vary
    param = stx.tab_bar(data=[
        stx.TabBarItemData(id='n_clusters',title='Number of Clusters',description=''),
        stx.TabBarItemData(id='clust_std',title='Clusters Dispersion',description=''),
        stx.TabBarItemData(id='n_num',title='Numerical Features',description=''),
        stx.TabBarItemData(id='n_cat',title='Categorical Features',description=''),
        stx.TabBarItemData(id='cat_unique',title='Categorical Unique Values',description=''),
        stx.TabBarItemData(id='n_indiv',title='Number of individuals',description=''),
    ])
    if param not in ['n_clusters','clust_std','n_num','n_cat','cat_unique','n_indiv']:
        st.info("Select a Paramater to compare the algorithms on")
        return

    comp = Comparator(
        KPrototypes=algos.kproto.process,
        FAMD_Pretopo=pretopo.FAMD_Pretopo.process,
        UMAP_Pretopo=pretopo.UMAP_pretopo.process,
        PaCMAP_Pretopo=pretopo.PaCMAP_preotopo.process
    )
    comp.compare_on_range(param)


def compare_upload():
    comp = Comparator(
            KPrototypes=algos.kproto.process,
            Kamila=algos.kamila.process,
            FAMD_KMeans=algos.famd_kmeans.process,
            FAMD_Pretopo=pretopo.FAMD_Pretopo.process,
            Laplacian_Pretopo=pretopo.Laplacian_pretopo.process,
            UMAP_Pretopo=pretopo.UMAP_pretopo.process,
            UMAP_Pretopo2=pretopo.UMAP_pretopo.process2,
            UMAP_Pretopo3=pretopo.UMAP_pretopo.process3cityblock,
            UMAP_huang = pretopo.UMAP_pretopo.process_huang,
            PaCMAP_Pretopo=pretopo.PaCMAP_preotopo.process,
            PaCMAP_Pretopo2=pretopo.PaCMAP_preotopo.process2
        )
    comp.uploaded_data_comparison()

def compare_punctual():
    cols = st.columns([1,3])
    #return
    with cols[0].container():
        n_indiv = st.select_slider('Number of individuals',[50,250,500,1000,1750,2500],250)
        n_clust = st.select_slider('Number of Clusters',[2,3,5,7,10,15,20],3)
        n_num = st.select_slider('Number of Numerical Features',[1,3,5,10,15,25,50,100,200],10)
        n_cat = st.select_slider('Number of Categorical Features',[1,3,5,10,15,25,50,100,200],10)
        cat_unique = st.select_slider('Unique values for Categorical Features',[2,3,5,7,10,50,100],3)
        clust_std = st.select_slider('Clusters Dispersion',[0.001,0.01,0.05,0.1,0.15,0.2,0.25,0.3],0.1)
        df = generate_data(n_indiv=n_indiv,n_clusters=n_clust,n_num=n_num,n_cat=n_cat,
                cat_unique=cat_unique,clust_std=clust_std)
    with cols[1].container():
        comp = Comparator(
            KPrototypes=algos.kproto.process,
            Kamila=algos.kamila.process,
            FAMD_KMeans=algos.famd_kmeans.process,
            FAMD_Pretopo=pretopo.FAMD_Pretopo.process,
            Laplacian_Pretopo=pretopo.Laplacian_pretopo.process,
            UMAP_Pretopo=pretopo.UMAP_pretopo.process,
            UMAP_Pretopo2=pretopo.UMAP_pretopo.process2,
            UMAP_Pretopo3=pretopo.UMAP_pretopo.process3cityblock,
            UMAP_huang = pretopo.UMAP_pretopo.process_huang,
            PaCMAP_Pretopo=pretopo.PaCMAP_preotopo.process,
            PaCMAP_Pretopo2=pretopo.PaCMAP_preotopo.process2
        )
        comp.punctual_comparison(df)

class Comparator:
    """
    Class Designed to compare different mixed-clustering algorithms on different generated datasets.

    """

    grid = {
        'n_clusters':[2,3,4,5,6,7,8,9,10],
        'clust_std':[0.001,0.01,0.08,0.1,0.15,0.2,0.3],
        'n_num':[2,5,7,10,15,22,30],
        'n_cat':[2,5,7,10,15,22,30],
        'cat_unique':[2,3,4,5,6,7,8],
        'n_indiv':[50,100,500,1000,2500]
    }

    def __init__(self, **algos):
        """
        Instantiate a `Comparator` instance, given algorithms to compare.

        Attributes:
            algos (dict): Algorithms to compare. Keys are names, Values are functions to call (usually
            module.process())
            results (dict): Results of the algorithms. Each Algorithm name is a key, and results are stored in a list
        """
        self.algos = algos
        self.results = {name:[] for name in algos.keys()}

    def compare_on_range(self, df_feature):
        """
        Compare algorithms (self.algos) on datasets which only differ on a given parameter. This parameter
        takes its values from the static dict `grid`.

        Parameters:
            df_feature (str): The name of the feature that has to change between each iteration. Has to be a
            key of static dict `grid`.

        Return:
            Void. But plots the results.

        """
        feat_values = Comparator.grid[df_feature]
        #st.write(f"number of clusters: {feat_values}")

        for val in feat_values:
            print(f"{df_feature} : {val}")

            df = Comparator.gen_from_str(df_feature, val)

            for name,algo in self.algos.items():
                output = algo(df)
                indices = Comparator.internal_indices(df, output)
                self.results[name].append(indices)
                with open("indices.txt", "a") as f:
                    f.write(f"{name} | {df_feature}={val} : {indices}\n")

        #st.write(self.results)

        # Plot FAMD-CH for our algorithms
        FAMD_CH = {}
        for k,v in self.results.items():
            #st.write(k)
            #st.write([elem["FAMD-CH"] for elem in self.results[k]])
            FAMD_CH[k]=[elem["FAMD-CH"] for elem in self.results[k]]
        fig = go.Figure()
        for k,v in FAMD_CH.items():
            fig.add_trace(
                go.Scatter(x=feat_values,y=v,name=k)
            )
        fig.update_layout(title=f"FAMD Calinski-Harabasz score, changing {df_feature}")
        st.plotly_chart(fig)

        # Plot UMAP-CH for our algorithms
        UMAP_CH = {}
        for k,v in self.results.items():
            #st.write(k)
            #st.write([elem["UMAP-CH"] for elem in self.results[k]])
            UMAP_CH[k]=[elem["UMAP-CH"] for elem in self.results[k]]
        fig2 = go.Figure()
        for k,v in UMAP_CH.items():
            fig2.add_trace(
                go.Scatter(x=feat_values,y=v,name=k)
            )
        fig2.update_layout(title=f"UMAP Calinski-Harabasz score, changing {df_feature}")
        st.plotly_chart(fig2)
  
    def gen_from_str(feature, value):
        """
        Allows Data Generation, calling args by name

        Parameters:
            feature (str): feature to chose the value
            value (numeric): value to give to the chosen feature

        Returns:
            df (pandas.DataFrame): Generate Dataframe, with default parameters except the one we chose

        """

        if feature == 'n_clusters':
            return generate_data(n_clusters=value)
        elif feature == 'clust_std':
            return generate_data(clust_std=value)
        elif feature == 'n_num':
            return generate_data(n_num=value)
        elif feature == 'n_cat':
            return generate_data(n_cat=value)
        elif feature == 'cat_unique':
            return generate_data(cat_unique=value)
        elif feature == 'n_indiv':
            return generate_data(n_indiv=value)
        else:
            raise ValueError(f"{feature} not implemented for gen_from_str")


    def internal_indices(df, clusters):
        """
        Computes internal evaluation indices for a given clustering. Those indices are :
        - Calinski-Harabasz on FAMD coordinates

        Parameters:
            df (pandas.DataFrame): DataFrame that is clustered
            clusters (list[int]): list of clusters labels output by a clustering algorithm

        Returns:
            indices (dict): Dictionary of various indices computed for the given data and clusters 
        """

        indices = {}

        # PROCESS FAMD, TO COMPUTE INDICES ON IT
        famd = prince.FAMD(3) # 3 components = 3D, to plot
        famd = famd.fit(df) # Last column is clusters, so it must not affect FAMD coordinates (just color)
        reduced = famd.row_coordinates(df) # Get coordinates of each row
        indices['FAMD-CH'] = calinski_harabasz_score(reduced, clusters) # Calinski-Harabasz on FAMD coordinates
        indices['FAMD-DB'] = davies_bouldin_score(reduced, clusters)
        indices['FAMD-Si'] = silhouette_score(reduced, clusters)

        # PROCESS UMAP
        n_components=3
        numerical = df.select_dtypes(exclude='object')
        for c in numerical.columns:
            numerical[c] = (numerical[c] - numerical[c].mean())/numerical[c].std(ddof=0)
        ##preprocessing categorical
        categorical = df.select_dtypes(include='object')
        categorical = pd.get_dummies(categorical)
        #Embedding numerical & categorical
        fit1 = umap.UMAP(#random_state=12,
                        n_components=n_components).fit(numerical)
        fit2 = umap.UMAP(metric='dice', 
                        #n_neighbors=15,
                        n_components=n_components).fit(categorical)
        # intersection will resemble the numerical embedding more.
        # union will resemble the categorical embedding more.
        embedding = fit1 + fit2
        um = pd.DataFrame(embedding.embedding_) # Each points' UMAP coordinate
        indices['UMAP-CH'] = calinski_harabasz_score(um, clusters) # Calinski-Harabasz on FAMD coordinates
        indices['UMAP-DB'] = davies_bouldin_score(um, clusters)
        indices['UMAP-Si'] = silhouette_score(um, clusters)


        g_mat = gower.gower_matrix(df)
        indices['Gower-Si'] = silhouette_score(
                 g_mat, 
                 clusters,
                 metric="precomputed")

        categorical = df.select_dtypes(include='object')
        categorical = pd.get_dummies(categorical)
        gamma = np.mean(np.std(numerical))/2
        # Compute pairwise distance matrix
        huang = (cdist(numerical,numerical,'sqeuclidean')) + cdist(categorical,categorical,'hamming')*gamma*2
        model = MDS(n_components=len(df.columns), dissimilarity='precomputed')
        emb = model.fit(huang).embedding_
        indices['MDS-CH'] = calinski_harabasz_score(emb, clusters) # Calinski-Harabasz on FAMD coordinates
        indices['MDS-DB'] = davies_bouldin_score(emb, clusters)
        indices['MDS-Si'] = silhouette_score(emb, clusters)


        return indices
    
    def punctual_comparison(self, df):
        self.results = {
            name:self.algos[name](df) for name in self.algos.keys() 
        }
        #for k,v in self.results.items():
        #    st.write(k)
        #    st.write(Comparator.internal_indices(df, v))

        indices = {k:Comparator.internal_indices(df, v) for k,v in self.results.items()}
        #st.write(indices)

        for index in indices[list(indices.keys())[0]]:
            #st.write(index)
            #st.write({k:v[index] for k,v in indices.items()})
            index_dict = {k:v[index] for k,v in indices.items()}

            fig = px.histogram(x=index_dict.keys(), y=index_dict.values(),title=index)
            fig.update_xaxes(title="Algorithm")
            fig.update_yaxes(title=index)
            st.plotly_chart(fig)

    def uploaded_data_comparison(self):

        up = st.file_uploader('Upload File')    
        st.session_state['truth'] = False
        if up:
            df = pd.read_csv(up).dropna()
        else:
            return
        self.results = {
            name:self.algos[name](df) for name in self.algos.keys() 
        }
        st.dataframe(df)
        #for k,v in self.results.items():
        #    st.write(k)
        #    st.write(Comparator.internal_indices(df, v))

        indices = {k:Comparator.internal_indices(df, v) for k,v in self.results.items()}
        #st.write(indices)

        for index in indices[list(indices.keys())[0]]:
            #st.write(index)
            #st.write({k:v[index] for k,v in indices.items()})
            index_dict = {k:v[index] for k,v in indices.items()}

            fig = px.histogram(x=index_dict.keys(), y=index_dict.values(),title=index)
            fig.update_xaxes(title="Algorithm")
            fig.update_yaxes(title=index)
            st.plotly_chart(fig)