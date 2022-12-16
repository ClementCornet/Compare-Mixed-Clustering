import streamlit as st
import extra_streamlit_components as stx

from data_generation.gen_data import generate_data

import algos.kproto
import pretopo.FAMD_Pretopo

import prince
from sklearn.metrics import calinski_harabasz_score

import plotly.express as px
import plotly.graph_objects as go

def comp_page():
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
        FAMD_Pretopo=pretopo.FAMD_Pretopo.process
    )
    comp.compare(param)


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
        'n_indiv':[30,100,250,500,1000,2500]
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

    def compare(self, df_feature):
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

            df = Comparator.gen_from_str(df_feature, val)

            for name,algo in self.algos.items():
                output = algo(df)
                indices = Comparator.internal_indices(df, output)
                self.results[name].append(indices)

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

        return indices