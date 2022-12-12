import streamlit as st
import extra_streamlit_components as stx

def comp_page():
    param = stx.tab_bar(data=[
        stx.TabBarItemData(id='n_clust',title='Number of Clusters',description=''),
        stx.TabBarItemData(id='clust_std',title='Clusters Dispersion',description=''),
        stx.TabBarItemData(id='num_features',title='Numerical Features',description=''),
        stx.TabBarItemData(id='cat_features',title='Categorical Features',description=''),
        stx.TabBarItemData(id='cat_unique',title='Categorical Unique Values',description=''),
        stx.TabBarItemData(id='n_indiv',title='Number of individuals',description=''),
    ])