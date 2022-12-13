import streamlit as st
import extra_streamlit_components as stx

def comp_page():
    """
    The page data that generates data, and evaluates performances of different algorithms on each
    clustering result.
    """


    # Let the User choose which parameter may vary
    param = stx.tab_bar(data=[
        stx.TabBarItemData(id='n_clust',title='Number of Clusters',description=''),
        stx.TabBarItemData(id='clust_std',title='Clusters Dispersion',description=''),
        stx.TabBarItemData(id='num_features',title='Numerical Features',description=''),
        stx.TabBarItemData(id='cat_features',title='Categorical Features',description=''),
        stx.TabBarItemData(id='cat_unique',title='Categorical Unique Values',description=''),
        stx.TabBarItemData(id='n_indiv',title='Number of individuals',description=''),
    ])

    grid = {
        'n_clust':[2,3,4,5,6,7,8,9,10],
        'clust_std':[0.05,0.1,0.15,0.20,0.25,0.3,0.35,0.4],
        'num_features':[2,3,4,5,6,7,8,9,10],
        'cat_features':[2,3,4,5,6,7,8,9,10],
        'cat_unique':[2,3,4,5,6,7,8],
        'n_indiv':[30,100,250,500,1000,2500]
    }

    st.write("TODO")