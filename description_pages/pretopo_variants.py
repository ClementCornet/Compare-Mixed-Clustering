import streamlit as st
from data_generation.gen_data import generate_data
from pretopo.FAMD_Pretopo import process
import prince
import plotly.express as px


def pretopo_variants_page():

    st.title("Pretopological Clustering")

    st.markdown("""
    As we aim to develop the best possible pretopological clustering algorithm, we should test different variations,
    and compare their results. The varying features are :

    Preprocessing to adapt to mixed Data:
    - FAMD (and possible scaling (TODO) depending on components' inertia )
    - Laplacian Eigenmaps
    - Use of Gower's Distance
    - Use of K-Prototype's Distance (TODO)
    - ...

    Algorithmic features: (TODO)
    - Scaling to make the used area equal to 1 (changes size of the square length)
    - Change square length calculation
    - Largeron / Mine method
    - Degree
    - ...

    """)


    # SAMPLE GRAPH
    df = generate_data(
        n_clusters=3,
        n_num=15,
        n_cat=15,
        cat_unique=3,
        n_indiv=1000,
        clust_std=0.08
    )
    famd = prince.FAMD(3) # 3 components = 3D, to plot
    famd = famd.fit(df) # Last column is clusters, so it must not affect FAMD coordinates (just color)
    reduced = famd.row_coordinates(df) # Get coordinates of each row
    reduced.columns = ['X','Y','Z']
    reduced['cluster'] = process(df)#['cluster']

    st.write("Sample Graph : ")
    fig = px.scatter_3d(reduced, 'X', 'Y', 'Z', color='cluster')
    fig.update_layout(margin=dict(r=0,t=0,b=0,l=0))
    st.plotly_chart(fig)

    
