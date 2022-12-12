import streamlit as st

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