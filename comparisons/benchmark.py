import streamlit as st
from data_generation.gen_melnykov import gen_data_melnykov

import algos.kproto

def benchmark_page():
    st.title("BENCHMARK")

    cols = st.columns([2,5])
    with cols[0].container():
        get_data()

def get_data():
    data_source = st.radio("Data Source :", ["Generated", "Real World", "Upload"])
    if data_source == "Generated":
        df = melnykov_widget()
        st.dataframe(df)

def melnykov_widget():
    n_indiv = st.select_slider("Number of Individuals", [50,100,250,500,750,1000,1500,2000,2500],value=500)
    num_var = st.select_slider("Numerical Variables", [2,5,10,15,25,35,50,75,100,150,200],value=5)
    cat_var = st.select_slider("Categorical Variables", [2,5,10,15,25,35,50,75,100,150,200],value=5)
    cat_levels = st.slider("Categorical Levels",min_value=2,max_value=100,value=2,step=2)
    nConWithErr = st.slider("Numerical Variables with Error",min_value=1,max_value=num_var,value=1,step=1)
    nCatWithErr = st.slider("Categorical Variables with Error",min_value=1,max_value=cat_var,value=1,step=1)
    popProportions = st.slider("Proportion (%) of individuals in biggest cluster",min_value=50,max_value=99,value=50,step=1)
    conErrLev = st.slider("Univariate overlap (%) between clusters on the numerical variables specified to have error",
                    min_value=1,max_value=100,value=50, step=1)
    catErrLev = st.slider("Univariate overlap (%) between clusters on the categorical variables specified to have error",
                    min_value=1,max_value=100,value=10, step=1)

    df = gen_data_melnykov(
        sampsize = n_indiv,
        nConVar = num_var,
        nCatVar = cat_var,
        nCatLevels = cat_levels,
        nConWithErr = nConWithErr,
        nCatWithErr = nCatWithErr,
        popProportions = popProportions,
        conErrLev = conErrLev,
        catErrLev = catErrLev
    )

    return df