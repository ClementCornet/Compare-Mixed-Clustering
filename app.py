# Global imports
import streamlit as st
from streamlit_option_menu import option_menu

# Local imports
from description_pages.methodology import methodo_page
from description_pages.indices import indices_descr_page
from description_pages.pretopo_variants import pretopo_variants_page
from description_pages.gen_data_descr import gen_data_descr_page
from description_pages.dim_reduction import dim_reduction_page

from comparisons.comparison_page import comp_page_range, compare_punctual, compare_upload
from comparisons.benchmark import benchmark_page

st.set_page_config(page_title="Compare Mixed Clustering",layout="wide")

with st.sidebar:
    selected = option_menu("Comparisons", 
            [
             "Methodology",
             "Evaluation Indices",
             "Pretopological Clustering",
             "Data Generation",
             "Compare On Range",
             "Punctual Comparison",
             "Uploaded data Comparison",
             "Dimension Reductions",
             "Benchmark"
              ],
    menu_icon=None, default_index=1,orientation='vertical')


if selected == 'Methodology':
    methodo_page()
elif selected == "Evaluation Indices":
    indices_descr_page()
elif selected == "Pretopological Clustering":
    pretopo_variants_page()
elif selected == "Data Generation":
    gen_data_descr_page()
elif selected == "Compare On Range":
    comp_page_range()
elif selected == "Punctual Comparison":
    compare_punctual()
elif selected == "Uploaded data Comparison":
    compare_upload()
elif selected == "Dimension Reductions":
    dim_reduction_page()
elif selected == "Benchmark":
    benchmark_page()