import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import r

from rpy2.robjects.conversion import localconverter
import numpy as np
import pandas as pd

def distance_matrix(df, metric='ahmad'):
    """
    Returns the pairwise distance matrix from a dataframe with mixed types.
    Uses distmix() from R package `kmed`.

    Parameters:
        df (pd.DataFrame): The DataFrame to process. Should contain both numerical and categorical features. 
        Some distances use 'binary' features, where some features only have 2 distinct values.

        metric (str): The distance to compute. Available distances are: Gower, Huang, Podani, 
        Harikumar,Wishart and Ahmad & Dey.

    Return:
        dist_matrix (numpy.array):  The pairwise distance of the dataframe
    """

    utils = importr('utils')
    utils.chooseCRANmirror(ind=1)
    utils.install_packages('kmed')
    
    with localconverter(ro.default_converter + pandas2ri.converter):
        kmed = importr('kmed')
        num_ids = []
        cat_ids = []
        bin_ids = []
        for i,col in enumerate(df.columns):
            if df[col].nunique() <= 2:
                bin_ids.append(i+1)
                continue
            elif np.issubdtype(df[col].dtype, np.number):
                num_ids.append(i+1)
                continue
            else:
                cat_ids.append(i+1)

        dist_matrix = kmed.distmix(
                                    data=df, 
                                    method=metric,  
                                    idbin=ro.r("NULL") if len(bin_ids)==0 else ro.IntVector(bin_ids),
                                    idnum=ro.r("NULL") if len(num_ids)==0 else ro.IntVector(num_ids),
                                    idcat=ro.r("NULL") if len(cat_ids)==0 else ro.IntVector(cat_ids)
                                )
    return dist_matrix



if __name__ == '__main__':
    num1 = [1,2,3]
    num2 = [4,5,6]
    cat1 = ['A','B','C']
    cat2 = ['D','E','F']
    df = pd.DataFrame()
    df['cat1'] = cat1
    df['cat2'] = cat2
    df['num1'] = num1
    df['num2'] = num2
    print(
        distance_matrix(df, 'harikumar')
    )