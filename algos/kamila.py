import pandas as pd
import os
from subprocess import check_output
from algos.utils import elbow_method

import json

def process(df, **kwargs):
    """Process Kamila Clustering algorithm
    
    Using kamila R package, interacting with Rscript executable"""

    k=None
    # Using Elbow Method
    if 'k' not in kwargs.keys():
        k = elbow_method(df)
    # From kwargs
    else:
        k = kwargs['k']
    k=int(k)

    # Write categorical and numerical values to split CSV files, as kamila R package needs
    numerical_columns = df.select_dtypes('number').columns
    categorical_columns = df.select_dtypes('object').columns
    df[categorical_columns].to_csv('temp_cat.csv',index=False)
    df[numerical_columns].to_csv('temp_continue.csv',index=False)

    # Pass the desired number of clusters to R through a json file
    json_data = { "n_clusters" : k}
    with open('k.json', 'w') as f:
        json.dump(json_data, f)

    # Calling Rscript Executable to process data
    check_output("Rscript algos/kamila.R", shell=True).decode()

    df_out = pd.read_csv('temp_clustered.csv') # R stored clusters vector in a file

    #df['cluster'] = df_out['cluster']

    # Remove files created to interact with Rscript
    os.remove("temp_cat.csv")
    os.remove("temp_continue.csv")
    os.remove("temp_clustered.csv")
    os.remove("k.json")

    return df_out['cluster']