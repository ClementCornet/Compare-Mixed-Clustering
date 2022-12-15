from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler
from algos.utils import elbow_method

def process(df, **kwargs):
    # Get the number of clusters to process
    k=None
    # Using Elbow Method
    if 'k' not in kwargs.keys():
        k = elbow_method(df)
    # From kwargs
    else:
        k = kwargs['k']

    numerical_columns = df.select_dtypes('number').columns
    categorical_columns = df.select_dtypes('object').columns
    categorical_indexes = []

    # Scaling
    scaler = StandardScaler()
    for c in categorical_columns:
        categorical_indexes.append(df.columns.get_loc(c))
    if len(numerical_columns) == 0 or len(categorical_columns) == 0:
        return
    # create a copy of our data to be scaled
    df_scale = df.copy()
    # standard scale numerical features
    for c in numerical_columns:
        df_scale[c] = scaler.fit_transform(df[[c]])

    # Process Data
    kproto = KPrototypes(n_clusters=k)

    kproto.fit_predict(df_scale, categorical=categorical_indexes)

    # add clusters to dataframe
    df["cluster"] = kproto.labels_
    return df