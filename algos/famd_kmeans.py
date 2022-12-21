import prince
import pandas as pd
from sklearn.cluster import KMeans
from algos.utils import elbow_method

def process(df, **kwargs):
    """Process K-Means of a Dataset's FAMD Coordinate"""

    k=None
    # Using Elbow Method
    if 'k' not in kwargs.keys():
        k = elbow_method(df)
    # From kwargs
    else:
        k = kwargs['k']

    # Get FAMD Coordinates
    famd = prince.FAMD(n_components=3) # Using 3 dimensions by default, could use more to maximize inertia
    famd = famd.fit(df)
    reduced = famd.row_coordinates(df)

    # Process Standard K-Means
    km = KMeans(n_clusters=k)
    km.fit(reduced)
    return km.labels_