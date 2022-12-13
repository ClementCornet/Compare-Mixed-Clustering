import prince
from pretopo.pretopo_base_clustering import Pretopocluster
import numpy as np

def process(df, *k):
    famd = prince.FAMD(3) # 3 components = 3D, to plot
    famd = famd.fit(df) # Last column is clusters, so it must not affect FAMD coordinates (just color)
    reduced = famd.row_coordinates(df) # Get coordinates of each row
    pretopo_clusters = Pretopocluster().fit_predict(np.asarray(reduced))
    df['cluster'] = pretopo_clusters.astype(str)
    return df