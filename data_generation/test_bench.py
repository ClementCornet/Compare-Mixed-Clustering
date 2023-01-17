import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import r

from rpy2.robjects.conversion import localconverter

import sys
class recursion_depth:
    def __init__(self, limit):
        self.limit = limit
        self.default_limit = sys.getrecursionlimit()
    def __enter__(self):
        sys.setrecursionlimit(self.limit)
    def __exit__(self, type, value, traceback):
        sys.setrecursionlimit(self.default_limit)
with recursion_depth(500000):


    with localconverter(ro.default_converter + pandas2ri.converter):
        #base = importr('base')
        kam = importr('kamila')
        ee = kam.genMixedData(sampSize= 1000, nConVar=2, nCatVar=2, nCatLevels=4, nConWithErr=1, nCatWithErr=1,
                    popProportions=ro.FloatVector([0.2,0.8]), conErrLev=0.5, catErrLev=0.1)