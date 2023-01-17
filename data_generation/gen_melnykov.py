from subprocess import check_output
import pandas as pd
import os

def gen_data_melnykov(
                sampsize=1000,
                nConVar = 2,
                nCatVar = 2,
                nCatLevels = 2,
                nConWithErr = 1,
                nCatWithErr = 1,
                popProportions = 50,
                conErrLev = 50,
                catErrLev = 10
                        ):


    cmd_args = [sampsize,
                nConVar,
                nCatVar,
                nCatLevels,
                nConWithErr,
                nCatWithErr,
                popProportions,
                conErrLev,
                catErrLev]
    cmd_args = [str(arg) for arg in cmd_args]
    cmd_args_str = " ".join(cmd_args)                  
    check_output(f"Rscript data_generation/gen_melnykov.r {cmd_args_str}", shell=True).decode()

    df = pd.read_csv("melnykov.csv")
    
    cols = []
    for i in range(nConVar):
        cols.append(f"num{i+1}")
    for i in range(nCatVar):
        cols.append(f"cat{i+1}")
        df.iloc[:,nConVar+i+1] = df.iloc[:,nConVar+i+1].astype(str)
    cols.append("target")
    df.columns = cols

    return df


if __name__ == "__main__":
    gen_data_melnykov()