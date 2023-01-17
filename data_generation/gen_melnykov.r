print("oauaiohriuazhrgfiuzerhg")

library(kamila)

args <- commandArgs(trailingOnly = TRUE)
print(args)

dat <- genMixedData(
                    sampSize = as.numeric(args[1]),
                    nConVar = as.numeric(args[2]),
                    nCatVar = as.numeric(args[3]),
                    nCatLevels = as.numeric(args[4]),
                    nConWithErr = as.numeric(args[5]),
                    nCatWithErr = as.numeric(args[6]),
                    popProportions = c(as.numeric(args[7]) / 100, 1 - (as.numeric(args[7]) / 100)), # nolint
                    conErrLev = as.numeric(args[8]) / 100,
                    catErrLev = as.numeric(args[9]) / 100
                )

df <- cbind(dat$conVars, dat$catVars, dat$trueID)
write.csv(df, "melnykov.csv", row.names = FALSE)