#!/usr/bin/env Rscript


#--------------------------------------------------------------------------#
#                            Collect arguments
#--------------------------------------------------------------------------#

args = commandArgs(trailingOnly = TRUE)


if (length(args) == 0) {
    stop("At least one argument must be supplied (input file).txt", call. = FALSE)
}

#--------------------------------------------------------------------------#
#                              FUNCTIONS
#--------------------------------------------------------------------------#

run_norm <- function(path_train_data, name_dir){
    data = read.table(path_train_data, header = F, sep = ",", row.names = NULL)
    colnames(data) <- NULL
    data = as.matrix(data)
    s <- prelim.norm(data) #do preliminary manipulations
    thetahat <- em.norm(s, showits = FALSE) #compute mle
    param <- getparam.norm(s, thetahat, corr = FALSE) #look at estimated correlations
    mu <- param$mu
    cov <- param$sigma
    write.table(cov, file = paste(name_dir, "cov.txt", sep = ""), sep = ",", row.names = FALSE, col.names = FALSE)
    write.table(mu, file = paste(name_dir, "mu.txt", sep = ""), sep = ",", row.names = FALSE, col.names = FALSE)
}



#--------------------------------------------------------------------------#
#                            Program
#--------------------------------------------------------------------------#

#if (!require("norm")) install.packages("norm")
suppressMessages(library(norm))

run_norm(args[1], args[2])
