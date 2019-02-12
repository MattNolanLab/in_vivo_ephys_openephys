

count_modes <- function(hd){
    #install.packages("useful")
    #install.packages("movMF")
    X <- read.csv("C:/Users/s1466507/Documents/GitHub/in_vivo_ephys_openephys/OverallAnalysis/M13_2018_05_09_c17.csv")
    print(file.path(R.home("bin"), "R"))
    print(R.version)

    # and brought them to radians using
    print(str(X))
    print(typeof(X[1]))
    X <- data.frame(lapply(hd, as.double))
    print(str(X))
    print(typeof(X[1]))

    #print(str(Y))
    #print(typeof(Y[1]))
    #X <- data.frame(lapply(X, as.numeric))
    # and brought them to radians using
    X[,1] <- X[,1]*pi/180
    X     <- cbind(X, rep(1, nrow(X)))
    # next I converted to cartesian co-ordinates in order to pass the data of right type to function movMF
    library(useful)
    cartcoords <- (pol2cart(r=X[,2], theta=X[,1]))[,1:2]
    # Next, I fitted mixture models with 1,2,., 30 models and computed the
    # AIC in order to decide what model fits the data best from the given
    # class of models. The code for the fitting part is
    ## fitting mvM
    library(movMF)
    M <- 12
    fit <- vector('list', M)


    for(i in 1:M){
      fit[[i]] <- movMF(as.matrix(cartcoords), k=i, control=list(maxiter=1000))
        print(i)
        flush.console()
        }


    pars <- 2*seq(1:M) + 0:(M-1)
    cov  <- 1:M
    aic  <- -2*(loess(unlist(lapply(fit, function(x) "[["(x, "L")))~cov)$fitted - pars)

    library(ggplot2)
    plot(aic, type="l", ylab="AIC", xlab="Number of components (or modes??)")
    gc()
    return(aic)
    }