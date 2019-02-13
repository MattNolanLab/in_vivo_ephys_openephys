

get_aic <- function(fit, M){
    #install.packages("useful")
    #install.packages("movMF")

    pars <- 2*seq(1:M) + 0:(M-1)
    cov  <- 1:M
    print(cov)
    aic  <- -2*(loess(unlist(lapply(fit, function(x) "[["(x, "L")))~cov)$fitted - pars)

    library(ggplot2)
    plot(aic, type="l", ylab="AIC", xlab="Number of components (or modes??)")
    gc()
    return(aic)
    }


    #  convert python df to double and convert coordinates to cart

    cart_coord <- function(hd){
    hd_double <- data.frame(lapply(hd, as.double))
    hd_double     <- cbind(hd_double, rep(1, nrow(hd)))
    library(useful)
    hd_cart     <- (pol2cart(r=hd_double[,2], theta=hd_double[,1]))[,1:2]
    return(hd_cart)
    }


    # Next, I fitted mixture models with 1,2,., 30 models and computed the
    # AIC in order to decide what model fits the data best from the given
    # class of models. The code for the fitting part is
    ## fitting mvM

     fit_mixed_models <- function(cartcoords, M){
         library(movMF)
         #M <- 12
         fit <- vector('list', M)

         for(i in 1:M){
              fit[[i]] <- movMF(as.matrix(cartcoords), k=i, control=list(maxiter=1000))
         }

         return(fit)

     }


     get_fit_value <- function(i, cartcoords){
        library(movMF)
        fit <- vector('list', 1)
        fit[[1]] <- movMF(as.matrix(cartcoords), k=i, control=list(maxiter=1000))
        return(fit)}


