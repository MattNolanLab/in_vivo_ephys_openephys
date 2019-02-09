# I started by loading the data

X <- read.csv("/home/ipapasta/Desktop/M13_2018_05_09_c17.csv")
# and brought them to radians using

X[,1] <- X[,1]*pi/180
X     <- cbind(X, rep(1, nrow(X)))

# next I converted to cartesian co-ordinates in order to pass the data of right type to function movMF

library(useful)
cartcoords <- (pol2cart(r=X[,2], theta=X[,1]))[,1:2]

# Next, I fitted mixture models with 1,2,…, 30 models and computed the
# AIC in order to decide what model fits the data best from the given
# class of models. The code for the fitting part is

## fitting mvM
library(movMF)
M <- 30
fit <- vector('list', M)
for(i in 1:M)
    fit[[i]] <- movMF(as.matrix(cartcoords), k=i, control=list(maxiter=1000))

# Unfortunately, the reported log-likelihood does not vary smoothly over the number
# of mixture components because models are fitted by Expectation-Maximization
# (or for reasons unknown to me but potentially related to the package?). To alleviate from
# the jiggly behaviour of the likelihood, I used a local regression in order to smooth the
# likelihood values. This part, together with the computation and the plotting of the AIC is

pars <- 2*seq(1:M) + 0:(M-1)
cov  <- 1:M
aic  <- -2*(loess(unlist(lapply(fit, function(x) "[["(x, "L")))~cov)$fitted - pars)
plot(aic, type="l", ylab="AIC", xlab="Number of components (or modes?)")