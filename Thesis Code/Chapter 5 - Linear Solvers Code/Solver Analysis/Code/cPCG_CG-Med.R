#!/usr/bin/env Rscript
library(Matrix)
library(microbenchmark)

if (!require("cPCG", quietly = TRUE)) {
    install.packages("cPCG")
    library(cPCG)
}


generate_D <- function(p) {
  return(diag(runif(p, 0.5, 5)))
}

generate_X <- function(n, p) {
  return(matrix(runif(n * p, 0, p), nrow = n, ncol = p))
}

simulate <- function(n, p, tol, repetitions) {
  performance_times <- numeric(repetitions)
  for (i in 1:repetitions) {
    set.seed(i)
    m <- c((1/2)^(1:min(p, 5)), rep(0, max(0, p - 5)))
    delta <- n
    sigma <- 3
    maxIter = 10*max(n,p)
    
    X <- generate_X(n, p)
    y <- X %*% m + rnorm(n, 0, sigma)
    
    D <- generate_D(p)
    A <- t(X) %*% X + delta * D
    b <- t(X) %*% y
    
    start_time <- proc.time()
    cg_solution <- cgsolve(A, b, tol = tol, maxIter = maxIter)
    end_time <- proc.time()

    time_taken <- end_time - start_time
    user_time_taken <- time_taken["user.self"]
    
    performance_times[i] <- as.numeric(user_time_taken)
  }
  return (list(mean = mean(performance_times), sd = sd(performance_times)))
}

# Simulation parameters
args <- commandArgs(trailingOnly = TRUE)
tol <- 1e-5
repetitions <- 100
n <- as.numeric(args[1])
p <- as.numeric(args[2])

data <- simulate(n, p, tol, repetitions)
results <- data.frame(
    solver = "Conjugate Gradient",
    n = n,
    p = p,
    time = data$mean,
    sdTime = data$sd)

# Save results to CSV file
filename <- sprintf("r-cPCG_CG-Med-n-%d-p-%d.csv", n, p)

write.csv(results, filename, row.names = TRUE)
