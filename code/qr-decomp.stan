/*

It is recommended to use the QR decomposition for linear models since it is
much easier for the Markov chain to move in the QR reparametrization, compared
to the original feature space.

This is especially helpful if we do not have an informative prior on the location
of the regression coefficients

*/


data {
    int<lower=0> N; //number of samples
    int<lower=0> K; //number of predictors
    matrix[N, K] X; //design matrix
    vector[N] y; // outcome
}

transformed data {
    matrix[N, K] Q_ast; // Q* = sqrt(n-1) * Q
    matrix[K, K] R_ast; // R* = R/sqrt(n-1)
    matrix[K, K] R_ast_inv;

    Q_ast = qr_Q(X)[, 1:K] * sqrt(N-1);
    R_ast = qr_R(X)[1:K, ] / sqrt(N-1);
    R_ast_inv = inverse(R_ast);
}

parameters {
    real alpha;
    vector[K] theta;
    real<lower=0> sigma;
}

model {
    y ~ normal(Q_ast*theta + alpha, sigma);
}

generated quantitites {
    vector[K] beta;
    beta = R_ast_inv * theta;
}
