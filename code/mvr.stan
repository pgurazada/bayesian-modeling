data {
    int<lower=0> N; //number of samples
    int<lower=0> K; // number of predictors

    matrix[N, K] X;
    vector[N] y;
}

parameters {
    real alpha;
    vector[K] beta;
    real<lower=0> sigma;
}

model {
    y ~ normal(x * beta + alpha, sigma);
}
