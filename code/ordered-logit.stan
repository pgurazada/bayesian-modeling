data {
    int<lower=2> K; // number of classes
    int<lower=0> N; // number of samples
    int<lower=1> D; // number of predictors

    int<lower=1, upper=K> y[N];
    row_vector[D] x[N];
}

parameters {
    vector[D] beta;
    ordered[K-1] c; // vector of cutpoints
}

model {
    for (n in 1:N) {
        y[n] ~ ordered_logistic(x[n] * beta, c);
    }
}
