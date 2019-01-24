data {
    int<lower=2> K; // number of classes
    int<lower=0> N; // number of samples
    int<lower=1> D; // number of predictors

    int<lower=1, upper=K> y[N];
    matrix[N, D] X;
}

parameters {
    matrix[D, K] beta; //each row will capture the probability distribution of the K classes

}

model {
    matrix[N, K] x_beta = X * beta;

    to_vector(beta) ~ normal(0, 2); //prior on beta

    for (n in 1:N) {
        y[n] ~ categorical_logit(x_beta[n]);
        // y[n] ~ categorical(softmax(x[n] * beta)); // softmax scaling instead of logit
    }
}
