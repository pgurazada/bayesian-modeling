data {
    int<lower=0> N; //samples
    vector[N] X;
    int<lower=0, upper=1> y[N];
}

parameters {
    real alpha;
    real beta;
}

model {
    y ~ bernoulli_logit(alpha + beta * x); // logit
    // y ~ bernoulli(Phi(alpha + beta * x)); //probit
}
