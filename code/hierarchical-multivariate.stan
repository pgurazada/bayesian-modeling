data {
    int<lower=0> N; // number of individuals
    int<lower=1> K; // number of sample level predictors
    int<lower=1> J; // number of groups
    int<lower=1> L; // number of group level predictors
    
    int<lower=1, upper=J> jj[N]; // group memmership for each of the individuals
    matrix[N, K] X; // individual predictors design matrix
    row_vector[L] u[J]; //group predictors

    vector[N] y;
}

parameters {
    corr_matrix[K] Omega; // prior correlation
    vector<lower=0>[K] tau; // prior scale
    matrix[L, K] gamma; //group coeffs
    vector[K] beta[J];
    real<lower=0> sigma;
}

model {
    tau ~ cauchy(0, 2.5);
    Omega ~ lkj_corr(2);
    to_vector(gamma) ~ normal(0, 5);

    {
        row_vector[K] u_gamma[J];
        
        for (j in 1:J) {
            u_gamma[j] = u[j] * gamma;
        }

        beta ~ multi_normal(u_gamma, quad_form_diag(Omega, tau));

    }

    {

        vector[N] x_beta_jj;

        for (n in 1:N) {
            x_beta_jj[n] = x[n] * beta[jj[n]];
        } 

        y ~ normal(x_beta_jj, sigma);

    }


}
