functions {
  real continuous_abs(real x, real b) {
    return x*(2/(1+exp(-b*x)) - 1);
  }
  real angular_distance(real theta1, real theta2, real b) {
    return pi()-continuous_abs(pi()-continuous_abs(theta1-theta2, b), b);
  }
}

data {
  int<lower=3> N;

  array[2] int<lower=1, upper=N> fixed_vertices;
  array[(N*(N-1))%/%2] int<lower=0, upper=1> edge;
  array[N] real<lower=-pi(), upper=pi()> theta;

  real radius_div_mu;
  real<lower=1> beta_;
  real<lower=0> gamma_;
  real<lower=0> b;
}

parameters {
  array[N] real<lower=1e-10> kappa;
}

model {
  kappa ~ cauchy(0, gamma_);
  int k=1;
  for (i in 1:N-1) {
    for (j in i+1:N) {
      if (edge[k]==1) {
        target += -log1p_exp(beta_*log(radius_div_mu*angular_distance(theta[i], theta[j], b)/kappa[i]/kappa[j]));
      }
      else {
        target += -log1p_exp(-beta_*log(radius_div_mu*angular_distance(theta[i], theta[j], b)/kappa[i]/kappa[j]));
      }
      k+=1;
    }
  }
}
