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

  real average_degree;
  real beta_average;
  real beta_std;
  real<lower=0> gamma_;
  real<lower=0> b;
}

transformed data {
  real theta_0 = 0;
}

parameters {
  unit_vector[2] restricted_vertex;
  array[N-2] unit_vector[2] vertex_positions;
  array[N] real<lower=1e-10> kappa;
  real<lower=1> beta_;
}

transformed parameters {
  real<lower=-pi(), upper=pi()> theta_1 = atan2(restricted_vertex[2], restricted_vertex[1]);
  array[N-2] real<lower=-pi(), upper=pi()> theta_;

  for (i in 1:N-2) {
    theta_[i] = atan2(vertex_positions[i][2], vertex_positions[i][1]);
  }
}

model {
  beta_ ~ normal(beta_average, beta_std);
  kappa ~ cauchy(0, gamma_);
  real radius_div_mu = N * average_degree / ( beta_ * sin(pi() / beta_));

  int k=1;
  int shift_i = 0;
  for (i in 1:N-1) {

    real theta_i;
    if (i==fixed_vertices[1]) {
      theta_i = theta_0;
      shift_i+=1;
    }
    else if (i==fixed_vertices[2]) {
      theta_i = theta_1;
      shift_i+=1;
    }
    else {
      theta_i = theta_[i-shift_i];
    }

    int shift_j = shift_i;
    for (j in i+1:N) {

      real theta_j;
      if (j==fixed_vertices[1]) {
        theta_j = theta_0;
        shift_j+=1;
      }
      else if (j==fixed_vertices[2]) {
        theta_j = theta_1;
        shift_j+=1;
      }
      else {
        theta_j = theta_[j-shift_j];
      }

      if (edge[k]==1) {
        target += -log1p_exp(beta_*log(radius_div_mu*angular_distance(theta_i, theta_j, b)/kappa[i]/kappa[j]));
      }
      else {
        target += -log1p_exp(-beta_*log(radius_div_mu*angular_distance(theta_i, theta_j, b)/kappa[i]/kappa[j]));
      }
      k+=1;
    }
  }
}

generated quantities {
  array[N] real theta;
  int shift_i=0;

  for (i in 1:N) {
    if (i==fixed_vertices[1]) {
      theta[i] = theta_0;
      shift_i+=1;
    }
    else if (i==fixed_vertices[2]) {
      theta[i] = theta_1;
      shift_i+=1;
    }
    else {
      theta[i] = theta_[i-shift_i];
    }
  }
}
