
functions {
    int[] clip_difference(int[] x, int[] y, int low, int hi) {
        int n = size(x);
        int clipped_difference[n];
        for (i in 1:n) {
            clipped_difference[i] = min(max(x[i] - y[i], low), hi);
        }
        return clipped_difference;
    }
}

data {
    int<lower = 0> num_teams;

    // train data
    int<lower = 0> num_train_examples;

    int<lower = 0> train_home_goals[num_train_examples];
    int<lower = 0> train_away_goals[num_train_examples];

    int<lower = 1, upper = num_teams> train_home_ix[num_train_examples];
    int<lower = 1, upper = num_teams> train_away_ix[num_train_examples];

    // test data
    int<lower = 0> num_test_examples;

    int<lower = 1, upper = num_teams> test_home_ix[num_test_examples];
    int<lower = 1, upper = num_teams> test_away_ix[num_test_examples];
}

parameters {
    real intercept;
    real home_advantage;
    vector[num_teams] attack;
    vector[num_teams] defence;
    real<lower = 0.0> sigma_teams;
}

model {
    vector[num_train_examples] train_home_log_rate = intercept +
        home_advantage +
        attack[train_home_ix] -
        defence[train_away_ix];

    vector[num_train_examples] train_away_log_rate = intercept +
        attack[train_away_ix] -
        defence[train_home_ix];

    intercept ~ std_normal();
    home_advantage ~ std_normal();
    attack ~ normal(0.0, sigma_teams);
    defence ~ normal(0.0, sigma_teams);
    sigma_teams ~ std_normal();

    train_home_goals ~ poisson_log(train_home_log_rate);
    train_away_goals ~ poisson_log(train_away_log_rate);
}

generated quantities {
    vector[num_test_examples] test_home_log_rate = intercept +
        home_advantage +
        attack[test_home_ix] -
        defence[test_away_ix];

    vector[num_test_examples] test_away_log_rate = intercept +
        attack[test_away_ix] -
        defence[test_home_ix];

    int rep_test_home_goals[num_test_examples] = poisson_log_rng(test_home_log_rate);
    int rep_test_away_goals[num_test_examples] = poisson_log_rng(test_away_log_rate);

    int away_draw_home[num_test_examples] = clip_difference(
        rep_test_home_goals, rep_test_away_goals, -1, 1);
}
