function [mu, sigma, priors, alpha, classes] = adaboost(data, T)

labels = data(:, end);
classes = unique(labels);

[M, N] = size(data(:, 1:end-1));
C = length(classes);

% allocate space
mu = zeros(C, N, T);
sigma = zeros(C, N, T);
priors = zeros(T, C);
alpha = zeros(1, T);

% initialise weights
ws = ones(M, 1) / M;

for t = 1:T
    % train weak classifier with current weights
    [mu_t, sigma_t] = bayes_weight(data, ws);
    mu(:, :, t) = mu_t;
    sigma(:, :, t) = sigma_t;
    priors(t, :) = prior(data, ws);
    
    % compute error of weak classifier
    g = discriminant(data(:, 1:end-1), mu_t, sigma_t, priors(t, :));
    [~, class] = max(g, [], 2);
    class = class - 1;
    correct = class == labels; % In two class case: 1 if correct guess, else 0
    err_t = 1.0 - sum(ws .* correct);
    alpha(t) = 0.5 * log((1 - err_t) / err_t);
    
    % TODO: error should never be above 50%, break loop?
    
    % update weights and normalise
    alpha_sign = -1 * (correct * 2 - 1); % negative if correct else positive
    ws = ws .* exp(alpha(t) * alpha_sign);
    ws = ws ./ sum(ws);
end

end

