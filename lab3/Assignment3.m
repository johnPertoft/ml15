addpath('data');

hand = imread('hand', 'ppm');
book = imread('book', 'ppm');

data1 = normalize_and_label(hand, 0);
data2 = normalize_and_label(book, 1);
test_data = [data1; data2];

[M, N] = size(test_data);
p = prior(test_data);
[mu, sigma] = bayes(test_data);
[mu_w, sigma_w] = bayes_weight(test_data, ones(M, 1) / M);

disp('Difference of mu and mu_w with equal uniform weights');
mu - mu_w