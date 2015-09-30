addpath('data');

hand = imread('hand', 'ppm');
book = imread('book', 'ppm');

data1 = normalize_and_label(hand, 0);
data2 = normalize_and_label(book, 1);
test_data = [data1; data2];

hold on;
plot(data2(:, 1), data2(:, 2), '.');
plot(data1(:, 1), data1(:, 2), '.r');

[mu, sigma] = bayes(test_data);
theta = 0:0.01:2*pi;
x1 = 2 * sigma(1,1) * cos(theta) + mu(1,1);
y1 = 2 * sigma(1,2) * sin(theta) + mu(1,2);
x2 = 2 * sigma(2,1) * cos(theta) + mu(2,1);
y2 = 2 * sigma(2,2) * sin(theta) + mu(2,2);
hold on;
plot(x1, y1, 'r');
plot(x2, y2);