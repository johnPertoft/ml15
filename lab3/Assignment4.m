hand = imread('hand', 'ppm');
book = imread('book', 'ppm');

data1 = normalize_and_label(hand, 0);
data2 = normalize_and_label(book, 1);
test_data = [data1; data2];

% train boosted classifier
T = 6;
[M, ~] = size(test_data);
[mu, sigma, priors, alpha, classes] = adaboost(test_data, T);
class = adaboost_discriminant(test_data(:, 1:end-1), mu, sigma, priors, alpha, classes);
boost_error_test = 1.0 - sum(class == test_data(:, end)) / M;

boost_error_test

% plot the points
figure;
hold on;
plot(data2(:,1), data2(:,2), '.');
plot(data1(:,1), data1(:,2), '.r');
legend('Hand holding book', 'Hand');

% plot the 95 % confidence intervals
theta = 0:0.01:2*pi;
for ci = 1:T
    x1 = 2 * sigma(1, 1, ci) * cos(theta) + mu(1, 1, ci);
    y1 = 2 * sigma(1, 2, ci) * sin(theta) + mu(1, 2, ci);
    x2 = 2 * sigma(2, 1, ci) * cos(theta) + mu(2, 1, ci);
    y2 = 2 * sigma(2, 2, ci) * sin(theta) + mu(2, 2, ci);
    plot(x1, y1, 'r');
    plot(x2, y2, 'b');
    
end


% plot the decision boundary
ax = [0.2 0.5 0.2 0.45];
axis(ax);
x = ax(1):0.01:ax(2);
y = ax(3):0.01:ax(4);
[z1, z2] = meshgrid(x, y);
z1 = reshape(z1, size(z1,1) * size(z1,2), 1);
z2 = reshape(z2, size(z2,1) * size(z2,2), 1);
g = adaboost_discriminant([z1 z2], mu, sigma, priors, alpha, classes);
gg = reshape(g, length(y), length(x));
[c,h] = contour(x, y, gg, [0.5 0.5]);
set(h, 'LineWidth', 3);

tmp = reshape(book_rg, size(book_rg,1)*size(book_rg,2), 2);
g = adaboost_discriminant(tmp, mu, sigma, priors, alpha, classes);
gg = reshape(g, size(book_rg,1), size(book_rg,2));
mask = gg > 0.5;

mask3D(:,:,1) = mask;
mask3D(:,:,2) = mask;
mask3D(:,:,3) = mask;
result_im = uint8(double(book) .* mask3D);
figure;
imagesc(result_im);