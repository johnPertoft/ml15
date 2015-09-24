% train boosted classifier
T = 6;
[M, ~] = size(test_data);
[mu, sigma, priors, alpha, classes] = adaboost(test_data, T);
class = adaboost_discriminant(test_data(:, 1:end-1), mu, sigma, priors, alpha, classes);
boost_error_test = 1.0 - sum(class == test_data(:, end)) / M;

boost_error_test

figure;
hold on;
plot(data2(:,1), data2(:,2), '.');
plot(data1(:,1), data1(:,2), '.r');
legend('Hand holding book', 'Hand');
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