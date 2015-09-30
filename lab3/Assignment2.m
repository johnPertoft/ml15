addpath('data');

hand = imread('hand', 'ppm');
book = imread('book', 'ppm');

data1 = normalize_and_label(hand, 0);
data2 = normalize_and_label(book, 1);
test_data = [data1; data2];

[M, N] = size(test_data);
p = prior(test_data);
[mu, sigma] = bayes(test_data);
g = discriminant(test_data(:, 1:2), mu, sigma, p);

% for each row, which class had the highest posterior
[~, class] = max(g, [], 2);
class = class - 1;
error_test = 1.0 - sum(class == test_data(:, end)) / M;
error_test


plotDecisionBoundary = true;
if (plotDecisionBoundary)
  ax = [0.2 0.5 0.2 0.45];
  axis(ax);
  x = ax(1):0.01:ax(2);
  y = ax(3):0.01:ax(4);
  [z1, z2] = meshgrid(x, y);
  z1 = reshape(z1, size(z1,1)*size(z1,2), 1);
  z2 = reshape(z2, size(z2,1)*size(z2,2), 1);
  g = discriminant([z1 z2], mu, sigma, p);
  
  % decision boundary where class 1 is bigger than class 2
  gg = g(:,1) - g(:,2);
  gg = reshape(gg, length(y), length(x));
  [c,h] = contour(x, y, gg, [0.0 0.0]);
  set(h, 'LineWidth', 3);
end

showBook = true;
if (showBook)
  book_rg = zeros(size(book,1), size(book,2), 2);
  for y=1:size(book,1)
    for x=1:size(book,2)
      s = sum(book(y,x,:));
      if (s > 0)
        book_rg(y,x,:) = [double(book(y,x,1))/s double(book(y,x,2))/s];
      end
    end
  end
  
  tmp = reshape(book_rg, size(book_rg, 1) * size(book_rg, 2), 2);
  g = discriminant(tmp, mu, sigma, p);
  gg = g(:, 1) - g(:, 2);
  gg = reshape(gg, size(book_rg, 1), size(book_rg, 2));
  mask = gg < 0;
  mask3D(:, :, 1) = mask;
  mask3D(:, :, 2) = mask;
  mask3D(:, :, 3) = mask;
  result_im = uint8(double(book) .* mask3D);
  figure;
  imagesc(result_im);
  figure;
  imagesc(book);
end