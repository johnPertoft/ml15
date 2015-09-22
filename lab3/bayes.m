function [mu, sigma] = bayes(data)

% assuming classes are in the last column
labels = data(:, end);
classes = unique(labels);
X = data(:, 1:end-1);

mu = zeros(length(classes), size(X, 2));
sigma = zeros(length(classes), size(X, 2)); 
ci = 1;
for c = classes'
    Xc = X(labels == c, :);
    mu(ci, :) = mean(Xc);
    sigma(ci, :) = std(Xc);
    ci = ci + 1;
end

end

