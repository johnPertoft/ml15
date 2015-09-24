function [mu, sigma] = bayes_weight(data, W)
% assuming classes are in the last column
labels = data(:, end);
classes = unique(labels);
X = data(:, 1:end-1);

mu = zeros(length(classes), size(X, 2));
sigma = zeros(length(classes), size(X, 2)); 
ci = 1;

for c = classes'
    Xc = X(labels == c, :);
    Wc = W(labels == c);
    Wc = Wc / sum(Wc);
    
    mu(ci, :) = sum(bsxfun(@times, Xc, Wc));
    sigma(ci, :) = sqrt(sum(bsxfun(@times, (bsxfun(@minus, Xc, mu(ci, :))).^2, Wc)));
    
    ci = ci + 1;
end

end

