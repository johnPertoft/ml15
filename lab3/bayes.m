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
    Mi = size(Xc, 1);
    mu(ci, :) = mean(Xc);
    %sigma(ci, :) = std(Xc); % gives wrong answer
    sigma(ci, :) = sqrt(sum(bsxfun(@minus, Xc, mu(ci, :)) .^ 2) ./ Mi);
    
    ci = ci + 1;
end

end

