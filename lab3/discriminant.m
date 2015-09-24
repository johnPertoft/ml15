function g = discriminant(data, mu, sigma, priors)
% computes posterior probabilities
g = repmat(log(priors), [size(data, 1), 1]);
for ci = 1:length(priors)
    g(:, ci) = g(:, ci) - sum(log(sigma(ci, :)));
    
    % naive, do some arrayfun here or something instead
    for xi = 1:size(data, 1)
        x = data(xi, :);
        g(xi, ci) = g(xi, ci) - sum((x - mu(ci, :)).^2 ./ (2 * sigma(ci, :).^2));
    end
end

end

