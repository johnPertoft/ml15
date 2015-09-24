function c = adaboost_discriminant(data, mu, sigma, priors, alpha, classes)
T = size(mu, 3);
[M, ~] = size(data);

c_weak = zeros(M, T);

for t = 1:T
    g = discriminant(data, mu(:, :, t), sigma(:, :, t), priors(t, :));
    [~, class] = max(g, [], 2);
    class = class - 1;
    
    c_weak(:, t) = class;
end

% translate to classes in {-1, 1}
c_weak = c_weak * 2 - 1;

% multiply each weak classifiers response with its
% alpha and then sum the alpha weighted responses
% TODO: might have to change this
c = sign(sum(bsxfun(@times, c_weak, alpha), 2));

% translate classes back to {0, 1}
c = (c + 1) ./ 2;

%choose the class ci that 

end