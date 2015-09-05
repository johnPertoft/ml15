function [err] = buildPruneAndCalcError(data, test, frac)
  n = size(data, 1);
  p = randperm(n);
  trainingSet = data(p(1:floor(n * frac)), :);
  validationSet = data(p(floor(n * frac) + 1:n), :);
  
  tree = prune_tree(build_tree(trainingSet), validationSet);
  err = calculate_error(tree, test);
end

