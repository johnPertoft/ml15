readmonks;

assignments = [0 0 0 1];
show_subassignments = 0;

% Assignment 1
if (assignments(1))
  ent(monks_1_train(:, 7))
  ent(monks_2_train(:, 7))
  ent(monks_3_train(:, 7))
end
  
% Assignment 2
if (assignments(2))
  gain(monks_1_train)
  gain(monks_2_train)
  gain(monks_3_train)
end

if (assignments(2) && show_subassignments)
  data = monks_1_train;
  [maxGain, attr] = max(gain(data));
  for v = values(data, attr)', v
    S_v = subset(data, attr, v);
    nextLevelGains = gain(S_v)
    [nextMaxGain, nextSplitAttr] = max(nextLevelGains)
    
    % is this really right?
    for w = values(data, nextSplitAttr)'
      majority_class(subset(S_v, nextSplitAttr, w))
    end
  end
end

if (assignments(3))
  trees = cell(1, 3);
  trees{1} = build_tree(monks_1_train);
  trees{2} = build_tree(monks_2_train);
  trees{3} = build_tree(monks_3_train);
  
  [n, ~] = size(monks_1_train);
  p = randperm(n);
  frac = 0.7;
  monks_1_train_new = monks_1_train(p(1:floor(n * frac)), :);
  monks_1_validation = monks_1_train(p(floor(n * frac) + 1:n), :);
  T1 = build_tree(monks_1_train_new);
  trees{4} = prune_tree(T1, monks_1_validation);
  
  for ti = 1:length(trees), ti
    t = trees{ti};
    calculate_error(t, monks_1_train)
    calculate_error(t, monks_1_test)
  end
end

if (assignments(4))
  err1 = [];
  err3 = [];
  
  fracs = 0.3:0.05:0.8;
  
  for frac = fracs
    err1 = [err1 buildPruneAndCalcError(monks_1_train, monks_1_test, frac)];
    err3 = [err3 buildPruneAndCalcError(monks_3_train, monks_3_test, frac)];
  end
  
  plot(fracs, err1);
  hold on;
  plot(fracs, err3);
end