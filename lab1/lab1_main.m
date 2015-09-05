readmonks;

assignments = [0 1 0];
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
  dt1 = build_tree(monks_1_train);
  dt2 = build_tree(monks_2_train);
  dt3 = build_tree(monks_3_train);
  
  calculate_error(dt1, monks_1_test);
end