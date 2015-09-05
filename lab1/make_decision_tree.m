function [ ] = make_decision_tree(data, depth, maxDepth)
  disp(['Entropy: ' num2str(ent(data(:, end)))]);

  if (depth == maxDepth)
    disp(['Decision: ' num2str(majority_class(data))]);
    return;
  end
  
  [~, attr] = max(gain(data));
  disp(['Depth: ' num2str(depth) ' splitting at: ' num2str(attr)]);
  
  for v = values(data, attr)'
    S_k = data(data(:, attr) == v, :);
    make_decision_tree(S_k, depth + 1, maxDepth);
  end
end