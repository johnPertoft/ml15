function T = make_decision_tree(data, maxDepth)
  T = make_decision_tree_(data, [], 0, maxDepth);
end

function T = make_decision_tree_(S, used, depth, maxDepth)
  if (depth == maxDepth)
    T = struct('decision', majority_class(S));
  else
    entropy = ent(S(:, end));
    
    if (entropy < 0.1)
      T = struct('decision', majority_class(S));
    else   
      splitNode = struct;

      candidates = setdiff(1:(size(S, 2) - 1), used);
      candidateGains = arrayfun(@(attr) infoGain(S, attr), candidates);
      [~, idx] = max(candidateGains);
      splitAttr = candidates(idx);
      
      splitNode.split = splitAttr;
      splitNode.children = cell(1);
      
      i = 0;
      for v = unique(S(:, splitAttr))'
        i = i + 1;
        S_v = S(S(:, splitAttr) == v, :);
        
        Tv = make_decision_tree_(S_v, [used v], depth + 1, maxDepth);
        splitNode.children{i} = Tv;
      end
      
      T = splitNode;
    end
  end
end