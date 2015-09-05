function entropy = ent(data)
  n = max(size(data));
  p1 = sum(data) / n;
  p0 = 1 - p1;
  
  if (p0 == 0 | p1 == 0)
    entropy = 0;
  else
    % only works for binary classes
    entropy = -p0 * log2(p0) - p1 * log2(p1);
  end
end

