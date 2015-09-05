function gain = gain(data)
  numFeatures = size(data, 2) - 1;
  entropy = ent(data(:, end));
  
  gain = zeros(1, numFeatures);
  for attr = 1:numFeatures
    g = entropy;
    
    for k = unique(data(:, attr))'
      S_k = data(data(:, attr) == k, :);
      g = g - (size(S_k, 1) / size(data, 1)) * ent(S_k(:, end));
    end
    
    gain(attr) = g;
  end
end