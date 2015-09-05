function [ gain ] = infoGain(S, attr)
    gain = ent(S(:, end));
    
    for k = unique(S(:, attr))'
      S_k = S(S(:, attr) == k, :);
      gain = gain - (size(S_k, 1) / size(S, 1)) * ent(S_k(:, end));
    end
end

