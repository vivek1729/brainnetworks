function y = isMaxK(x, k)
%returns the logical vector of maximal k items
[~, ii] = sort(x, 'descend');
y = zeros(size(x));
y(ii(1:k)) = 1;
y = logical(y);