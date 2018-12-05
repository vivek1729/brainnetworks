function y = maxk(x, k)
y = sort(x, 'descend');
y = y(1:k);