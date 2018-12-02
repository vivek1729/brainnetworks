clear all;
size_seq = 20 * [10, 20, 40];  % size of the matrix
a_seq = [10, 20, 40];    % number of sources
b_seq = [10, 20, 40];    % number of sinks
p_seq = [0.2, 0.5, 0.8];      % density of graph (to be explained)
r_seq = 10;      % nunmber of trauma (to be explained)

rng(1000);  % set seeds

allComb = allcomb(size_seq, a_seq, b_seq, p_seq, r_seq);
nRow = nrow(allComb);
for iRow = 1:nRow
%     disp(iRow)
    s = round(allComb(iRow, 1));
    a = round(allComb(iRow, 2));
    b = round(allComb(iRow, 3));
    p = (allComb(iRow, 4));
    r = round(allComb(iRow, 5));
    [A,B, maxFlowBefore, maxFlowAfter]=createadjmat(s,a,b,p,r);
%      disp(size(A))
%      disp(size(B))
    disp(strcat(num2str(maxFlowBefore), '\t', num2str(maxFlowAfter)))
    if isSquare(A) && isSquare(B)
        csvwrite(strcat("testcases/linear/adjm_A_", num2str(iRow), ".csv"), A);
        csvwrite(strcat("testcases/linear/adjm_B_", num2str(iRow), ".csv"), B);
    end
    
    %allComb(iRow, 6:7) = [maxFlowBefore, maxFlowAfter];
end
%%
csvwrite("testcases/linear/allComb.csv", allComb);
%%
rng(1000);  % set seeds
[A, B, f1, f2] = createadjmat(100, 1, 1, 1, 10000);
size(A)
size(B)
f1, f2