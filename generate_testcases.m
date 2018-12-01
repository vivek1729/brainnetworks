clear all;
size_seq = 20 * [10, 20, 40]
a_seq = 1:3
b_seq = 1:3
p_seq = [0.5 0.6 0.7]
r_seq = 1

rng(1000);  % set seeds

allComb = allcomb(size_seq, a_seq, b_seq, p_seq, r_seq);
nRow = nrow(allComb);
for iRow = 1:1
%     disp(iRow)
    s = round(allComb(iRow, 1));
    a = round(allComb(iRow, 2));
    b = round(allComb(iRow, 3));
    p = (allComb(iRow, 4));
    r = round(allComb(iRow, 5));
    [A,B]=createadjmat(s,a,b,p,r);
     disp(size(A))
     disp(size(B))
    if isSquare(A) && isSquare(B)
        csvwrite(strcat("testcases/linear/adjm_A_", num2str(iRow), ".csv"), A);
        csvwrite(strcat("testcases/linear/adjm_B_", num2str(iRow), ".csv"), B);
    end
end
csvwrite("testcases/linear/allComb.csv", allComb);
%%
rng(1000);  % set seeds
[A, B] = createadjmat(800, 3, 1, 0.7, 1);
size(A)
size(B)