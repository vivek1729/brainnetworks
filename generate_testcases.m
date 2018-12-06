clear all;
%% linear cost function test cases
% %testcase 1
% size_seq = 10 * [10, 15, 20];  % size of the matrix
% a_seq = [4, 8, 16];    % number of sources
% b_seq = [4, 8, 16];    % number of sinks
% p_seq = [0.2, 0.5, 0.8];      % density of graph (to be explained)
% r_seq = [8, 32, 64];      % nunmber of trauma (to be explained)

% %testcase 2 size of graph
% size_seq = 10 * (10:10:100);  % size of the matrix
% a_seq = [16];    % number of sources
% b_seq = [16];    % number of sinks
% p_seq = [0.3];      % density of graph (to be explained)
% r_seq = [64];      % nunmber of trauma (to be explained)

%testcase 3 percentage of trauma
size_seq = 200;  % size of the matrix
a_seq = [16];    % number of sources
b_seq = [16];    % number of sinks
p_seq = [0.2, 0.8];      % density of graph (to be explained)
r_seq = [0.01, 0.05, 0.2, 0.5, 0.8]*200;      % nunmber of trauma (to be explained)

%y axises
%time
%cost 

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
    
    disp([s, a, b, p, r]);
    
    [A,B, maxFlowBefore, maxFlowAfter]=createadjmat(s,a,b,p,r);
%      disp(size(A))
%      disp(size(B))
    disp(strcat(num2str(maxFlowBefore), '\t', num2str(maxFlowAfter)))
    
    if isSquare(A) && isSquare(B)
        csvwrite(strcat("testcases/sizeofgraphset2/adjm_A_", num2str(iRow), ".csv"), A);
        csvwrite(strcat("testcases/sizeofgraphset2/adjm_B_", num2str(iRow), ".csv"), B);
    end
    
    allComb(iRow, 6:7) = [maxFlowBefore, maxFlowAfter];
end
%%
delete('testcases/sizeofgraphset2/allComb.csv')
csvwrite("testcases/sizeofgraphset2/allComb.csv", allComb);
%% 
rng(1000);  % set seeds
[A, B, f1, f2] = createadjmat(100, 1, 1, 0.05, 50);
size(A)
size(B)
f1, f2
%
G = digraph(B);
plot(G, 'EdgeLabel',G.Edges.Weight,'Linewidth', 0.5, 'NodeLabel', '', 'Layout', 'circle')
% Remove axes ticks
set(gca,'XTick',[],'YTick',[])
saveas(gcf, 'tmp2.png')
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% BN_DATASET testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
filename = 'bn-cat-mixed-species_brain_1.txt'
M = csvread(strcat('bn_dataset/', filename));
size(M)
nSourceSink = [5, 10];
nTrauma = [5, 10, 20, 40];
%%
filename = 'bn-fly-drosophila_medulla_1.txt'
M = csvread(strcat('bn_dataset/', filename));
size(M)
nSourceSink = [30, 300];
nTrauma = [10 50 250 1000];
%%
filename = 'bn-macaque-rhesus_brain_1.txt'
M = csvread(strcat('bn_dataset/', filename));
size(M)
nSourceSink = [10, 40];
nTrauma = [5 10 50 100];
%%
filename = 'bn-mouse_brain_1.txt'
M = csvread(strcat('bn_dataset/', filename));
size(M)
nSourceSink = [10, 40];
nTrauma = [5 10 50 100];

%% test cases
% allComb.csv header: nSource/nSink, nTrauma, maxFlowBefore, maxFlowAfter
rng(1000);
clearvars allComb
allComb = allcomb(nSourceSink, nTrauma);
for iRow = 1:nrow(allComb)
    M = csvread(strcat('bn_dataset/', filename));
    nNode = nrow(M);
    nSource = allComb(iRow, 1);
    nSink = nSource;
    nTrauma = allComb(iRow, 2); 
    [MBefore, MAfter, maxFlowBefore, maxFlowAfter] = ...
        get_stroke_random(M, nSource, nSink, nTrauma);
    csvwrite(strcat('bn_dataset/out/', filename, num2str(iRow), 'MBefore.csv'), MBefore);
    csvwrite(strcat('bn_dataset/out/', filename, num2str(iRow), 'MAfter.csv'), MAfter);
    allComb(iRow, 3) = maxFlowBefore;
    allComb(iRow, 4) = maxFlowAfter;
    disp(allComb(iRow,:));
end
delete(strcat('bn_dataset/out/', filename, 'allComb.csv'));
csvwrite(strcat('bn_dataset/out/', filename, 'allComb.csv'), allComb);
