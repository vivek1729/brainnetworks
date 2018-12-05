function [MBefore, MAfter, maxFlowBefore, maxFlowAfter] = get_stroke_random(M, nSource, nSink, nTrauma)
%input and output: adjecency matrix representing a graph
%SOURCES, SINKS: source and sink nodes.

assert(isSquare(M));

%remove self connecting edges
M(logical(eye(nrow(M)))) = 0;

%assign sources and sinks
ingoing = sum(M, 1); 
outgoing = sum(M, 2)';
sources = isMaxK(outgoing-ingoing, nSource);
sinks = isMaxK(ingoing-outgoing, nSink);

assert(sum(sources)*sum(sinks) ~= 0);

%arrange sources and sinks 
M = [M(sources,:); M(~sources, :)];
M = [M(:, sources), M(:, ~sources)];
M = [M(~sinks, :); M(sinks, :)];
M = [M(:, ~sinks), M(:, sinks)];
sources = sort(sources, 'descend');
sinks = sort(sinks);

MBefore = M;

%compute max flow
M2 = zeros(size(M)+2);
M2(1:nrow(M), 1:nrow(M)) = M;
M2(nrow(M)+1, [sources]) = sum(sources);
M2([sinks], nrow(M)+2) = sum(sinks); 
maxFlowBefore = maxflow(digraph(M2), nrow(M)+1, nrow(M)+2);

%apply trauma
traumaNode = randperm(nrow(M)) .* ~sources .* ~sinks;
traumaNode = traumaNode(traumaNode ~= 0);
traumaNode = traumaNode(1:nTrauma);
M(traumaNode, :) = 0;
M(:, traumaNode) = 0;

%compute max flow
M2 = zeros(size(M)+2);
M2(1:nrow(M), 1:nrow(M)) = M;
M2(nrow(M)+1, [sources]) = sum(sources);
M2([sinks], nrow(M)+2) = sum(sinks); 
maxFlowAfter = maxflow(digraph(M2), nrow(M)+1, nrow(M)+2);

MAfter = M;