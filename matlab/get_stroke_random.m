function [M, sources, sinks, flow] = get_stroke_random(M)
%input and output: adjecency matrix representing a graph
%SOURCES, SINKS: source and sink nodes.

%remove self connecting edges
for i = 1:nrow(M)
    M(i, i) = 0;
end

%remove disconnected nodes
ingoing = sum(M, 1);
outgoing = sum(M, 2);
M = M(ingoing|outgoing, ingoing|outgoing);

%assign sources and sinks
%remove outgoing edges to assign sinks
%remove ingoing edges to assign sources
sources = ~ingoing & outgoing;
sinks = ingoing & ~outgoing;

%compute max flow

