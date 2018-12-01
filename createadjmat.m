function [A,B]=createadjmat(size,a,b,p,r)
%Input:
%size: number of nodes in the graph
%a: number of sources. Source nodes would be 1 to a
%b: number of sinks. Sink nodes would be size-b+1 to size
%p: probability of adding an edge. So if p is large, the graph would be
%more sparse
%r: number of trauma
%s: size of trauma
%Output:
%A: adjacency matrix before trauma
%B: adjacency matrix after trauma
A=zeros(size,size);
for i=1:size-b
    A(i,a+i)=1;
end
for i=1:size-b
    for j=a+1:size
    r=rand(1);
    if r>p
        A(i,j)=1;
    end
    end
end
for i=1:size
    A(i,i)=0;
end

B=A;

for i=1:r
    k=randi([1,size])
    B(:,k)=0;
    B(k,:)=0;
end
end