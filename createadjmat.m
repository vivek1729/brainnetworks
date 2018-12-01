function [A,B,f1,f2]=createadjmat(size,a,b,p,r)
%Input:
%size: number of nodes in the graph
%a: number of sources. Source nodes would be 1 to a
%b: number of sinks. Sink nodes would be size-b+1 to size
%p: probability of adding an edge. So if p is small, the graph would be
%more sparse
%r: number of trauma
%Output:
%A: adjacency matrix before trauma
%B: adjacency matrix after trauma
%f1: the max flow for original graph
%f2: the max flow for graph after trauma
A=zeros(size,size);
A(1,size)=1;
for i=1:size-b
    for j=a+1:size
    r=rand(1);
    if r<p
        A(i,j)=1;
    end
    end
end
for i=1:size
    A(i,i)=0;
end

temp=sum(A,2);
c=sum(temp(1:a));
A1=zeros(size+2,size+2);
A1(2:size+1,2:size+1)=A;
A1(1,2:a+1)=c;
A1(size-b+2:size+1,size+2)=c;
G1=digraph(A1);
f1 = maxflow(G1,1,size+2);

B=A;

for i=1:r
    k=randi([1,size])
    B(:,k)=0;
    B(k,:)=0;
end

B1=zeros(size+2,size+2);
B1(2:size+1,2:size+1)=B;
B1(1,2:a+1)=c;
B1(size-b+2:size+1,size+2)=c;
G2=digraph(B1);
f2 = maxflow(G2,1,size+2);

end