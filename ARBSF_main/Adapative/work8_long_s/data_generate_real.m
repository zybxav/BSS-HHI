clc;
clear all;

%input
epsilon=4;
k=10;
load('POS.mat')
dbstop if error
d=Max_A;
temp00=floor(h1/2); 
N=temp00;
ti=6;
item1=A(1:N,:);
H=size(A);
h1=H(1);
h2=H(2);
AA=zeros(h1,Max_A);
for ik=1:h1
    for ij=1:h2
        temp0=A(ik,ij);
        if temp0==0
        else
            AA(ik,temp0)=1;
        end 
    end
end

epsilon=4;
e=exp(epsilon);
h3=randperm(h1);
AAA=AA(h3,:);
H=size(A);
h1=H(1);
h2=H(2);
AA=zeros(h1,Max_A);
for ik=1:h1
    for ij=1:h2
        pk_testtemp0=A(ik,ij);
        if temp0==0
        else
            AA(ik,temp0)=1;
        end 
    end
end

e=exp(epsilon);
h3=randperm(h1);
% AAA is a ramdom sorting of AAA
 AAA=AA(h3,:);

 
B=zeros(h1,d);
for i1=1:1:h1
    itemi=A(i1,:);
    H=find(itemi~=0);
    B(i1,itemi(H))=1;
end


Ground_truth=Ground_statistic(A,h1,h2,d);
Ground_truth2=Ground_truth./sum(Ground_truth);

[H1,H_location]=sort(Ground_truth2);
H_location2=H_location(d-k+1:d);
C=B(:,H_location2);

Ground_truth_p2=Ground_truth(H_location2);
%Ground_truth_2_p2=Ground_truth2(H_location2);

%output
%AAA dataset in binary form
%Ground_truth: Ground_truth_p2，the number of all set-values

%Phase1_compare
writematrix(AAA,'data.csv');