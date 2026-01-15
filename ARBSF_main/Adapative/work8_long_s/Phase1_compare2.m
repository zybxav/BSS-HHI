temp11=0;
times=10;
Ans3=zeros(10,4);
Ans4=zeros(10,4);
for epsilon=0.5:0.5:5   
temp11=temp11+1;
Ans1=zeros(times,4);
Ans2=zeros(times,4);
e=exp(epsilon);
for  ij=1:times
vector_SRR=SRR(d,e,item_SRR,N,ti);
vector_GRR=GRR_single(d,e,item_GRR,N);
vector_OLH=OLH_single(d,e,item_OLH,N);
vector_SRR=vector_SRR(1,1:d);
vector_OLH_SRR=OLH_SRR(d,e,item_SRR,N);
k=60;
Ground_truth=Ground_statistic(A,h1,h2,d);
Ground_truth2=Ground_truth./sum(Ground_truth);
[result_SRR,xor_SRR]=ncr0(N,k,d,vector_SRR,Ground_truth2);
[result_GRR,xor_GRR]=ncr0(N,k,d,vector_GRR,Ground_truth2);
[result_OLH,xor_OLH]=ncr0(N,k,d,vector_OLH,Ground_truth2);
[result_OLH1,xor_OLH1]=ncr0(N,k,d,vector_OLH_SRR,Ground_truth2);
Ans1(ij,:)=[result_SRR,result_GRR,result_OLH,result_OLH1];
Ans2(ij,:)=[xor_SRR,xor_GRR,xor_OLH,xor_OLH1];
end
Ans3(temp11,:)=mean(Ans1);
Ans4(temp11,:)=mean(Ans2);
end