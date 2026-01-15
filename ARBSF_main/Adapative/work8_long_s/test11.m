temp11=0;
times=10;
temp11=temp11+1;

epsilon=1;
e=exp(epsilon);
vector_SRR=SRR(d,e,item_SRR,N,ti);
vector_GRR=GRR_single(d,e,item_GRR,N);
vector_OLH=OLH_single(d,e,item_GRR1,N);
vector_SRR=vector_SRR(1,1:d);
vector_OLH_SRR=OLH_SRR(d,e,item_SRR,N);
vector_GRR1=GRR_single(d,e,item_GRR1,N);
%vector_bandit_moving_pk=bandit_adaptive_moving_pk(d,k,itemB,e,N2);

k=60;
[result_bandit_moving_pk,xor_bandit_moving_pk,se_moving_pk,var_correct_moving_pk]=ncr(N2,k2,d2,Ground_truth3,vector_bandit_moving_pk);
[result_SRR,xor_SRR]=ncr0(N,k,d,vector_SRR,Ground_truth2);
[result_GRR,xor_GRR]=ncr0(N,k,d,vector_GRR,Ground_truth2);
[result_OLH,xor_OLH]=ncr0(N,k,d,vector_OLH,Ground_truth2);
[result_OLH1,xor_OLH1]=ncr0(N,k,d,vector_OLH_SRR,Ground_truth2);
[result_GRR1,xor_GRR1]=ncr0(N,k,d,vector_GRR1,Ground_truth2);
Ans1=[result_SRR,result_GRR,result_OLH,result_OLH1,result_GRR1];
Ans2=[xor_SRR,xor_GRR,xor_OLH,xor_OLH1,xor_GRR1];

