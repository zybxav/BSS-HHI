epsilon=0.5;
e=exp(epsilon);
v0=zeros();
loop=10;
Ground_truth2=Ground_truth./sum(Ground_truth);
for l=1:loop
ti=15;
vector_GRR=GRR1_padding(d2,e,item,N,ti);
%vector_OLH=OLH(d2,e,item,N);
vector_OLH=OLH_padding(d2,e,item,N,ti);
[result_GRR,xor_GRR,se_GRR,correct_GRR]=ncr(N,k2,d2,Ground_truth2,vector_GRR);
[result_OLH,xor_OLH,se_OLH,correct_OLH]=ncr(N,k2,d2,Ground_truth2,vector_OLH);
%[result_bandit_moving_pk,xor_bandit_moving_pk,se_moving_pk,var_correct_moving_pk]=ncr(N,k2,d2,Ground_truth2,vector_bandit_moving_pk);
%[result_ranking_moving_pk,xor_ranking_moving_pk,se_ranking_pk,var_correct_ranking_pk]=ncr(N,k2,d2,Ground_truth2,vector_ranking_moving_pk);
                 
% [result_bandit_moving_pk,result_ranking_moving_pk,result_GRR,result_OLH,result_RR,result_RR_half]
% [xor_bandit_moving_pk,xor_ranking_moving_pk,xor_GRR,xor_OLH,xor_RR,xor_RR_half]
% [se_moving_pk,se_ranking_pk,se_GRR,se_OLH,se_RR,se_RR_half]
% [var_correct_moving_pk,var_correct_ranking_pk, correct_GRR,correct_OLH,correct_RR,correct_RR_half]
% %当k比较小的时候，SRR的表现没有OLH好。

v=[se_GRR,correct_GRR,se_OLH,correct_OLH];
v0=v0+v;
end
% v1=mean(power((vector_GRR-Ground_truth2),2))
% v2=mean(power((vector_OLH-Ground_truth2),2))
% vector_GRR1=vector_GRR(1:k2);
% vector_OLH1=vector_OLH(1:k2);
% Ground_truth3=Ground_truth2(1:k2);
% v3=mean(power((vector_GRR1-Ground_truth3),2))
% v4=mean(power((vector_OLH1-Ground_truth3),2))
% 
% vector_GRR3=vector_GRR(k2+1:d);
% vector_OLH3=vector_OLH(k2+1:d);
% Ground_truth4=Ground_truth2(k2+1:d);
% v5=mean(power((vector_GRR3-Ground_truth4),2))
% v6=mean(power((vector_OLH3-Ground_truth4),2))
v0=v0/loop
