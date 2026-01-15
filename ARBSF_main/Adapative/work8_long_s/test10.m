%vector_SRR=SRR_padding(d2,e,item,N);
%vector_bandit_fix_pk=bandit_adaptive_fix_pk(d2,k2,item,e,N);

[result_ground_truth,xor_ground_truth,se_ground_truth,correct_ground_truth]=GTtopk(k2,d2,Ground_truth2,item,e,N2);
vector_bandit_moving_pk=bandit_adaptive_moving_pk(d2,k2,item,e,N2);
vector_ranking_moving_pk=ranking_adaptive_moving_pk(d2,k2,item,e,N2);
vector_GRR=GRR1_padding(d2,e,item,N2,ti);
vector_OLH=OLH_padding(d2,e,item,N2,ti);
vector_pure=pure(d2,item,e,N2);
vector_pure_half=pure_half(d2,item,e,N2);

%合理的应该是归一化
Ground_truth3=Ground_truth2./sum(Ground_truth2);
vector_bandit_moving_pk=vector_bandit_moving_pk./sum(vector_bandit_moving_pk);
vector_ranking_moving_pk=vector_ranking_moving_pk./sum(vector_ranking_moving_pk);
vector_pure=vector_pure./sum(vector_pure);
vector_pure_half=vector_pure_half./sum(vector_pure_half);



%[result_bandit_fix_pk,xor_bandit_fix_pk,se_fix_pk]=ncr(N,k2,d2,Ground_truth2,vector_bandit_fix_pk)
[result_bandit_moving_pk,xor_bandit_moving_pk,se_moving_pk,var_correct_moving_pk]=ncr(N2,k2,d2,Ground_truth3,vector_bandit_moving_pk);
[result_ranking_moving_pk,xor_ranking_moving_pk,se_ranking_pk,var_correct_ranking_pk]=ncr(N2,k2,d2,Ground_truth3,vector_ranking_moving_pk);
[result_GRR,xor_GRR,se_GRR,correct_GRR]=ncr(N2,k2,d2,Ground_truth3,vector_GRR);
[result_OLH,xor_OLH,se_OLH,correct_OLH]=ncr(N2,k2,d2,Ground_truth3,vector_OLH);
[result_RR,xor_RR,se_RR,correct_RR]=ncr(N2,k2,d2,Ground_truth3,vector_pure);
[result_RR_half,xor_RR_half,se_RR_half,correct_RR_half]=ncr(N2,k2,d2,Ground_truth3,vector_pure_half);
                 
% [result_bandit_moving_pk,result_ranking_moving_pk,result_GRR,result_OLH,result_RR,result_RR_half]
% [xor_bandit_moving_pk,xor_ranking_moving_pk,xor_GRR,xor_OLH,xor_RR,xor_RR_half]
% [se_moving_pk,se_ranking_pk,se_GRR,se_OLH,se_RR,se_RR_half]
% [var_correct_moving_pk,var_correct_ranking_pk, correct_GRR,correct_OLH,correct_RR,correct_RR_half]
% %当k比较小的时候，SRR的表现没有OLH好。
 
v1=[result_bandit_moving_pk,result_ranking_moving_pk,result_GRR,result_OLH,result_RR,result_RR_half,result_ground_truth];
v2=[xor_bandit_moving_pk,xor_ranking_moving_pk,xor_GRR,xor_OLH,xor_RR,xor_RR_half,xor_ground_truth];
v3=[se_moving_pk,se_ranking_pk,se_GRR,se_OLH,se_RR,se_RR_half,se_ground_truth];
v4=[var_correct_moving_pk,var_correct_ranking_pk, correct_GRR,correct_OLH,correct_RR,correct_RR_half,correct_ground_truth];
v=[v1;v2;v3;v4]