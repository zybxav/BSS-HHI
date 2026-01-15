function [v1,v2,v3,v4,v5,T1, bandit_topk, gt_topk_items]=phase2_process(d2,epsilon_idx,item,N2,k2,Ground_truth,part1,delay,time_Hu,unique_values)
epsilon=epsilon_idx;
e=exp(epsilon);
[vector_bandit_moving_pk,T1,n22]=bandit_adaptive_moving_pk(d2,k2,item,e,N2,part1,delay);
% [vector_ranking_moving_pk,T2,n33]=ranking_adaptive_moving_pk(d2,k2,item,e,N2,part1,delay);
% [vector_pure,T3]=pure(d2,item,e,N2);
% [vector_rr,T4]=RR(d2,item,e,N2);


vector_bandit_moving_pk=vector_bandit_moving_pk*2;
% vector_ranking_moving_pk=vector_ranking_moving_pk*2;
% vector_pure=vector_pure*2;
% vector_rr=vector_rr*2;

[fre_bandit,index_bandit] = maxk(vector_bandit_moving_pk, k2);
% [fre_ranking,index_ranking] = maxk(vector_ranking_moving_pk, k2);
% [fre_pure,index_pure] = maxk(vector_pure, k2);
% [fre_rr,index_rr] = maxk(vector_rr, k2);


N=N2;
% fre_bandit=fre_bandit*2;
% fre_ranking=fre_ranking*2;
% fre_pure=fre_pure*2;
% fre_rr=fre_rr*2;

%把index，对应回去数据，而不是简单的编码，
topk_bandit=unique_values(index_bandit);
% topk_ranking=unique_values(index_ranking);
% topk_pure=unique_values(index_pure);
% topk_rr=unique_values(index_rr);



%我现有一个ground_truth
 [ncr_bandit, var_bandit, accu_bandit,f1_bandit,mcr_bandit] = sort_compare2(topk_bandit, fre_bandit, Ground_truth(:,1), Ground_truth(:,2), k2, N)
 bandit_topk = [topk_bandit(:), fre_bandit(:)];
 [gt_counts, gt_idx] = maxk(Ground_truth(:,2), k2);
 gt_topk_items = Ground_truth(gt_idx, :);     % [itemID, freq]

 %然后我现在比较intersection
 % [ncr_ranking, var_ranking, accu_ranking] = sort_compare2(topk_ranking, fre_ranking, Ground_truth(:,1), Ground_truth(:,2), k2, N)
 % [ncr_pure, var_pure, accu_pure] = sort_compare2(topk_pure, fre_pure, Ground_truth(:,1), Ground_truth(:,2), k2, N)
 % [ncr_rr, var_rr, accu_rr] = sort_compare2(topk_rr, fre_rr, Ground_truth(:,1), Ground_truth(:,2), k2, N)
 % 


% v1=[ncr_bandit,ncr_ranking,ncr_pure,ncr_rr];
% v2=[var_bandit,var_ranking,var_pure,var_rr];
% v3=[accu_bandit,accu_ranking,accu_pure,accu_rr];

v1=ncr_bandit;
v2=var_bandit;
v3=accu_bandit;
v4=f1_bandit;
v5=mcr_bandit;

%filenm = ['Epsilon_'    num2str(epsilon_idx)    '_round_'    num2str(time_Hu)     '_delay_'   num2str(delay)   '.mat' ];
%save(filenm);
end