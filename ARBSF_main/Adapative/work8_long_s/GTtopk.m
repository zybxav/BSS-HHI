function [result_ground_truth,xor_ground_truth,se_ground_truth,correct_ground_truth]=GTtopk(k,d,Ground_truth,item,e,N2)
% %将数组进行排序
Ground_truth2=Ground_truth./sum(Ground_truth);
data_sort=zeros(d,2);
data_sort(:,1)=1:1:d;
data_sort(:,2)=Ground_truth2;

%第三行是评分
data_sort2=sortrows(data_sort,2);
Ground_truth3=data_sort2(d-k+1:d,2);
%data_sort2(d-k+1:d,3)=1:1:k;
T1=data_sort2(d-k+1:d,1);
item3=item(:,T1);
vector_pure=pure_RR(k,item3,e,N2);
vector_pure=vector_pure./sum(vector_pure);
se_ground_truth=mean(power(vector_pure'-Ground_truth3,2));
correct_ground_truth=se_ground_truth;
xor_ground_truth=k;
result_ground_truth=1;
end