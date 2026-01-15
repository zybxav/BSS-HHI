function [result1,result2]=ncr0(N,k,d,pertured_vector,Ground_truth)
%pertured_vector=pertured_vector(1:d);
[item_fy,location]=sort(pertured_vector);
%这个是topk的排序
Ty_topk=location(d-k+1:d);

data_sort=zeros(d,3);
data_sort(:,1)=1:1:d;
data_sort(:,2)=Ground_truth;
%第三行是评分
data_sort2=sortrows(data_sort,2);
data_sort2(d-k+1:d,3)=1:1:k;

score_location=zeros(1,k);
temp33=data_sort2(:,1);
temp55=zeros(k,1);
temp55=temp33(d-k+1:d);
%temp55是原始数据topk个数据的排序
temp66=zeros(k,1);
temp66=data_sort2(:,3);
temp66=temp66(d-k+1:d);
%temp66是原始数据对应top-k的排名
fre_topk=zeros(1,k);

   correct_fre=zeros(1,k);
for jk=1:1:k
    %写出现在location的排序
    temp44=Ty_topk(jk);
    tempm = find(temp55==temp44);
    %写出现在location的序号
    if  isempty(tempm)==0
        score_location(jk)=temp66(tempm);
    else
    end
end
T0=1:1:k;
result1=sum(score_location)/sum(T0);
result2=length(intersect(Ty_topk,temp55));
end