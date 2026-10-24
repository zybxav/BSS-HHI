function [result1,result2,result3,result4]=ncr(N,k,d,Ground_truth,pertured_vector)
%pertured_vector=pertured_vector(1:d);
[item_fy,location]=sort(pertured_vector);
THH=[location;item_fy];
Ty_topk=location(d-k+1:d);
%Fy_topk=item(d-k+1:d);
%将数组进行排序
data_sort=zeros(d,3);
data_sort(:,1)=1:1:d;
data_sort(:,2)=Ground_truth;
data_sort(:,4)=pertured_vector-Ground_truth;
%第三行是评分
data_sort2=sortrows(data_sort,2);
data_sort2(d-k+1:d,3)=1:1:k;

score_location=zeros(1,k);
temp33=data_sort2(:,1);
temp55=zeros(k,1);
temp55=temp33(d-k+1:d);
% %temp55是原始数据topk个数据的排序
% temp66=zeros(k,1);
% temp66=data_sort2(:,3);
% temp66=temp66(d-k+1:d);
%? we can find the  number with ranking score
%temp66是原始数据对应top-k的排名
fre_topk=zeros(1,k);

   correct_fre=zeros(1,k);
   vector_groundtruth=zeros(3,k);
  
for jk=1:1:k
    %写出现在location的排序
    temp44=Ty_topk(jk);
    tempm = find(temp55==temp44);
    %写出现在location的序号
 
    if  isempty(tempm)==0
        
        score_location(jk)=tempm;
         temp42h=find(THH(1,:)==temp44);
        temp1=item_fy(temp42h);
         
        vector2_evaluate(1,jk)=temp44;
        vector2_evaluate(2,jk)=temp1;
        
        %这个是真实的频率
        temp44h=find(data_sort2(:,1,1)==temp44);
        temp2=data_sort2(temp44h,4);
       
        vector2_evaluate(3,jk)=temp2;
        %这个是估计算法的频率
        fre_topk(jk)=power(temp1-temp2,2);
        correct_fre(jk)=power(temp1-temp2,2);
    else
        score_location(jk)=0;
         temp44h=find(data_sort2(:,1,1)==temp44);
        temp2=data_sort2(temp44h,4);
        %temp2=data_sort2(temp44,2);
        fre_topk(jk)=power(temp2,2);
    end
end
T0=1:1:k;
result1=sum(score_location)/sum(T0);
result2=length(intersect(Ty_topk,temp55));
%我们按照频率来计算吧
result3=mean(fre_topk);
length0=length(correct_fre(correct_fre~=0));
result4=sum(correct_fre)/length0;
end