%SR算法
clc;
clear;
epsilon=2;
% 设置参数的值
n = 40;   % 设置机器的个数，这是是维度，set-value的维度 
user_number=1000; %设定用户的数目 
%先说数据集
%应该是设计了一个矩阵，矩阵中的数据，生成不同概率的01列，组合起来。
user_attribute_original=zeros(user_number,n);
t_possible=0:1/(n):1-1/n;
for i1=1:n
    temp1=t_possible(i1);
    for j1=1:user_number
        temp2=rand(1,1);
        if temp2<temp1
            user_attribute_original(j1,i1)=1;
        end
    end
end

record=zeros();
%加一个数组，统计频率


%等一下，我好像记得这里是sampling,每个只选一个来着

sample_number=floor(user_number/2/n);

%sta_rood1=
temp_user_number2=1;
fre_round1=zeros(1,n);
for i2=1:n
   temp_user_numer1=temp_user_number2; 
    temp_user_number2=temp_user_number2+sample_number;
    
        %扰动+收集 
          %第一个维度找多少个人
         temp2_2=user_attribute_original(temp_user_numer1:temp_user_number2,i2);
         fre_round1(i2)=RRpertub(temp2_2,epsilon);
    end
    %然后统计次数
[bt,location_round1]=sort(fre_round1);
  n2=10;  
 fre_round1_sort=[location_round1;bt];
%挑选，再次统计后一半
fre_round1_top=fre_round1_sort(:,n-n2+1:n);


fre_round2=zeros(1,n2);
for i3=1:n2
   temp_user_numer1=temp_user_number2; 
    temp_user_number2=temp_user_number2+sample_number;
    
        %扰动+收集 
          %第一个维度找多少个人
         temp2_2=user_attribute_original(temp_user_numer1:temp_user_number2,fre_round1_top(1,i3));
         fre_round2(i3)=RRpertub(temp2_2,epsilon);
end
    fre_round2_sta=[fre_round1_top;fre_round2];
        
%两组数据求平均，挑出top-K出来
fre_round2_result=[fre_round2_sta(1,:);(fre_round2_sta(2,:)+fre_round2_sta(3,:))/2];
fre_round2_sort=sortrows(fre_round2_result,2);

n3=5;
fre_round2_top=fre_round2_sort(:,n2-n3+1:n2)


