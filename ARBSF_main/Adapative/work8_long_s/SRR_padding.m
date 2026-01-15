function result=SRR_padding(d,e,item,N)
%GRR版本,这个要修改，这个好像是等价的
d1=d+1;
ti=floor(d1/2);
pertured_vector_SRR=zeros(1,d1);
po=e/(ti*e+d1-ti);
qo=1/(ti*e+d1-ti);
for i=1:N
      k1=0;
     k2=0;
    itemi=item(i,:);
   % ti=sum(sum(itemi~=0));
    probobilty_vector1=zeros(1,d1);
          H=find(itemi==1);    
          if length(H)>=ti
          random_num = H(randperm(numel(H),ti));
          else
              u=10000000;
              random_num=zeros(1,d+1)+d+1;
              rand_num(1:length(H))=H;
          end
     r=1;
     %随机挑10个附送
   
      if ismember(1,random_num)==1
            k1=1;
           probobilty_vector1(1)=po;
      else
          K2=1;
           probobilty_vector1(1)=qo;
      end
    
   for r=2:d1
    if ismember(r,random_num)==1
        k1=k1+1;
        probobilty_vector1(r)=probobilty_vector1(r-1)+po;
    else    
        k2=k2+1;
    probobilty_vector1(r)=probobilty_vector1(r-1)+qo;
    end
   end
     temp1=rand(1,1);
    temp3=min(find(probobilty_vector1>=temp1));
    pertured_vector_SRR(temp3)=pertured_vector_SRR(temp3)+1;
end
    result=(pertured_vector_SRR-(N/ti)*qo)/((po-qo)*(N/ti));
end