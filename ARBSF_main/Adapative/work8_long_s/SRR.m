function result=SRR(d,e,item,N,ti)
%GRR版本,这个要修改，这个好像是等价的
d1=d+1;
pertured_vector_SRR=zeros(1,d1);
po=e/(ti*e+d1-ti);
qo=1/(ti*e+d1-ti);
for i=1:N
    random_num=item(i,:);
     r=1;
      if ismember(1,random_num)==1
           probobilty_vector1(1)=po;
      else
           probobilty_vector1(1)=qo;
      end
    
   for r=2:d1
    if ismember(r,random_num)==1
        probobilty_vector1(r)=probobilty_vector1(r-1)+po;
    else       
    probobilty_vector1(r)=probobilty_vector1(r-1)+qo;
    end
   end
     temp1=rand(1,1);
    temp3=min(find(probobilty_vector1>=temp1));
    pertured_vector_SRR(temp3)=pertured_vector_SRR(temp3)+1;
end
    result=(pertured_vector_SRR-(N)*qo)/((po-qo)*(N));
end