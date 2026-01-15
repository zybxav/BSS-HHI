function result=GRR(d,e,item,N)
%GRR版本,这个要修改，这个好像是等价的
po_GRR=e/(e+d-1);
qo_GRR=1/(e+d-1);
pertured_vector_GRR=zeros(1,d);
for i=1:N
    probobilty_vector2=zeros(1,d);
    itemi=item(i,:);
      H=find(itemi==1);
      %我从结果中
     c1 = randi(length(H),1,1);
     r=H(c1);
    r2=1;
    if r==1
        probobilty_vector2(1)=po_GRR;
    else
         probobilty_vector2(1)=qo_GRR;
    end
    for r2=2:d
    if r2==r
       probobilty_vector2(r2)=probobilty_vector2(r2-1)+po_GRR;
    else    
       probobilty_vector2(r2)=probobilty_vector2(r2-1)+qo_GRR;
    end
    end
     temp1=rand(1,1);
     temp3=min(find(probobilty_vector2>=temp1));
     pertured_vector_GRR(temp3)=pertured_vector_GRR(temp3)+1;
end
 result=(pertured_vector_GRR-N*qo_GRR)/(po_GRR-qo_GRR)/N;
end