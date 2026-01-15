function result=GRR1_padding(d,e,item,N,ti)
%GRR版本,这个要修改，这个好像是等价的
%先添加一个空白
d=d+1;

po_GRR=e/(e+d-1);
qo_GRR=1/(e+d-1);
pertured_vector_GRR=zeros(1,d);
%ti=floor(d/2);

for ikk=1:N
    probobilty_vector2=zeros(1,d);
    itemi=item(ikk,:);
    H=find(itemi==1);
    %%%%%%
      if length(H)>=ti
          random_num = H(randperm(numel(H),ti));
          else
          random_num=zeros(1,ti)+d;
          random_num(1:length(H))=H;
      end
     temp_num = randperm(ti,1);
     r=random_num(temp_num);
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
% result=(pertured_vector_GRR-(N/ti)*qo_GRR)./((po_GRR-qo_GRR)*(N/ti));
 result=(pertured_vector_GRR-(N)*qo_GRR)./((po_GRR-qo_GRR)*(N));
 result=result(1:d-1);
end