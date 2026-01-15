%RR perturb
function pi=RRpertub(original_data,epsilon)
Hsize=size(original_data);
H1=Hsize(1);
ee=exp(epsilon);
p=ee/(ee+1);
q=1/(ee+1);

for RRi=1:H1
   temp1= rand(1,1);
   if temp1<=p

   else
       original_data(RRi,1)=1-original_data(RRi,1);
   end
end
  m=sum(original_data);
  %统计结果是
  pi=(m-H1*q)/(p-q)/H1;
end