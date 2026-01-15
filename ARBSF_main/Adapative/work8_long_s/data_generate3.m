AAA=Tz;
N=12393
LL=zeros(1,N);
for gd=1:N
    LL=length(find(AAA(gd,:))~=0);
end
DDD=zeros(N,LL);
for ih=1:N
  temp=find(AAA(ih,:)~=0);
  temp2=length(temp);
  DDD(ih,1:temp2)=temp;
end
writematrix(DDD,'Steam.csv');