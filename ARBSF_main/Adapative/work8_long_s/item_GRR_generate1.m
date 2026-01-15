function item_grr1=item_GRR_generate1(N,item,ti,d)
for i=1:N
    itemi=item(i,:);
   % ti=sum(sum(itemi~=0));
          H=find(itemi~=0);
                 % 先生成一个H长度的
                 a=randperm(length(H));
                 %然后挑选前ti个。
           item_grr1(i)=itemi(a(1));
end
end