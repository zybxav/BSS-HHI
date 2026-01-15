function item_grr=item_GRR_generate(N,item,ti,d)
for i=1:N
    itemi=item(i,:);
   % ti=sum(sum(itemi~=0));
          H=find(itemi~=0);
          if length(H)>=ti
                 % 先生成一个H长度的
                 a=randperm(length(H));
                 %然后挑选前ti个。
                 a2=a(1:ti);
                 random_num=itemi(a2);
                %   item_grr(i)=a3(a(1));
          else
              random_num=zeros(1,ti)+d+1;
              random_num(1:length(H))=itemi(1:length(H));
          end
           a3=randperm(ti);
           item_grr(i)=random_num(a3(1));
end
end