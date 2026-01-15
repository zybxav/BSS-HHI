%把数据恢复成
%找出对应的位置
B=zeros(h1,d);
for i1=1:1:h1
    itemi=A(i1,:);
    H=find(itemi~=0);
    B(i1,itemi(H))=1;
end

  