%input
%top_2k=zeros();
top_2k=[1 2 3 4 5 6 7 8 9 10];
%K2=10;
%split data
%n/2
%
A1=AAA(1:floor(N/2),:);
A2=AAA(floor(N/2)+1:N,:);
%catch top2k values
%BBB=zeors(N-floor(N/2), K2);
BBB=A2(:,top_2k);