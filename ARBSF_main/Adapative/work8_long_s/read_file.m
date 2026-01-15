%先导入excel中，然后复制到第二个sheet，转为数字
%A=xlsread('POS.xlsx')
A=xlsread('Retail.xlsx');
A(isnan(A)) = -1;
A=A+1;

%POS最大是1655
%retai 最大是2603