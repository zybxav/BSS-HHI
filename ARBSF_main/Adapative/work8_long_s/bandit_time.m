% exp(-2*n*power(t,2))=delta
t=0.01;
delta0=0.05;
%最少的次数设定为 (1/delta)/2,按照一半来算嘛，或者说我们起码要淘汰一半以上再说？
delta1=50*delta0;
n=-log(delta1)/(2*power(t,2));
%具体的步骤应该是不断累计，直到所有的 第m-th的在多个阶段不变
%当第m-th的元素是A嘛
%如果我们直到真正的第m-th数据
