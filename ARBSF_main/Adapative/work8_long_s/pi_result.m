clc;
clear all;
syms  ppi p  q  m n fun1 fun2

fun1=power((ppi*p+(1-ppi)*q),m)*power((1-(ppi*p+(1-ppi)*q)),n-m);
fun2=diff(fun1);
ppi=solve(fun2==0,ppi)
%fun2
