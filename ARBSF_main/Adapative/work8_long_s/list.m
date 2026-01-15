
data_generate_syndata
%input
epsilon=10;
e=exp(epsilon);

k2=30;
item=AAA;
N=length(mean(AAA'));
d2=length(N);

%ti=20;

part=n0_caculate(N,d2,epsilon);
if part>N/d2
    part=N/d2;
end
delay=11;
dbstop if error
