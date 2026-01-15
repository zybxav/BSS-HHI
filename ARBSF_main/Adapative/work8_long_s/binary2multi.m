H = size(AAA);
h1 = H(1);
h2 = H(2);
for i = 1:h1
    temp0 = AAA(i,:);
    k(i) = length(find(temp0==1));
end
max_length = max(k(i));
kl = zeros(h1, max_length);
for j = 1: h1
    temp0 = AAA(j,:);
        ko = length(find(temp0==1));
        ko1 = find(temp0==1);
        kl(j,1:ko)=ko1;
end
