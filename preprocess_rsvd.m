[nm t] = size(D);

Y2=randn(t,rank_k);
power=1;

for i=1:power+1
   Y1=L*Y2;
   Y2=L'*Y1;
end

[Q,R]=qr(Y2,0);
L_new=(L*Q)*Q';