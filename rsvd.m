function [L3_new,S] = rsvd(L3, power, rank_k)
    [n1,n2,n3] = size(L3);
    L=reshape(L3, n1*n2, n3);
    Y2=randn(n3,rank_k);
    for i=1:power+1
       Y1=L*Y2;
       Y2=L'*Y1;
    end
    [Q,~]=qr(Y2,0);
    L_new=(L*Q)*Q';
    L3_new = reshape(L_new, n1, n2, n3);
    S=L3-L3_new;
end