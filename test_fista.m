clc;
close all;
r = 3; % desired input rank
s = 0.01; % desired sparisity i.e. #nonzero/#zero
AL = 0;
l2sdb = 10; % 20*log10(norm(L)/norm(S)) low rank to sparse ratio in dB
SNR = 40; % signal to noise raio in dB
N1 = 100;
N2 = 200;
sigma = rand(1,r)+1e-10; % random non-zero singular values
% Low rank
for i = 1:r
    u = randn(N1,1)+1i*randn(N1,1); u = u/norm(u); % normalized complex random vector;
    v = randn(N2,1); v = v/norm(v); % normalized random vector;
    AL = AL+sigma(i)*u*v';
end
AL = AL/norm(AL,'fro');
% Sparse
AS = zeros(N1,N2);
nelemspare = max(1,min(N1*N2,round(s*N1*N2)));
idx = randsrc(1,nelemspare,[1:N1*N2; ones(1,N1*N2)/N1/N2]); AS(idx) = randn(size(idx))+1i*randn(size(idx));AS = AS/norm(AS,'fro')*10^(-l2sdb/20);
%
X = AL+AS; % input to fista/ista or resnet

AN = randn(size(X))+1i*randn(size(X)); AN = AN/norm(AN,'fro')*norm(X,'fro')*10^(-SNR/20); % random noise

[L,S,error]=istalpscomp(X+AN,1e-4,2e-3,250,[N1 N2 1],AL,AS);

function [L,S,error]=istalpscomp(X,lambda1,lambda2,maxitr,dims,l,s)
% Mahdi Bayat, 5/1/2020
% ISTA, FISTA implementation for L+S
% X: is the input matrix that is going to be modeled as L+S. It can be compelx
% in general
% lambda1 is the rank regularization param
% lambda2 is the sparsity regularization param
% maxitr is the maximum number of iterations
% dims is the dimensions of original spatiotemporal matrix. if using
% synthetic data with size N1xN2, use dim = [N1 N2 1]; if using real data
% of size N1xN2xN3 then use dim = [N1 N2 N3];

n1 = dims(1);n2 = dims(2);n3 = dims(3);
error_thr = 1e-3; % not used right now
iter = 1;
error = []; % cost value
Err = []; % data mismatch error
% % [u1 s1 v1] = svd(X,'econ');
% % e1 = 5; e2 = min(size(X));
% % L = u1(:,e1:e2)*s1(e1:e2,e1:e2)*v1(:,e1:e2)';
L= 0*X; % initialization 
S = 0*L;% initialization 
Lf = 2;% Lipschitz constant. should be greater than 1.
% K = [L;S];Y = K;
Y1 = 0*L; Y2 = 0*S;
prevY1 = 0*L; prevY2 = 0*S;
prevL = 0*L; prevS = 0*S;
%%
r = 5;%min(300,min(size(X))); % a predetermined rank if we need to fix rank
prevk = 1;
k = 1;
    
while true    
%     G = K-1/Lf*(HhH*K-Hh*X); % this is ISTA. very simple.
    % Nesterov (FISTA)
    
    GL = L+1*(prevk-1)/k*(L-prevL);
    GS = S+1*(prevk-1)/k*(S-prevS);
    G = (GL+GS)-X;
    YL = GL-1/Lf*G;
    YS = GS-1/Lf*G;
    
    prevL = L;
    prevS = S;
    %%%% comment out this block and uncomment next block if you want to use
    %%%% fixed rank r
    [u1 s1 v1] = svd(YL,'econ');
    L = u1*wthresh(s1,'s',lambda1*s1(1,1))*v1';
    
    % this block finds rank r app. of input matrix YL
% % % % %     Y2=randn(size(YL,2),r); 
% % % % %     for i=1:2
% % % % %         Y1=YL*Y2;
% % % % %         Y2=YL'*Y1;
% % % % %     end
% % % % %     [Q,R]=qr(Y2,0);
% % % % %     L=(YL*Q)*Q';
%     r = max(1,r);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    S = 1*(wthresh(real(YS),'s',lambda2*max(abs(YS(:))))+1i*wthresh(imag(YS),'s',lambda2*max(abs(YS(:)))));   
    prevk = k;
    k = 0.5*(1+sqrt(1+4*k^2));
    
%     L = repmat(max(0,1-lambda*max(abs(G1(:)))./sqrt(abs(sum(G1.^2,2)))),1,len).*G1;
%     figure(1); imagesc(rms(reshape(abs(full(L)),height*div,width*div,len),3));colormap hot;title(['itr = ' num2str(iter)])
    
%     error = [error;1/2*norm(X-H*K,'fro')^2+lambda1*sum(s1(:))+lambda2*norm(S,1)];
    Err = [Err;norm(X-(L+S),'fro')^2];
    
    error = [error;1/2*Err(iter)+lambda1*trace(sqrt(L'*L))+lambda2*norm(S,1)];
    figure(2);
    subplot 231;imagesc(rms(reshape(L,n1,n2,n3),3));colorbar;title('$$\hat{L}$$','Interpreter','Latex')
    subplot 232;imagesc(rms(reshape(S,n1,n2,n3),3));colorbar;title('$$\hat{S}$$','Interpreter','Latex')
    subplot 233;imagesc(rms(reshape(X,n1,n2,n3),3));colorbar;title('$$A(L+S): input$$','Interpreter','Latex')
%     subplot 231;imagesc(reshape(abs(L(:,idx)),height,width));title('$$\hat{L}$$','Interpreter','Latex')
%     subplot 232;imagesc(reshape(abs(S(:,idx)),height*div,width*div));title('$$\hat{S}$$','Interpreter','Latex')
%     subplot 233;imagesc(reshape(abs(X(:,idx)),height,width));title('$$A(L+S): input$$','Interpreter','Latex')
    subplot 234;imagesc(abs(l));colorbar;title('$$L : input$$','Interpreter','Latex')
    subplot 235;imagesc(abs(s));colorbar;title('$$S : input$$','Interpreter','Latex')
%     subplot 231;imagesc(reshape(rms(A*L,2),height,width));
%     subplot 232;imagesc(reshape(rms(S,2),height*div,width*div));
%     subplot 234;imagesc(reshape(rms(A*l,2),height,width));
%     subplot 235;imagesc(reshape(rms(s,2),height*div,width*div));
    subplot 236;semilogy((error),'.-b');% hold on;semilogy((Err),'.-r');
    title(['error = ' num2str(error(iter)) char(10) '||X-(L+S)||_F^2 = ' num2str(Err(iter))],'Interpreter','Latex')
    if iter > maxitr %| (iter > 1 & (error(end-1)-error(end))/error(end) < error_thr)
        L = prevL;
        S = prevS;
        break;
    end
    iter=iter+1
    
end
end