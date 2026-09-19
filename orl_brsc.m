%% BRAUER SUBSPACE CLUSTERING ON THE FULL ORL DATASET


clear; clc; close all;
%% ff
clear;
%% Settings
cfg.data_file = 'ORL_32x32.mat';
cfg.seed = 0;
cfg.max_iter = 800;                 % fixed horizon
cfg.learning_rate = 1e-3;
cfg.beta1 = 0.9;
cfg.beta2 = 0.999;
cfg.adam_eps = 1e-8;
cfg.max_grad_norm = 100;

cfg.lambda_brauer = 0.10 %10.0;           % starting value for sensitivity analysis
cfg.lambda_frobenius =0  %0.1;
cfg.abs_smoothing_eps = 1e-8;
cfg.radius_eps = 1e-12;

cfg.enforce_symmetric = true;      % recommended for eigenvalue interpretation
cfg.init_mode = 'zeros';           % 'zeros' or 'ridge'
cfg.ridge_init = 1e-1;
cfg.center_features = true;
cfg.l2_normalize_samples = true;

cfg.print_every = 10;
cfg.evaluate_every = 10;           % evaluation never affects training
cfg.kmeans_replicates = 20;
cfg.kmeans_max_iter = 300;
cfg.output_file = 'ORL_Brauer_full_results.mat';

% --- KSS monitoring / stopping (ADDED; optimization itself is unchanged) ---
cfg.kss_dim = 6;                    % pre-specified; do NOT tune with labels
cfg.kss_seed = 12345;               % fixed seed for deterministic CQM evaluation
cfg.kss_warmup_iter = 40;
cfg.kss_rel_tol = 1e-3;             % exploratory threshold
cfg.kss_patience = 4;               % consecutive checks without meaningful improvement
cfg.kss_actual_stop = true;        % FIRST RUN: false = simulate only


rng(cfg.seed,'twister');

%% Load and preprocess
[X,labels,data_name,label_name] = load_orl_mat(cfg.data_file);
labels = labels(:);
[~,~,labels] = unique(labels,'stable');
X = double(X);
[d,n] = size(X);
K = numel(unique(labels));

%fprintf('Data variable: %s | labels: %s\n',data_name,label_name);
%fprintf('Data size: %d x %d | classes: %d\n',d,n,K);
%fprintf('Class counts: '); fprintf('%d ',accumarray(labels,1)); fprintf('\n');

if n ~= numel(labels)
    error('Number of samples and labels does not match.');
end

if cfg.center_features
    X = X - mean(X,2);
end
if cfg.l2_normalize_samples
    X = X ./ max(sqrt(sum(X.^2,1)),eps);
end

%% Initialize representation matrix C
switch lower(cfg.init_mode)
    case 'zeros'
        C = zeros(n,n);
    case 'ridge'
        G = X'*X;
        C = (G + cfg.ridge_init*eye(n))\G;
    otherwise
        error('Unknown initialization mode: %s',cfg.init_mode);
end
C(1:n+1:end) = 0;
if cfg.enforce_symmetric
    C = 0.5*(C+C');
    C(1:n+1:end) = 0;
end

% Adam state
m = zeros(n,n);
v = zeros(n,n);

% Histories
loss_hist = nan(cfg.max_iter,1);
self_hist = nan(cfg.max_iter,1);
brauer_hist = nan(cfg.max_iter,1);
frob_hist = nan(cfg.max_iter,1);
grad_hist = nan(cfg.max_iter,1);

eval_iters = unique([1,cfg.evaluate_every:cfg.evaluate_every:cfg.max_iter,cfg.max_iter]);
acc_hist = nan(numel(eval_iters),1);
nmi_hist = nan(numel(eval_iters),1);
ari_hist = nan(numel(eval_iters),1);

% --- KSS diagnostic histories (ADDED) ---
kss_hist = nan(numel(eval_iters),1);
kss_acc_hist = nan(numel(eval_iters),1);  % labels used only retrospectively
kss_nmi_hist = nan(numel(eval_iters),1);
kss_ari_hist = nan(numel(eval_iters),1);

eval_counter = 0;

% --- KSS stopping state (ADDED) ---
best_kss = inf;
best_kss_iter = NaN;
reference_kss = inf;
kss_patience_counter = 0;
simulated_stop_iter = NaN;
simulated_return_iter = NaN;
last_iter = 0;

fprintf('\n============================================================\n');
fprintf('FULL ORL: BRAUER  + SELF-EXPRESSION\n');
fprintf('============================================================\n');
fprintf('lambdaB=%.4g  | lr=%.4g | iterations=%d\n\n',...
    cfg.lambda_brauer,cfg.learning_rate,cfg.max_iter);

%% Optimization
for t = 1:cfg.max_iter
    last_iter = t;
    [L,Lself,LB,LF,grad,radii] = brauer_objective_gradient(...
        X,C,cfg.lambda_brauer,cfg.lambda_frobenius,...
        cfg.abs_smoothing_eps,cfg.radius_eps);

    if cfg.enforce_symmetric
        grad = 0.5*(grad+grad');
    end
    grad(1:n+1:end) = 0;

    grad_norm = norm(grad,'fro');
    if grad_norm > cfg.max_grad_norm
        grad = grad*(cfg.max_grad_norm/max(grad_norm,eps));
        grad_norm = cfg.max_grad_norm;
    end

    m = cfg.beta1*m + (1-cfg.beta1)*grad;
    v = cfg.beta2*v + (1-cfg.beta2)*(grad.^2);
    mhat = m/(1-cfg.beta1^t);
    vhat = v/(1-cfg.beta2^t);
    C = C - cfg.learning_rate*mhat./(sqrt(vhat)+cfg.adam_eps);

    C(1:n+1:end) = 0;
    if cfg.enforce_symmetric
        C = 0.5*(C+C');
        C(1:n+1:end) = 0;
    end

    loss_hist(t)=L;
    self_hist(t)=Lself;
    brauer_hist(t)=LB;
    frob_hist(t)=LF;
    grad_hist(t)=grad_norm;

    do_eval = any(eval_iters==t);
    if do_eval
        eval_counter = eval_counter+1;
        Wt = build_affinity(C);
        predt = spectral_cluster(Wt,K,cfg.kmeans_replicates,...
            cfg.kmeans_max_iter,cfg.seed+t);
        [acct,~] = clustering_accuracy(labels,predt);
        nmit = nmi_score(labels,predt);
        arit = ari_score(labels,predt);
        f = compute_f(labels,predt);
        
        acc_hist(eval_counter)=acct;
        nmi_hist(eval_counter)=nmit;
        ari_hist(eval_counter)=arit;
        %f_hist(eval_counter) = f

        % =============================================================
        % KSS MONITORING (ADDED)
        % =============================================================
        % Keep the original reporting clustering above unchanged.
        % For KSS, use a fixed seed so random k-means initializations do
        % not masquerade as changes in clustering quality.
        pred_kss = spectral_cluster(Wt,K,cfg.kmeans_replicates,...
            cfg.kmeans_max_iter,cfg.kss_seed);

        kss_t = kss_cost(X,pred_kss,K,cfg.kss_dim);
        kss_hist(eval_counter) = kss_t;

        % Ground-truth metrics below are diagnostic ONLY.
        [kss_acct,~] = clustering_accuracy(labels,pred_kss);
        kss_nmit = nmi_score(labels,pred_kss);
        kss_arit = ari_score(labels,pred_kss);
        kss_acc_hist(eval_counter) = kss_acct;
        kss_nmi_hist(eval_counter) = kss_nmit;
        kss_ari_hist(eval_counter) = kss_arit;

        % Simulated/actual label-independent early stopping.
        % Lower KSS is better.
        if t >= cfg.kss_warmup_iter
            if kss_t < best_kss
                best_kss = kss_t;
                best_kss_iter = t;
            end

            if isinf(reference_kss)
                reference_kss = kss_t;
                kss_patience_counter = 0;
            else
                rel_improvement = (reference_kss-kss_t) / ...
                    (abs(reference_kss)+1e-12);

                if rel_improvement > cfg.kss_rel_tol
                    reference_kss = kss_t;
                    kss_patience_counter = 0;
                else
                    kss_patience_counter = kss_patience_counter + 1;
                end

                if kss_patience_counter >= cfg.kss_patience && ...
                        isnan(simulated_stop_iter)
                    simulated_stop_iter = t;
                    simulated_return_iter = best_kss_iter;
                    fprintf('\n[KSS] stop signal at iter %d; best checkpoint = %d; KSS = %.6e\n',...
                        simulated_stop_iter,simulated_return_iter,best_kss);
                end
            end
        end
    end

    if t==1 || mod(t,cfg.print_every)==0 || t==cfg.max_iter
        fprintf(['Iter %04d | Loss %.6e | Self %.6e | Brauer %.6e | ',...
            'Frob %.6e | ||g|| %.3e | mean(R) %.3e'],...
            t,L,Lself,LB,LF,grad_norm,mean(radii));
        if do_eval
            fprintf(' | ACC %.4f | NMI %.4f | ARI %.4f | KSS %.6e',...
                acct,nmit,arit,kss_t);
        end
        fprintf('\n');
    end

    % Disabled by default: first validate the simulated stopping point.
    if cfg.kss_actual_stop && ~isnan(simulated_stop_iter) && ...
            t >= simulated_stop_iter
        fprintf('ACTUAL KSS EARLY STOP at iteration %d.\n',t);
        break;
    end
end

%% Final spectral clustering: final iteration only
W = build_affinity(C);
pred = spectral_cluster(W,K,cfg.kmeans_replicates,...
    cfg.kmeans_max_iter,cfg.seed+last_iter+1);
[ACC,mapped_pred] = clustering_accuracy(labels,pred);
NMI = nmi_score(labels,pred);
ARI = ari_score(labels,pred);
Fscore = compute_f(labels,pred)

fprintf('\n============================================================\n');
fprintf('FINAL RESULT AT ITERATION %d\n',last_iter);
fprintf('============================================================\n');
fprintf('ACC: %.4f (%.2f%%)\n',ACC,100*ACC);
fprintf('NMI: %.4f (%.2f%%)\n',NMI,100*NMI);
fprintf('ARI: %.4f (%.2f%%)\n',ARI,100*ARI);
fprintf('Fscore: %.4f (%.2f%%)\n',Fscore,100*Fscore);

fprintf('||C||_F: %.6f\n',norm(C,'fro'));
fprintf('||C||_*: %.6f\n',sum(svd(C,'econ')));
fprintf('rank(C): %d\n',rank(C));
fprintf('rho(C): %.6f\n',max(abs(eig(C))));

%% Save
results = struct;
results.config = cfg;
results.X = X;
results.labels = labels;
results.C = C;
results.W = W;
results.predicted_labels = pred;
results.mapped_predicted_labels = mapped_pred;
results.ACC = ACC;
results.NMI = NMI;
results.ARI = ARI;
results.Fscore = Fscore;
results.loss_history = loss_hist;
results.self_expression_history = self_hist;
results.brauer_history = brauer_hist;
results.frobenius_history = frob_hist;
results.gradient_norm_history = grad_hist;
results.evaluation_iterations = eval_iters(:);
results.ACC_history = acc_hist;
results.NMI_history = nmi_hist;
results.ARI_history = ari_hist;

% --- KSS monitoring results (ADDED) ---
results.KSS_history = kss_hist;
results.KSS_ACC_history = kss_acc_hist;
results.KSS_NMI_history = kss_nmi_hist;
results.KSS_ARI_history = kss_ari_hist;
results.KSS_simulated_stop_iteration = simulated_stop_iter;
results.KSS_simulated_return_iteration = simulated_return_iter;
results.KSS_best_value = best_kss;
results.KSS_best_iteration = best_kss_iter;
results.actual_final_iteration = last_iter;
%results.Fscore_history =f_hist;
save(cfg.output_file,'results','-v7.3');
fprintf('Saved: %s\n',cfg.output_file);

%% Plots
figure('Name','Losses');
semilogy(max(loss_hist,eps),'LineWidth',1.2); hold on;
semilogy(max(self_hist,eps),'LineWidth',1.2);
semilogy(max(cfg.lambda_brauer*brauer_hist,eps),'LineWidth',1.2);
semilogy(max(cfg.lambda_frobenius*frob_hist,eps),'LineWidth',1.2);
grid on; xlabel('Iteration'); ylabel('Loss');
legend('Total','Self-expression','lambda_B Brauer','lambda_F Frobenius','Location','best');

figure('Name','Evaluation history');
plot(eval_iters,100*acc_hist,'-o','LineWidth',1.2); hold on;
plot(eval_iters,100*nmi_hist,'-s','LineWidth',1.2);
plot(eval_iters,100*ari_hist,'-d','LineWidth',1.2);
%plot(eval_iters,100*f_hist,'-g','LineWidth',1.2);
grid on; xlabel('Iteration'); ylabel('Score (%)');
legend('ACC','NMI','ARI', 'Fscore','Location','best');

% --- KSS diagnostic plot (ADDED) ---
figure('Name','KSS stopping diagnostic');
tiledlayout(2,1);

nexttile;
plot(eval_iters,kss_hist,'-o','LineWidth',1.2);
grid on; xlabel('Iteration'); ylabel('KSS cost');
title('Label-independent KSS cost (fixed spectral-clustering seed)');
if ~isnan(simulated_stop_iter)
    xline(simulated_stop_iter,'--',sprintf('Stop signal %d',simulated_stop_iter));
    if ~isnan(simulated_return_iter)
        xline(simulated_return_iter,':',sprintf('Return %d',simulated_return_iter));
    end
end

nexttile;
plot(eval_iters,100*kss_acc_hist,'-o','LineWidth',1.2); hold on;
plot(eval_iters,100*kss_nmi_hist,'-s','LineWidth',1.2);
plot(eval_iters,100*kss_ari_hist,'-d','LineWidth',1.2);
grid on; xlabel('Iteration'); ylabel('Score (%)');
legend('ACC','NMI','ARI','Location','best');
title('Ground-truth metrics for the SAME deterministic clustering (analysis only)');
if ~isnan(simulated_stop_iter)
    xline(simulated_stop_iter,'--');
end

[~,order] = sort(labels);
figure('Name','Affinity matrix');
imagesc(W(order,order)); axis image; colorbar;
title('W ordered by class (labels used only for visualization)');

sv = svd(C,'econ');
figure('Name','Singular values');
semilogy(max(sv,eps),'o-'); grid on;
xlabel('Index'); ylabel('Singular value');

%% =========================== LOCAL FUNCTIONS ============================
function [X,labels,data_name,label_name] = load_orl_mat(filename)
    if ~isfile(filename)
        error('Cannot find %s. Set cfg.data_file to the correct path.',filename);
    end
    S = load(filename);
    names = fieldnames(S);

    labels=[]; label_name='';
    preferred_labels={'gnd','labels','label','Y','y','truth','gt','classes'};
    for i=1:numel(preferred_labels)
        nm=preferred_labels{i};
        if isfield(S,nm) && isnumeric(S.(nm)) && isvector(S.(nm))
            tmp=S.(nm); labels=double(tmp(:)); label_name=nm; break;
        end
    end
    if isempty(labels)
        for i=1:numel(names)
            a=S.(names{i});
            if isnumeric(a) && isvector(a) && numel(a)>=20
                a=double(a(:)); u=unique(a);
                if numel(u)>=2 && numel(u)<=100 && all(abs(a-round(a))<1e-10)
                    labels=a; label_name=names{i}; break;
                end
            end
        end
    end
    if isempty(labels), error('Could not detect label vector.'); end
    n=numel(labels);

    X=[]; data_name='';
    preferred_data={'fea','X','x','data','Data','features','images'};
    for i=1:numel(preferred_data)
        nm=preferred_data{i};
        if isfield(S,nm)
            a=S.(nm);
            if isnumeric(a) && ismatrix(a) && ~isvector(a) && any(size(a)==n)
                X=double(a); data_name=nm; break;
            end
        end
    end
    if isempty(X)
        best=-inf;
        for i=1:numel(names)
            a=S.(names{i});
            if isnumeric(a) && ismatrix(a) && ~isvector(a) && any(size(a)==n)
                if numel(a)>best
                    X=double(a); data_name=names{i}; best=numel(a);
                end
            end
        end
    end
    if isempty(X), error('Could not detect data matrix.'); end
    if size(X,2)==n
        % already features x samples
    elseif size(X,1)==n
        X=X';
    else
        error('Detected data matrix does not contain %d samples.',n);
    end
end

function [total,Lself,LB,LF,grad,R] = brauer_objective_gradient(X,C,lambdaB,lambdaF,abs_eps,radius_eps)
    n=size(C,1);

    E=X*C-X;
    Lself=0.5*sum(E(:).^2);
    grad_self=X'*E;

    smooth_abs=sqrt(C.^2+abs_eps^2);
    smooth_abs(1:n+1:end)=0;
    R=sum(smooth_abs,2);

    s=sqrt(R+radius_eps);
    S=sum(s);
    % Exactly equal to 2/(n-1)*sum_{i<j}s_i*s_j.
    LB=(S^2-sum(s.^2))/(n-1);

    dB_dR=(S./s-1)/(n-1);
    smooth_sign=C./sqrt(C.^2+abs_eps^2);
    smooth_sign(1:n+1:end)=0;
    grad_B=(dB_dR*ones(1,n)).*smooth_sign;

    LF=0.5*sum(C(:).^2);
    total=Lself+lambdaB*LB+lambdaF*LF;
    grad=grad_self+lambdaB*grad_B+lambdaF*C;
end

function W=build_affinity(C)
    n=size(C,1);
    W=0.5*(abs(C)+abs(C'));
    W(1:n+1:end)=0;
    if max(W(:))>0, W=W/max(W(:)); end
end

function labels=spectral_cluster(W,K,reps,max_iter,seed)
    n=size(W,1);
    W=0.5*(W+W'); W(1:n+1:end)=0;
    deg=sum(W,2);
    dinv=1./sqrt(max(deg,1e-12));
    L=eye(n)-diag(dinv)*W*diag(dinv);
    L=0.5*(L+L');
    [V,e]=eig(L,'vector');
    [~,idx]=sort(real(e),'ascend');
    U=real(V(:,idx(1:K)));
    U=U./max(sqrt(sum(U.^2,2)),1e-12);
    rng(seed,'twister');
    if exist('kmeans','file')==2
        labels=kmeans(U,K,'Replicates',reps,'MaxIter',max_iter,...
            'EmptyAction','singleton','Start','plus');
    else
        labels=simple_kmeans(U,K,reps,max_iter);
    end
end

function best_labels=simple_kmeans(X,K,reps,max_iter)
    n=size(X,1); best_obj=inf; best_labels=ones(n,1);
    for r=1:reps
        centers=zeros(K,size(X,2));
        centers(1,:)=X(randi(n),:);
        minD=sum((X-centers(1,:)).^2,2);
        for k=2:K
            p=minD/max(sum(minD),eps); cdf=cumsum(p);
            id=find(cdf>=rand,1); if isempty(id), id=randi(n); end
            centers(k,:)=X(id,:);
            minD=min(minD,sum((X-centers(k,:)).^2,2));
        end
        labels=ones(n,1);
        for it=1:max_iter
            D=sum(X.^2,2)+sum(centers.^2,2)'-2*X*centers';
            [mind,newlabels]=min(D,[],2);
            if it>1 && all(newlabels==labels), break; end
            labels=newlabels;
            for k=1:K
                id=labels==k;
                if any(id)
                    centers(k,:)=mean(X(id,:),1);
                else
                    [~,q]=max(mind); centers(k,:)=X(q,:); labels(q)=k;
                end
            end
        end
        D=sum(X.^2,2)+sum(centers.^2,2)'-2*X*centers';
        obj=sum(min(D,[],2));
        if obj<best_obj, best_obj=obj; best_labels=labels; end
    end
end

function cost = kss_cost(X,cluster_labels,K,d)
    % KSS cost:
    %   (1/N) sum_k sum_{x_i in c_k} ||x_i-U_k U_k' x_i||_2^2
    %
    % No ground-truth labels are used.
    N = size(X,2);
    total_residual = 0;

    for k = 1:K
        idx = find(cluster_labels==k);
        if isempty(idx)
            cost = inf;
            return;
        end

        Xk = X(:,idx);
        d_eff = min(d,size(Xk,2));

        [U,~,~] = svd(Xk,'econ');
        U = U(:,1:d_eff);

        R = Xk - U*(U'*Xk);
        total_residual = total_residual + sum(R(:).^2);
    end

    cost = total_residual/N;
end

function [acc,mapped]=clustering_accuracy(y,p)
    y=y(:); p=p(:);
    [~,~,y]=unique(y,'stable'); [~,~,p]=unique(p,'stable');
    K=max(max(y),max(p)); M=zeros(K,K);
    for i=1:numel(y), M(p(i),y(i))=M(p(i),y(i))+1; end
    assign=hungarian_min(max(M(:))-M);
    mapped=zeros(size(p));
    for k=1:K, mapped(p==k)=assign(k); end
    acc=mean(mapped==y);
end

function assignment=hungarian_min(cost)
    [n,m]=size(cost);
    if n>m, error('Hungarian input must have rows <= columns.'); end
    u=zeros(n+1,1); v=zeros(m+1,1); p=zeros(m+1,1); way=zeros(m+1,1);
    for i=1:n
        p(1)=i; j0=1; minv=inf(m+1,1); used=false(m+1,1);
        while true
            used(j0)=true; i0=p(j0); delta=inf; j1=0;
            for j=2:m+1
                if ~used(j)
                    cur=cost(i0,j-1)-u(i0+1)-v(j);
                    if cur<minv(j), minv(j)=cur; way(j)=j0; end
                    if minv(j)<delta, delta=minv(j); j1=j; end
                end
            end
            for j=1:m+1
                if used(j)
                    if p(j)~=0, u(p(j)+1)=u(p(j)+1)+delta; end
                    v(j)=v(j)-delta;
                else
                    minv(j)=minv(j)-delta;
                end
            end
            j0=j1; if p(j0)==0, break; end
        end
        while true
            j1=way(j0); p(j0)=p(j1); j0=j1;
            if j0==1, break; end
        end
    end
    assignment=zeros(n,1);
    for j=2:m+1, if p(j)~=0, assignment(p(j))=j-1; end, end
end

function nmi=nmi_score(a,b)
    a=a(:); b=b(:);
    [~,~,a]=unique(a,'stable'); [~,~,b]=unique(b,'stable');
    A=max(a); B=max(b); n=numel(a); M=zeros(A,B);
    for i=1:n, M(a(i),b(i))=M(a(i),b(i))+1; end
    P=M/n; pa=sum(P,2); pb=sum(P,1); mi=0;
    for i=1:A
        for j=1:B
            if P(i,j)>0, mi=mi+P(i,j)*log(P(i,j)/(pa(i)*pb(j))); end
        end
    end
    Ha=-sum(pa(pa>0).*log(pa(pa>0)));
    Hb=-sum(pb(pb>0).*log(pb(pb>0)));
    if Ha+Hb<=eps, nmi=1; else, nmi=2*mi/(Ha+Hb); end
end

function ari=ari_score(a,b)
    a=a(:); b=b(:);
    [~,~,a]=unique(a,'stable'); [~,~,b]=unique(b,'stable');
    A=max(a); B=max(b); n=numel(a); M=zeros(A,B);
    for i=1:n, M(a(i),b(i))=M(a(i),b(i))+1; end
    c2=@(x)x.*(x-1)/2;
    cells=sum(c2(M(:))); rows=sum(c2(sum(M,2))); cols=sum(c2(sum(M,1)));
    total=c2(n); expected=rows*cols/max(total,eps);
    denom=0.5*(rows+cols)-expected;
    if abs(denom)<=eps, ari=0; else, ari=(cells-expected)/denom; end
end
