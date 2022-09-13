% [mu,stats] = ENtr_ann2(exchange_nm,T,S[,Pi,mu,Kin,alpha,beta,ibeta,annrate,
%           max_it,max_cyc,min_K,tol,method,noise])
%
% --------------------------------------------------------------------------
% NOTE: this is the same function as ENtr_ann1.m except that the
% vectorised computations involving matrix W (which is very large) are
% done without explicitly computing W, using loops, and implemented in C
% (as a MEX-file called ENtr_annW.mex*).
% When the condition "K < min_K" holds, I don't use C code because it
% would make ENtr_annW too messy. I use the vectorised code instead,
% which may provoke an out-of-memory error. However, the condition "K <
% min_K" should be generally avoided, at least for cortical map
% modelling.
% --------------------------------------------------------------------------
%
% Annealed training for an elastic net with arbitrary topology
% (i.e., with an arbitrary tension term).
%
% The elastic net model has been most notably applied to the
% Travelling Salesman Problem and the maps of visual cortex.
%
% Given a training set T ("cities" in the Travelling Salesman Problem
% or "stimuli values" in a cortical map model), a topology matrix
% S (typically derived from a discretised differential operator) and
% possibly initial model parameters (reference vectors mu and standard
% deviation K), it returns a trained elastic net model using an annealing
% algorithm. That is, the standard deviation K is slowly annealed towards
% zero to ensure that the net interpolates the cities.
%
% Several training algorithms are provided (all with annealing),
% including the original Durbin and Willshaw algorithm. However, we
% recommend the algorithm based on direct inversion by Cholesky
% factorisation, since it is more robust while only slightly slower.
%
% Care must be taken to appropriately construct the training set and the
% topology matrix; separate functions ENtrset, ENgridtopo and ENgraph are
% provided for this.
%
% The reference vectors are associated with a low-dimensional grid
% (usually 1D or 2D) but we do not need that for training. All the
% information for computing the tension term both in the objective
% function and in the learning rule is in the matrix S; that is, the
% connectivity information (periodic / nonperiodic boundary conditions
% and neighbourhood relationships) and the functional form of the
% tension term (finite difference order and scheme).
%
% The schedule of K values can be controlled with the arguments Kin,
% annrate, max_it and max_cyc: max_cyc cycles are performed with the
% same value of K, which, upon the next iteration (up to max_it) is then
% changed to the next value of K (either the next one in the list Kin or
% the annealed one, K*annrate).
%
% ENtr_ann can also be controlled externally to e.g. plot or save
% intermediate values of mu:
%   for {appropriate values for Kin, annrate, max_it, max_cyc}
%     [mu,stats] = ...
%       ENtr_ann(T,S,mu,Kin,alpha,beta,annrate,min_K,tol,max_it,max_cyc);
%     plot/save mu, stats.K, etc.
%   end
%
% References:
%
%   Richard Durbin and David Willshaw: "An Analogue Approach to the
%   Traveling Salesman Problem Using an Elastic Net Method". Nature
%   326(6114):689-691 (Apr. 16, 1987).
%
%   Richard Durbin, Richard Szeliski and Alan Yuille: "An Analysis of
%   the Elastic Net Approach to the Traveling Salesman Problem". Neural
%   Computation 1(3):348-358 (1989).
%
%   M. A. Carreira-Perpinan and G. J. Goodhill: "Cortical map modelling
%   with elastic nets of arbitrary topology". In preparation.
%
% In:
%   T: NxD matrix of N D-dimensional row vectors (training set of
%      "cities" or "stimuli vectors" in D dimensions).
%   S: MxM sparse topology matrix for the tension term of the elastic net
%      objective function and learning rule.
%   Pi: 1xM list of mixing proportions (a priori probability of each grid
%      point, or centroid). They must be real numbers in [0,1] and add to 1,
%      although ENtr_ann normalises just in case. This allows to pass a
%      logical vector where 0/1 disables/enables a centroid, respectively,
%      which is useful to define non-rectangular cortical shapes in 2D.
%      Default: Pi(m) = 1/M (equiprobable Gaussian mixture). See also
%      the explanation in ENgridtopo. Note that Pi are fixed parameters of
%      the model: they are not estimated by ENtr_ann.
%   mu: MxD matrix of M D-dimensional row vectors containing the initial
%      value for the reference vectors, typically obtained with the
%      function ENtrset as a grid or uniform distribution. Default:
%      uniform in the range defined by T.
%   Kin: list of positive real numbers containing the values to be used
%      for the standard deviation parameter K of the elastic net. At
%      each iteration, the next value of the list is used; when no
%      more values are left, the current value is annealed towards
%      zero. For each value of K, max_cyc cycles are performed of the
%      training algorithm. Default: 0.5.
%   alpha: Nx1 vector of positive real numbers containing the weights of
%      the fitness term in the elastic net objective function (one per
%      training set point); if a single positive number, then all weights
%      are equal to it. Default: 1.
%   beta: positive real number containing the weight of the tension term
%      in the elastic net objective function. Default: 10.
%   annrate: real number in (0,1] containing the annealing rate for K in
%      the training algorithm, where K is multiplied by annrate during
%      the iterative procedure till it becomes zero (actually till it
%      becomes smaller than min_K, to avoid numerical problems). Default:
%      0.8.
%   max_it: positive integer number containing the maximum number of
%      iterations; at each such iteration, the value of K is
%      annealed. Default: 20.
%   max_cyc: positive integer number containing the number of
%      cycles for constant K per iteration. Default: 20.
%   min_K: positive real number containing the limit below which K is
%      taken as zero, in which case the iterative procedure stops and
%      the reference vectors are forced to interpolate the training
%      set (if M < N this is not possible, so some reference vectors
%      stay at equal distances from certain training set points). K
%      cannot be taken as zero directly because it leads to a division
%      by zero in the fitness term. Default: 0.000001.
%   tol: positive real number containing the minimum relative increase in
%      the objective function to keep iterating, e.g. tol = 0.0001 means 4
%      decimal places. Default: -1 (= don't use tol).
%   method: one of the following (default 'Cholesky'):
%      'gradient': one step down the gradient of the objective function.
%       This is the standard Durbin and Willshaw algorithm.
%      'Jacobi': Jacobi_it iterations of the Jacobi method for solving
%       linear systems (Jacobi_it = 5). For large Jacobi_it (around 10-20),
%       'Jacobi' converges to 'Cholesky'.
%      'Cholesky': direct jump to the zero-gradient point of the
%       objective function, but using Cholesky factorisation rather than
%       matrix inversion.
%      For each cycle, first the weight matrix W is computed and then the
%      centroids mu are updated (according to the method).
%   noise: amount of random uniform noise (in units of eps*mu) to add to mu
%      at every iteration (just once per iteration, in the last cycle).
%      This is useful to avoid metastable states if there are strong
%      symmetries in the net and/or training set. Default: 0 (no noise added).
% Out:
%   mu: MxD matrix of M D-dimensional row vectors containing the trained
%      reference vectors.
%   stats: structure containing the following statistics:
%      K: ?x1 historic list of values of the standard deviation
%       parameter K of the elastic net. This is not necessarily equal to
%       the list given in Kin (e.g. if Kin contained too few values,
%       annealed ones were added; or if min_K was attained, extra values
%       in Kin were ignored).
%      E: ?x3 historic list of values of the objective function of the
%       elastic net. Each row E(?,:) contains, in this order: the
%       objective function value; the fitness term (weights) value
%       (including the factor -alpha*K); and the tension term
%       (constraint) value (including the factor beta/2).
%      time: ?x3 historic list of computation times (in seconds). Each
%       row time(?,:) contains the times corresponding to the total
%       computation, the fitness term computation (the weights'
%       computation) and the tension term computation (the constraint
%       computation) (in parallel to E(?,:)). Note that what computations
%       correspond to which one of these two terms is only approximately
%       defined.
%       Also, the following holds: time(1,:) >= time(1,:) + time(1,:).
%      cpu: description of the computer and operating system used,
%       as given by `uname -a' in Unix systems or by the Matlab
%       `computer' command otherwise. This is useful to compare
%       computation times.
%      code: stopping reason (0: tolerance achieved; 1: maximum number of
%       iterations reached; 2: minimum value of K reached).
%      it: number of iterations performed.
%    K, E and time have "it" rows; row k corresponds to the mu resulting
%    from applying the algorithm with K(k).
%
% Notes:
%
% - It is possible to use a Gauss-Seidel algorithm (or even successive
%   overrelaxation) instead of a Jacobi one, and in fact it would
%   probably require about half of the Jacobi_it iterations. However, its
%   matrix form (convenient for Matlab) is inefficient, so since its only
%   advantage is a slight speedup, we do not offer it. Naturally, it
%   could be coded in C.
%
% - If Pi(m) = 0, denoting a disabled centroid, then mu(m,:) is the mean of
%   the training set T. We could set them to NaN instead, but this might
%   break other functions (EN*replay*, etc.). So when using mu, one should
%   check if Pi contains zeroes and decide what to do with any disabled
%   centroids.
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.
%
% See also ENtrset, ENgridtopo, ENgraph, ENplot, ENreplay.

% Copyright (c) 2002 by Miguel A. Carreira-Perpinan

function [mu,stats] = ENtr_ann2(exchange_nm,T,S,Pi,mu,Kin,alpha,beta,ibeta,annrate,...
max_it,max_cyc,min_K,tol,method,noise,stream)
if nargin < 15
	stream = RandStream.getGlobalStream;
end

[N,D] = size(T);
M = size(S,1);

% Argument defaults
if ~exist('Pi','var') || isempty(Pi)
	Pi = ones(1,M)/M;
	zPi = [];
else
	Pi = Pi/sum(Pi);    % Make sure the mixing proportions are normalised
	zPi = find(Pi==0);
end
meanT = mean(T);    % Default value for disabled centroids
if ~exist('mu','var') || isempty(mu)
	% The initial reference vectors are drawn from a uniform distribution
	% in the range of the training set.
	minT = min(T,[],1); maxT = max(T,[],1); rangeT = maxT - minT;
	mu = rangeT(ones(M,1),:) .* rand(stream,M,D) - minT(ones(M,1),:);
end
if ~exist('Kin','var') || isempty(Kin)
	Kin = 0.5;
end
if ~exist('alpha','var') || isempty(alpha)
	alpha = ones(N,1);
end
if all(size(alpha) == 1)
	alpha = repmat(alpha,N,1);
end
if ~exist('beta','var') || isempty(beta)
	beta = 10; 
end
if ~exist('annrate','var') || isempty(annrate)
	annrate = 0.8;
end
if ~exist('max_it','var') || isempty(max_it)
	max_it = 20;
end
if ~exist('max_cyc','var') || isempty(max_cyc)
	max_cyc = 20;
end
if ~exist('min_K','var') || isempty(min_K)
	min_K = 0.000001;
end
% Don't use "tol" with annealing
% $$$ if ~exist('tol','var') | isempty(tol) tol = 0.0001; end;
if ~exist('tol','var') || isempty(tol)
	tol = -1;
end
if ~exist('method','var') || isempty(method)
	method = 'Cholesky';
end
if ~exist('noise','var') || isempty(noise)
	noise = 0;
end
% Set the seed of the random number generator externally
% $$$ if ~exist('seed','var') | isempty(seed) seed = sum(100*clock); end;
% $$$ rand('state',seed);
if max_it<1
	code = 1;
	it = 0;
else
	code = -1;
	it = 1;
end

K = Kin(1); Kin = Kin(2:end);       % Pop first value of K from the list
Kout = []; E = []; time = []; Etension_vec = []; % Initialise stats list

if K >= min_K
	% We need this matrix for the learning rule (although typically, S is
	% symmetric and so SS = S):
	SS = (S+S')/2;
	% We need these matrices for the Jacobi learning rule:
	if strcmp(method,'Jacobi')
        if length(ibeta) == 0
		    JacobiD = beta * diag(SS);
		else
            JacobiLU = -beta * (triu(SS,1)+tril(SS,-1));
            JacobiD = zeros(M,D);
            JacobiLU = zeros(M,M,D);
            for i=1:D
		        JacobiD(:,i) = beta * ibeta(i) * diag(SS);
		        JacobiLU(:,:,i) = -beta * ibeta(i) * (triu(SS,1)+tril(SS,-1));
            end
        end
		Jacobi_it = 5;              % Number of Jacobi iterations
	end
end

%fprintf('Iterations performed:');
while code<0
	timenow1 = cputime;
	if K < min_K            % Min. value of K reached, we make K -> 0
		% NOTE: I don't use the C function ENtr_annW for this part because it
		% would be too messy, and K < min_K should be avoided, at least for
		% cortical map modelling.
		%
		% The following code is really like the usual elastic net learning
		% rule, but we avoid divisions by zero (term 1/(2*K*K)) and ensure
		% that the reference vectors interpolate the training set points.
		%
		% The case N > M leads to some reference vectors being shared by
		% the cities; the case N < M leads to a permutation weight matrix;
		% the case N = M should work in both ways, leading to a one-to-one
		% mapping between cities and reference vectors (except maybe in
		% pathological cases where a city may be equidistant from two
		% reference vectors or vice versa).
		timenow2 = cputime;
		if ~isempty(zPi)
			mu(zPi,:) = NaN;
        end
		W = ENsqdist(T,mu);
        disp(['K < min_K', num2str(min_K)]);
		if N > M            % More cities than reference vectors
			[temp_N_1b, temp_N_1a] = min(W,[],2);
			W2 = sparse((1:N)',temp_N_1a,ones(N,1),N,M);
			temp_N_1b = sum(W2,1);
			if ~isempty(zPi)
				temp_N_1b(zPi) = 1;
            end
			W2 = W2 ./ temp_N_1b(ones(1,N),:);  % Normalise
		else            % More reference vectors than cities
			[temp_M_1b, temp_M_1a] = min(W,[],1);
			tmp = ones(M,1); tmp(zPi) = 0;
			W2 = sparse(temp_M_1a,(1:M)',tmp,N,M);
		end
		mu = W2' * T;
		W = ENsqdist(T,mu);
		timenow2 = cputime-timenow2;
		% New value of the objective function:
		% - Tension term
		timenow3 = cputime;
        if length(ibeta) == 0
            itension_vec = sum(mu.*(S'*mu));
        else
            itension_vec = ibeta .* sum(mu.*(S'*mu));
        end
        Etension_vec = [Etension_vec; itension_vec];
        Etension = beta/2 * sum(itension_vec);
		timenow3 = cputime-timenow3;
		% - Fitness term
		if max(min(W,[],2)) > eps
			% At least one city is not matched by any node; then,
			% E -> Inf when K -> 0.
			E = [E;Inf Inf Etension];
		else
			% Every city is matched by at least one node, so the fitness term
			% is 0 and the objective function value is equal to the tension term.
			E = [E;Etension 0 Etension];
		end
		Kout = [Kout; 0];       % The value of K is taken as 0
		code = 2;           % Exit condition
	else                % Normal iteration, K >= min_K
		if ~isempty(E)
			E_old = E(end,1);
		end

		timenow2 = 0; timenow3 = 0;
		for cyc = 1:max_cyc
			% This is the inner loop of cycles: K is kept constant, mu is updated
			% max_cyc times. The values of K and E are not added to the respective
			% lists.

			% Compute weights in NxM matrix W.
			% $$$       % The code below ensures that no overflow occurs and is
			% $$$       % efficient in time but requires memory storage for two
			% $$$       % matrices of NxM, plus one more temporary matrix of NxM
			% $$$       % inside function ENsqdist, plus some more.
			% $$$       timenow2a = cputime;
			% $$$       W = ENsqdist(T,mu) / (2*K*K);
			% $$$       temp_N_1a = min(W,[],2);
			% $$$       W = W - temp_N_1a(:,ones(1,M));
			% $$$       W = exp(-W);
			% $$$       W = W .* Pi(ones(N,1),:);
			% $$$       temp_N_1b = sum(W,2);
			% $$$       W = W ./ temp_N_1b(:,ones(1,M));
			% $$$       W = W .* alpha(:,ones(1,M));
			% $$$
			% $$$       % Compute G matrix (sparse, diagonal, MxM).
			% $$$       G = diag(sparse(sum(W,1)));
			% $$$
			% $$$       % Compute "new average centroids" (MxD)
			% $$$       WT = W'*T;
			% $$$       timenow2 = timenow2 + (cputime-timenow2a);
			% $$$
			% $$$       % Compute fitness term of the objective function
			% $$$       fit_err = alpha' * ( log(temp_N_1b) - temp_N_1a );
			% The following function, implemented in C, computes WT, G and
			% fit_err without explicitly storing the weight matrix W.
			timenow2a = cputime;
			if exchange_nm
				[WT,G,fit_err] = ENtr_annW_exchanged(mu,T,K,alpha,Pi);
			else
				[WT,G,fit_err] = ENtr_annW(mu,T,K,alpha,Pi);
			end
			G = spdiags(G,0,M,M);
			timenow2 = timenow2 + (cputime-timenow2a);

			% Update reference vectors according to the method chosen
			timenow3a = cputime;
			switch method
			case 'gradient'
					% The following performs a single gradient step (times K) down
					% the objective function, as in the original Durbin and Willshaw
					% algorithm.
                    if length(ibeta) == 0
                        itenterm = SS * mu;
                    else
                        itenterm = zeros(size(mu));
                        for i=1:D
                            itenterm(:,i) = ibeta(i)* (SS * mu(:,i));
                        end
                    end
					tenterm = (beta * K) * itenterm;
					delta_mu = ...
					WT - G*mu - ...                 % Fitness term
					tenterm;                % Tension term
					mu = mu + delta_mu;         % Update reference vectors
					if ~isempty(zPi)
						mu(zPi,:) = 0;
					end
				case 'Jacobi'
					if ~isempty(zPi)
						% Trick for numerical instability (see 'Cholesky').
						G(sub2ind(size(G),zPi,zPi)) = 1;
					end
                    if length(ibeta) == 0
					    tmp = diag(sparse(1./(diag(G)/K + JacobiD)));
					    Jacobi1 = tmp * JacobiLU; 
                        Jacobi2 = (tmp/K) * WT;
					    for j = 1:Jacobi_it
					    	mu = Jacobi1 * mu + Jacobi2;
					    end
                    else
					    Jacobi1 = zeros(M,M,D);
                        Jacobi2 = zeros(M,D);
                        for i=1:D
					        tmp = diag(sparse(1./(diag(G)/K + JacobiD(:,i))));
					        Jacobi1(:,:,i) = tmp * JacobiLU(:,:,i); 
                            Jacobi2(:,i) = (tmp/K) * WT(:,i);
                        end
					    for j = 1:Jacobi_it
                            for i=1:D
					    	    mu(:,i) = Jacobi1(:,i) * mu(:,i) + Jacobi2(:,i);
                            end
					    end
                    end
				case 'Cholesky'
					if ~isempty(zPi)
						% If Pi(m) = 0 (centroid m is disabled) then G(m,m) = 0, which
						% leads to numerical instability, so we put a nonzero there (the
						% exact value is irrelevant as long as it is large enough).
						G(sub2ind(size(G),zPi,zPi)) = 1;
					end
					% Call A = G + K*beta*SS, which is sparse and symmetric.
					% The following, via the "\" operator, solves the system
					%   A * mu = W'*T
					% in a much more efficient and accurate way than inverting A, as
					% follows:
					% 1. Symmetric minimal degree preordering for the sparse matrix A.
					% 2. Cholesky factorisation of A as R'*R with R upper triangular.
					% 3. Lower triangular solve and then upper triangular solve by
					%    Gaussian elimination.
                    if length(ibeta) == 0
					    %mu = (G + (K*beta)*SS) \ WT;
                        for i=1:D
					        mu(:,i) = (G + (K*beta)*SS) \ WT(:,i);
                        end
                        % accuarcy difference
					    %mu0 = (G + (K*beta)*SS) \ WT;
                        %disp('mean difference') 
                        %mean((mu - mu0), 1)
                        %disp('std difference') 
                        %std((mu - mu0), 0, 1)
                    else
                        for i=1:D
					        mu(:,i) = (G + (K*beta)*ibeta(i)*SS) \ WT(:,i);
                        end
                    end
			end
			assert(~any(any(isnan(mu(Pi>0,:)))));
			% We set all disabled centroids to the centre-of-mass of the training
			% set by convention (this is a safe default value that ensures other
			% functions won't break).
			mu(zPi,:) = repmat(meanT,length(zPi),1);
			timenow3 = timenow3 + (cputime-timenow3a);
		end
		if noise ~= 0
			mu = mu + (rand(stream,size(mu))-0.5)*noise*eps*max(abs(mu(:)));
		end

		% New value of the objective function. Notes:
		% - Durbin et al. (1989) incorrectly use beta instead of beta/2.
		% - To compute the tension term, it is more efficient to do
		%     beta/2 * sum(sum(mu.*(S'*mu)))
		%   than
		%     beta/2 * trace(mu*mu'*S)
		%   because, even if S is sparse, mu*mu'*S is a very large MxM matrix.
		%   If S = DD'*DD for a discretised differential operator matrix DD
		%   then it can also be efficiently done as
		%     beta/2 * norm(DD*mu)
		Efitness = - K * fit_err;                   % Fitness term
		timenow3a = cputime;
        if length(ibeta) == 0
            itension_vec = sum(mu.*(S'*mu));
        else
            itension_vec = ibeta .* sum(mu.*(S'*mu));
        end
        Etension_vec = [Etension_vec; itension_vec];
        Etension = beta/2 * sum(itension_vec);			% Tension term
		timenow3 = timenow3 + (cputime-timenow3a);
		E = [E;Efitness+Etension Efitness Etension];
		Kout = [Kout; K];

		% Check whether exit condition is met
		if exist('E_old','var')
			if abs(E(end,1)-E_old)<tol*abs(E(end,1))
				code = 0;           % Relative error < tol => Tolerance achieved
			elseif it>=max_it
				code = 1;           % Max. no. iterations reached
			else
				it = it + 1;            % Continue iterating
				if ~isempty(Kin)
					K = Kin(1); Kin = Kin(2:end);   % Pop next value of K from the list
				else
					K = K * annrate;        % Anneal K
				end
				%             disp(['  current subK=', num2str(K)]);
			end
		else
			it = it + 1;            % Continue iterating
			if ~isempty(Kin)
				K = Kin(1); Kin = Kin(2:end);   % Pop next value of K from the list
			else
				K = K * annrate;        % Anneal K
			end
			%             disp(['  current subK=', num2str(K)]);
		end
	end
	time = [time;cputime-timenow1 timenow2 timenow3];
	assert(~any(isnan(Efitness)));
	assert(~any(isnan(Etension)));
	assert(~any(isinf(Efitness)));
	assert(~any(isinf(Etension)));
	%  fprintf(' %d',it);
end
%fprintf('\n');

if nargout > 1
	stats.K = Kout;
	stats.E = E;
	stats.time = time;
	stats.tension_vec = Etension_vec;
	if isunix
		[tmp,stats.cpu] = unix('uname -a');
	else
		stats.cpu = computer;
	end
	stats.code = code;
	stats.it = it;
end
end
