% Driver file for an EN simulation for cortical maps:
% 2D cortex in stimulus space (VFx,VFy,OD,ORx,ORy) where VF = visual
% field, OD = ocular dominance and OR = orientation.
%
% Note: the net is considered nonperiodic. For a periodic net, the
% starting centroids and the VF variables of the training set must be
% wrapped; thus, every VF variable must be coded using two variables.
%
% To set up the parameters (training set, initial centroids, net
% configuration and training parameters), edit the lines enclosed by
% "USER VALUES"; the most likely values to be edited are marked "***".
% Alternatively, if the variable ENsetupdone is set to 'yes', no
% parameters are set up; this is useful to use previously set parameters
% (e.g. read from a file or set manually in Matlab).

% ENtraining determines the training set Tr that is used at each iteration
% in the ENcounter loop. It can be one of these:
% - 'canonical': use T, i.e., a uniform grid in (VFx,VFy,OD,ORt).
% - 'noisy': add uniform noise to the canonical T at each iteration.
% - 'uniform1': generate a uniform sample in the (rectangular) domain of
%   (VFx,VFy,OD,ORt,ORr) at each iteration.
% - 'uniform2': like uniform1, but force OD to be binary in {-l,l}
%   and ORr = r.
% The canonical training set T is created independently of the value of
% ENtraining and saved in ENfilename. It is used for plotting, as a
% scaffolding for the underlying continuous set of training vectors.
% 'noisy' and 'uniform*' approximate online training over the continuous
% domain of (VFx,VFy,OD,ORt).
function stats=myV1driver(seed,ENproc,ENfilename0,ENfilename,non_cortical_LR,cortical_VF,cortical_shape,uniform_LR,test_dw,test_dh,alpha,beta,iters,max_it,Kin,Kend,Nx,nvf,rx,Ny,ry,l,NOD,rOD,r,NOR,ODnoise,ODabsol,nG,G,ecc,nod,a,b,k,fign,plots,new,saveLR,separateData,plotting,heteroAlpha,equi,VFpath,weightType)
	irange = fign;
    datafileabsolpath = [pwd,'/',ENfilename0,'-',ENfilename,'.mat'];
	stream = RandStream('mt19937ar','Seed',seed);
	[tmp, tmp] = mkdir([ENfilename0,'/',ENfilename]);

    if ~exist(datafileabsolpath,'file') || new
        ENtraining = 'canonical';

        % ----------------------------- USER VALUES -----------------------------
        % Processing of intermediate (historical) parameters:
        %ENproc = 'varplot';		% One of 'var', 'save', 'varplot', 'saveplot'
        %     ENproc = 'saveplot';
        
        % Canonical training set: uniform grids for retinotopy, OD and OR as
        % follows:
        % -- T is created for ploting purposes
        % -- retinotopy to be generated independently
        %%%% copy from parameters.m
        
        % - OR: NOR values in a periodic interval [-pi/2,pi/2] with modulus r,
        rORt = [-pi/2 pi/2];		% Range of ORtheta
        rORr = [0 r];			% Range of ORr
        tmp1 = linspace(rORt(1),rORt(2),NOR+1); tmp1 = tmp1(1:NOR);
		NORr = 1;
        %dOR = diff(rORt)/(NOR-1)*r;
        % - SF: spatial frequency, same as OD, to be tested with larger NSF
        lSF = 0.14;
        NSF = 2;
        rSF = [-lSF lSF];
        dSF = diff(rSF)/(NSF-1);

		if isempty(VFpath)
			if cortical_shape
				%assert(2*Nx == Ny);
				if strcmp(equi,'VF')
					x_vec0 = linspace(rx(1),rx(2),Nx+1);
					y_vec0 = linspace(ry(1),ry(2),Ny+1);
					% use midpoints for training
					x_vec = midpoints(x_vec0);
					y_vec = midpoints(y_vec0);

					halfNy = Ny/2;
					vfx = linspace(0,ecc,Nx+1);
					vfy = linspace(0,ecc,halfNy+1);
					mvfx = midpoints(vfx);
					mvfy = midpoints(vfy);
					[vfx, vfy] = meshgrid(vfx, vfy);
					[vfpolar, vfecc] = cart2pol(vfx, vfy);
        	        w = dipole_ext(vfecc,vfpolar,a,b,k) - k*log(a/b);
        	        x_cortex = real(w);
        	        y_cortex = imag(w);
					VFweights = ones(halfNy,Nx);
					if heteroAlpha ~= 0
        	            p = zeros(2,4);
					    % shape is (y,x)
						for i = 1:halfNy
							for j = 1:Nx
        	                    p(:,1) = [x_cortex(i,j),     y_cortex(i,j)]';
        	                    p(:,2) = [x_cortex(i,j+1),   y_cortex(i,j+1)]';
        	                    p(:,3) = [x_cortex(i+1,j+1), y_cortex(i+1,j+1)]';
        	                    p(:,4) = [x_cortex(i+1,j),   y_cortex(i+1,j)]';
								switch weightType
									case 'area'
        	                    		VFweights(i,j) = find_area(p);
									case 'min'
        	                    		VFweights(i,j) = min([norm(p(:,1) - p(:,2)), norm(p(:,2) - p(:,3)), norm(p(:,3) - p(:,4)), norm(p(:,4) - p(:,1))]);
									otherwise
								end
							end
						end
						if heteroAlpha == 1
        	            	VFweights_hlf = VFweights;
						else
							assert(heteroAlpha == -1);
        	            	VFweights_hlf = 1./VFweights;
						end
					else
        	            VFweights_hlf = VFweights;
					end
					VFweights_hlf(vfecc(1:halfNy,1:Nx)>ecc) = 0.0;
					VFweights = [flipud(VFweights_hlf); VFweights_hlf];
					%darea = @(e, p) dblock(e,p,k,a,b);
        	        %disp(['integral sum: ', num2str(integral2(darea,0,ecc,-pi/2,pi/2))]);
        	        %disp(['estimated sum: ', num2str(sum(sum(VFweights)))]);

					alpha_v = repmat(VFweights(:), NOD*NOR*NORr,1);
					alpha = alpha_v/sum(alpha_v)*length(alpha_v) * alpha;

					[vfx, vfy] = meshgrid(midpoints(linspace(0,ecc,Nx+1)), midpoints(linspace(0,ecc,halfNy+1)));
					[vfpolar, vfecc] = cart2pol(vfx, vfy);
        	        w = dipole_ext(vfecc,vfpolar,a,b,k) - k*log(a/b);
        	        xx_cortex = real(w);
        	        yy_cortex = imag(w);
				else
					assert(strcmp(equi,'cortex'));
					halfNy = Ny/2;
					x2ecc = @(x) (a-b*exp(x/k))./(exp(x/k)-1);

					v_max = dipole_ext(ecc,0,a,b,k)-k*log(a/b);
					xv_cortex0 = linspace(0,v_max,Nx+1);
					xv_cortex = midpoints(xv_cortex0);
					yv_cortex0 = linspace(0,v_max,halfNy+1);
					yv_cortex = midpoints(yv_cortex0);
					% equi-distance cortex ecc
					ecc_x0 = x2ecc(xv_cortex0+k*log(a/b));
					ecc_x0(1) = 0;
					ecc_x = x2ecc(xv_cortex+k*log(a/b));

					ecc_y0 = x2ecc(yv_cortex0+k*log(a/b));
					ecc_y0(1) = 0;
					ecc_y = x2ecc(yv_cortex+k*log(a/b));

					% use transformed midpoints for training points
					x_vec = rx(1) + ecc_x*(rx(2)-rx(1))/ecc;
					y_vec = ry(1) + ([-fliplr(ecc_y), ecc_y]+ecc)*(ry(2)-ry(1))/(2*ecc);

					[vfx, vfy] = meshgrid(ecc_x0, ecc_y0);
					[vfpolar, vfecc] = cart2pol(vfx, vfy);
        	        w = dipole_ext(vfecc,vfpolar,a,b,k) - k*log(a/b);
        	        x_cortex = real(w);
        	        y_cortex = imag(w);
					VFweights = ones(halfNy,Nx);
					if heteroAlpha ~= 0
        	            p = zeros(2,4);
						for i = 1:halfNy
							for j = 1:Nx
        	                    p(:,1) = [x_cortex(i,j),     y_cortex(i,j)]';
        	                    p(:,2) = [x_cortex(i,j+1),   y_cortex(i,j+1)]';
        	                    p(:,3) = [x_cortex(i+1,j+1), y_cortex(i+1,j+1)]';
        	                    p(:,4) = [x_cortex(i+1,j),   y_cortex(i+1,j)]';
								switch weightType
									case 'area'
        	                    		VFweights(i,j) = find_area(p);
									case 'min'
        	                    		VFweights(i,j) = min([norm(p(:,1) - p(:,2)), norm(p(:,2) - p(:,3)), norm(p(:,3) - p(:,4)), norm(p(:,4) - p(:,1))]);
									otherwise
								end
							end
						end
						if heteroAlpha == 1
        	            	VFweights_hlf = VFweights;
						else
							assert(heteroAlpha == -1);
        	            	VFweights_hlf = 1./VFweights;
						end
					else
        	            VFweights_hlf = VFweights;
					end
					VFweights_hlf(vfecc(1:halfNy,1:Nx)>ecc) = 0.0;
					VFweights = [flipud(VFweights_hlf); VFweights_hlf];
					%darea = @(e, p) dblock(e,p,k,a,b);
        	        %disp(['integral sum: ', num2str(integral2(darea,0,ecc,-pi/2,pi/2))]);
        	        %disp(['estimated sum: ', num2str(sum(sum(VFweights)))]);

					alpha_v = repmat(VFweights(:), NOD*NOR*NORr,1);
					alpha = alpha_v/sum(alpha_v)*length(alpha_v) * alpha;
					[vfx, vfy] = meshgrid(ecc_x, ecc_y);
					[vfpolar, vfecc] = cart2pol(vfx, vfy);
        	        w = dipole_ext(vfecc,vfpolar,a,b,k) - k*log(a/b);
        	        xx_cortex = real(w);
        	        yy_cortex = imag(w);
				end
			else
				x_vec = midpoints(linspace(rx(1),rx(2),Nx+1));
				%x_vec = cumsum([0.25, 0.5*ones(1,Nx/2-1), ones(1,Nx/2)]);
				%x_vec = rx(1) + x_vec/(max(x_vec)+0.5) * (rx(2) - rx(1));
				y_vec = midpoints(linspace(ry(1),ry(2),Ny+1));
			end
        	T = ENtrset('grid',zeros(1,5),...
        	    x_vec,...	% VFx
        	    y_vec,...	% VFy
        	    linspace(rOD(1),rOD(2),NOD),...	% OD
        	    tmp1,...				% ORtheta
        	    linspace(rORr(1),rORr(2),NORr),stream);%,... % ORr
		else
			assert(exist(VFpath,'file'));
			fid = fopen(VFpath,'r');
			limits = fread(fid,4,'double')';
			data = fread(fid,[3,inf],'double')';
			x_cortex = data(:,1);
			y_cortex = data(:,2);
			VFweights_hlf = data(:,3);
			np = size(data,1);
			T = ENtrset('grid',zeros(1,4),...
				linspace(1,np*2),... VF points
        	    linspace(rOD(1),rOD(2),NOD),...	% OD
        	    tmp1,...				% ORtheta
        	    linspace(rORr(1),rORr(2),NORr),stream);%,... % ORr
			T = [T(:,1), T];
			lengthOfRest = NOD*NOR*NORr;
			scaledPoints = zeros(np*2,2);
			x_vec = [ x_cortex; x_cortex];
			y_vec = [-y_cortex; y_cortex];
			scaledPoints(:,1) = rx(1) + (x_vec - limits(1)) / (limits(2)-limits(1)) * (rx(2)-rx(1));
			scaledPoints(:,2) = ry(1) + (y_vec - limits(3)) / (limits(4)-limits(3)) * (ry(2)-ry(1));
			for i = 1:lengthOfRest
				thisRange = (i-1)*2*np + 1: i*2*np;
				T(thisRange,1:2) = scalePoints;
			end
		end
		dOD = 2*l;
		dx = mean(diff(x_vec));
		dy = mean(diff(y_vec));
		disp(['#',num2str(irange),': dx = ',num2str(dx), ', dy = ', num2str(dy), ', dOD = ', num2str(dOD)]);
        % For non-rectangular cortex shapes, create a suitable Pi here:
        resol = 10;
        if cortical_shape
			manual_LR = ~non_cortical_LR && ~uniform_LR;
            [Pi, W, LR, VF] = myCortex(stream, G, rx, x_cortex, ry, y_cortex, VFweights_hlf, ecc, a, b, k, resol, nod, rOD*ODabsol, ODnoise, manual_LR, fign, [ENfilename0,'/',ENfilename]);
			fID = fopen([ENfilename0,'/',ENfilename,'-RefVF.bin'],'w');
			if isempty(VFpath)
				fwrite(fID, xx_cortex(:), 'double');
				fwrite(fID, yy_cortex(:), 'double');
			else
				fwrite(fID, x_cortex, 'double');
				fwrite(fID, y_cortex, 'double');
			end
			fwrite(fID, VFweights_hlf, 'double');
			fclose(fID);
        else
            Pi = zeros(G);
            Pi(1+test_dw*nG:G(1)-nG*test_dw, 1+test_dh*nG:G(2)-nG*test_dh) = 1;
            %Pi = [];			% Don't disable any centroid
            W = G(1)-nG*test_dw;			% Net width along 1st var. (arbitrary units)
            LR = zeros(G(1),G(2));
		end
        if non_cortical_LR
            LR = ones(G(1),G(2));
            OD_width = 20; % in pixels
            for i=1:round(G(1)/OD_width)
                if mod(i,2) == 0
                    is = (i-1)*OD_width+1;
                    ie = min(is + OD_width-1, G(1));
                    LR(is:ie, :) = -1;
                end
            end
            LR = LR*l*ODabsol + ODnoise*randn(stream,G(1),G(2));
            LR(LR < -l) = -l;
            LR(LR > l) = l;
        end
		if saveLR
			fID = fopen([ENfilename0,'/',ENfilename,'-LR_Pi.bin'],'w');
			fwrite(fID, Pi, 'int');
			fclose(fID);
        end
        
        %T = ENtrset('grid',zeros(1,3),...
        %    linspace(rx(1),rx(2),Nx),...	% VFx
        %    linspace(ry(1),ry(2),Ny),...	% VFy
        %    linspace(rOD(1),rOD(2),NOD),stream);
        
        %linspace(rSF(1),rSF(2),NSF),... % SF
        [N,D] = size(T);
        
        % match feature with id
        id = struct('VFx',1,'VFy',2,'OD',3,'ORx',4,'ORy',5,'OR',6,'ORr',7);
		T_vec = cell(D,1);
		T_vec{1} = x_vec;
		T_vec{2} = y_vec;
		T_vec{3} = [-l, l];
        tmp1 = linspace(rORt(1),rORt(2),NOR+1); tmp1 = tmp1(1:NOR);
		T_vec{6} = tmp1
		T_vec{7} = [r];
		[T_vec{4}, T_vec{5}] = pol2cart(2*tmp1, r);
        %id = struct('VFx',1,'VFy',2,'OD',3,'ORx',4,'ORy',5,'OR',6,'ORr',7);
        %id = struct('VFx',1,'VFy',2,'OD',3);
        % Ranges of stimuli variables (for greyscales, etc.)
        % v = [1 rx;2 ry;3 rOD];
        v = [1 rx Nx;2 ry Ny;3 rOD NOD;4 -r r 2;5 -r r 2;6 -pi/2 pi/2 NOR;7 0 r NORr];
        %v = [1 NaN NaN;2 NaN NaN;3 -l l;4 NaN NaN;5 NaN NaN;6 -pi/2 pi/2;7 0 r];
        %v = [1 rOD; 2 -r r; 3 -r r; 4 -pi/2 pi/2; 5 0 r]; % -- last 2 rows: OR augmented by polar
        disp([num2str(size(T,1)),' references(cities) x ',num2str(size(T,2)),' features']);
        if isfield(id, 'ORx')
            [tmp1,tmp2] = pol2cart(2*T(:,id.ORx),T(:,id.ORy)); % polar coords to Cartesian coords
            T(:,[id.ORx,id.ORy]) = [tmp1 tmp2];			% ORx, ORy
        end
        
        % The training set is slightly noisy to avoid symmetry artifacts.
        
        %     T = T + (rand(stream,size(T))-1)/10000;		% Tiny noise
        
        % Training parameters
        max_cyc = 1;		% Number of cycles per annealing iteration
        min_K = eps;		% Smallest K before K is taken as 0
        tol = -1;			% Smallest centroid update
        method = 'Cholesky'		% Training method
        %method = 'gradient';		% Training method
        annrate = (Kend/Kin)^(1/(max_it*iters-1));	% Annealing rate
        disp(['annealing rate: ', num2str(annrate)]);
        Ksched = Kin*repmat(annrate^max_it,1,iters).^(0:iters-1);
        %Ksched = Ksched(4:10);
        %iters = 7;
        
        % Elastic net configuration: 2D
        %     G = [64 104]*nG;		% Number of centroids ***    
        %     G = [64 96]*nG;		% Number of centroids ***   3.4
        bc = 'nonperiodic'		% One of 'nonperiodic', 'periodic' ***
        p = 1;			% Stencil order (of derivative) ***
        s = {[0 -1 1],[0;-1;1]};	% Stencil list ***
        L = length(G); M = prod(G);
        %%% parameters that does not matter
        % test_boundary = false;
        % prefix = '';
        % right_open = false;
        % B_width = 1;
        %%% acquire boundary points, for exact boundary setup for matrix S
        %[~, ~, B1_ind, ~, ~, ~, ~] = set_bc(reshape(logical(Pi),G(1),G(2)), B_width, right_open, test_boundary, prefix);
        
        %     pause;
        %gridVF = zeros(G(1), G(2),2);
        %for iy =1:G(2)
        %    gridVF(logical(Pi(:,iy)),iy,1) = linspace(rx(1),rx(2),sum(Pi(:,iy)))';
        %end
        %for ix =1:G(1)
        %    gridVF(ix,logical(Pi(ix,:)),2) = linspace(ry(1),ry(2),sum(Pi(ix,:)));
        %end
        Pi = Pi(:)';
        [S,DD,knot,A,LL] = ENgridtopo(G,bc,Pi,s{:});
        normcte = ENstennorm(G,W,p,s{:});	% Normalisation constant
        % $$$   % Use normcte = 1; when disregarding the step size and resolution:
        % $$$   normcte = 1;

        % Initial elastic net: retinotopic with some noise and random, uniform
        % OD and OR.
        
       	if cortical_VF && cortical_shape
            mu = reshape(VF, M, 2);
		else
        	mu = ENtrset('grid',zeros(1,2),...		% Small noise
        	    linspace(rx(1),rx(2),G(1)),...	% VFx
        	    linspace(ry(1),ry(2),G(2)),stream);	% VFy
		end
        %    mu = reshape(gridVF, M, 2);
        if uniform_LR
            mu = [mu, ENtrset('uniform',rOD,M,stream)];
        else
            mu = [mu reshape(LR,M,1)];		% OD
        end
        
        % mu = [mu ENtrset('uniform',...
        %     [-pi/2,pi/2;...		% ORtheta
        %     0,r],...        %ORr
        %     M,stream)];
        
        % mu = ENtrset('grid',zeros(1,2),...		% Small noise
        %    linspace(rx(1),rx(2),G(1)),...	% VFx
        %    linspace(ry(1),ry(2),G(2)),stream);	% VFy
        mu = [mu ENtrset('uniform',[-pi/2 pi/2;...		% ORtheta
           0 r],... % ORr
           M,stream)];
        %         -lSF lSF],...   % -- SF
        %         M)];
        if isfield(id, 'ORx')    
            [tmp1,tmp2] = pol2cart(2*mu(:,id.ORx),mu(:,id.ORy));
            mu(:,[id.ORx, id.ORy]) = [tmp1 tmp2];			% ORx, ORy
        end
        if ~isempty(Pi), mu(Pi==0,:) = 0; end;
        disp([num2str(size(mu,1)),' centroids x ',num2str(size(mu,2)),' features']);
        
        betanorm = beta*normcte;	% Normalised beta
        
        % Actual ranges of centroids (for axes)
        if isfield(id, 'OR')
            [tmp1,tmp2] = cart2pol(mu(:,id.ORx),mu(:,id.ORy));
            tmp1 = tmp1 / 2;
            murange = [mu tmp1 tmp2];
        else
            murange = mu;
        end
        murange = [min(murange);max(murange)];
        
        % Initialisation
        switch ENproc
            case {'var','varplot'}
                % Store parameters in a single variable called ENlist, which is
                % eventually saved in a file (whose name is contained in ENfilename).
                % This is useful for small nets.
                % ENlist(i) contains {mu,stats}, for i = 1, 2, etc.
                ENlist = struct('mu',mu,'stats',struct(...
                    'K',NaN,'E',[NaN NaN NaN],'time',[NaN NaN NaN],...
                    'cpu','','code',1,'it',0));
                % ENlist(1) contains only the initial value of mu.
                % I would prefer to have the initial value of mu in ENlist(0),
                % but crap Matlab does not allow negative or zero indices.
                if strcmp(ENproc,'varplot')
                    myV1replay(G,bc,ENlist,v,1,T,T_vec,Pi,murange,id);
                end
            case {'save','saveplot'}
                % Save parameters in separate files with names xxx0001, xxx0002, etc.
                % (assuming ENfilename = 'xxx'). File xxx0000 contains the simulation
                % parameters (including the initial value of mu). Once the training
                % loop finishes, all files are collected into a single variable. Thus,
                % the final result is the same as with 'var'.
                % This is useful when memory is scarce, and also more robust, since the
                % learning loop may be resumed from the last file saved in case of a
                % crash.
                eval(['save ' ENfilename0 '-' ENfilename '0000.mat ENfilename '...
                    'Nx rx dx Ny ry dy NOD rOD dOD l NOR rORt rORr r seed v ' ...
                    'N D L M G bc p s T T_vec Pi S DD knot A LL mu Kin Kend iters Ksched '...
                    'alpha beta annrate max_it max_cyc min_K tol method '...
                    'W normcte betanorm']);
                if strcmp(ENproc,'saveplot')
                    ENlist = struct('mu',mu,'stats',struct(...
                        'K',NaN,'E',[NaN NaN NaN],'time',[NaN NaN NaN],...
                        'cpu','','code',1,'it',0));
                    myV1replay(G,bc,ENlist,v,1,T,T_vec,Pi,murange,id);
                end
            otherwise
                % Do nothing.
        end
        
        if exist('ENtr_annW')
            disp('Will use ENtr_ann2');
        else
            disp('Will use ENtr_ann1');
        end
        Tr = T;
        % Learning loop with processing of intermediate (historical) parameters:
        for ENcounter = 1:length(Ksched)
            
            % Generate training set for this iteration. For 'uniform*' we use the
            % same N as for the canonical T.
            switch ENtraining
                case 'noisy'
					% TO DO
                    % Noise level to add to T as a fraction of the smallest variable range
                    Tr_noise = 0.5;
                    Tr = T + (rand(stream,size(T))-0.5)*Tr_noise.*min(diff([rx;ry;rOD;rORr]'));
                otherwise
                    % Do nothing.
            end
            
            % Update parameters:
        %     disp(['K#',num2str(ENcounter),' = ', num2str(Ksched(ENcounter))]);
            [mu,stats] = ENtr_ann(Tr,S,Pi,mu,Ksched(ENcounter),alpha,betanorm,...
                annrate,max_it,max_cyc,min_K,tol,method,0,stream);
            if isfield(id, 'OR')
                [tmp1,tmp2] = cart2pol(mu(:,id.ORx),mu(:,id.ORy));
                tmp1 = tmp1 / 2;
                murange = [mu tmp1 tmp2;murange];
            else
                murange = mu;
            end
            murange = [min(murange);max(murange)];
            
            % Process parameters:
            switch ENproc
                case {'var','varplot'}
                    ENlist(ENcounter+1).mu = mu;
                    ENlist(ENcounter+1).stats = stats;
                    if strcmp(ENproc,'varplot')
                        myV1replay(G,bc,ENlist,v,ENcounter+1,T,T_vec,Pi,murange,id);
                    end
                case {'save','saveplot'}
                    save(sprintf('%s-%s%04d.mat',ENfilename0,ENfilename,ENcounter),'mu','stats','murange');
                    if strcmp(ENproc,'saveplot')
                        myV1replay(G,bc,struct('mu',mu,'stats',stats),v,1,T,T_vec,Pi,murange,id);
                    end
                otherwise
                    % Do nothing.
            end
            disp(['K=',num2str(Ksched(ENcounter))]);
        end
        
        % Save results
        switch ENproc
            case {'var','varplot'}
                eval(['save ' ENfilename0 '-' ENfilename '.mat ENfilename '...
                    'Nx rx dx Ny ry dy NOD rOD dOD l NOR rORt rORr r seed v ' ...
                    'ENlist murange id ' ...
                    'N D L M G bc p s T T_vec Pi S DD knot A LL Kin Kend iters Ksched '...
                	'alpha beta annrate max_it max_cyc min_K tol method '...
                    'W normcte betanorm']);
            case {'save','saveplot'}
                % Collect all files into a single one:
                load(sprintf([ENfilename0,'-%s%04d.mat'],ENfilename,0));
                ENlist = struct('mu',mu,'stats',struct(...
                    'K',NaN,'E',[NaN NaN NaN],'time',[NaN NaN NaN],...
                    'cpu','','code',1,'it',0));
                for ENcounter = 1:length(Ksched)
                    load(sprintf('%s-%s%04d.mat',ENfilename0,ENfilename,ENcounter));
                    ENlist(ENcounter+1).mu = mu;
                    ENlist(ENcounter+1).stats = stats;
                end
                eval(['save ' ENfilename0 '-' ENfilename '.mat ENfilename '...
                    'Nx rx dx Ny ry dy NOD rOD dOD l NOR rORt rORr r seed v ' ...
                    'ENlist murange id ' ...
                    'N D L M G bc p s T T_vec Pi S DD knot A LL Kin Kend iters Ksched '...
                    'alpha beta annrate max_it max_cyc min_K tol method '...
                    'W normcte betanorm']);
                unix(['rm ' ENfilename0 '-' ENfilename '????.mat']);
            otherwise
                % Do nothing.
        end

        % Plot some statistics for the objective function value and computation time
        switch ENproc
            case {'varplot','saveplot'}
                myV1replay(G,bc,ENlist,v,1,T,T_vec,Pi,murange,id,[],[],[100, 102]);
            otherwise
                % Do nothing.
        end
        disp(['ratio of L/R = ', num2str(sum(mu(logical(Pi),id.OD) < 0)/sum(mu(logical(Pi),id.OD) > 0))]);
    else
        disp('data exist');
        load([ENfilename0,'-',ENfilename,'.mat']);
    end
    if plots
        figlist = [1,2,3,4,5,7,13,14,20,21,100,102,34, 40, 41, 50, 54, 60];
    else
        figlist = [];
    end
	statsOnly = true;
	right_open = cortical_shape;
    stats = myV1stats(stream,G,bc,ENlist,v,plotting,T,T_vec,Pi,murange,id,[],[ENfilename0,'/',ENfilename,'.png'],figlist,statsOnly,right_open,separateData);
	if saveLR
		fID = fopen([ENfilename0,'/',ENfilename,'-LR_Pi.bin'],'a');
		fwrite(fID, ENlist(end).mu(:,id.OD), 'double');
		fclose(fID);
	end
end
function f = midpoints(v)
	if size(v,1) == 1
		f = (v(1:end-1) + v(2:end))/2;
	else	
		v = zeros(size(v,1), size(v,2)-1);
		f = (v(:,1:end-1) + v(:,2:end))/2;
	end
end
function area = find_area(p)
    assert(size(p,1) == 2);
    a = p(:,2) - p(:,1);
    b = p(:,4) - p(:,1);
    area = abs(a(1)*b(2) - a(2)*b(1));
    a = p(:,2) - p(:,3);
    b = p(:,4) - p(:,3);
    area = area + abs(a(1)*b(2) - a(2)*b(1));
    area = area/2;
end
