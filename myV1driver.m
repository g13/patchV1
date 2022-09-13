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
function stats = myV1driver(exchange_nm,seed,ENproc,ENfilename0,ENfilename,non_cortical_LR,cortical_VF,cortical_shape,uniform_LR,test_dw,test_dh,alpha,beta,ibeta,iters,max_it,Kin,Kend,Nx,rx,Ny,ry,l,NOD,rOD,r,NOR,ODnoise,ODabsol,nG,G,aspectRatio,nT,ecc,nod,a,b,k,fign,plots,new,saveLR,separateData,plotting,heteroAlpha,equi,weightType,VFpath,old, randSeed, figlist, rotate)
iRange = fign;
datafileabsolpath = [pwd,'/',ENfilename0,'-',ENfilename,'.mat'];
if randSeed
    stream = RandStream('mt19937ar','Seed',seed+fign);
else
    stream = RandStream('mt19937ar','Seed',seed);
end
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
    dOR = (rORt(2)-rORt(1))/NOR;
    tmp0 = linspace(rORt(1),rORt(2),NOR+1)-dOR/2;
    %tmp0 = linspace(rORt(1),rORt(2),NOR+1);
    tmp1 = midpoints(tmp0);
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
            w = dipole_ext(ecc,0,a,b,k) - k*log(a/b);
            mu_rx = [0, real(w)]
            w = dipole_ext(ecc,pi/2,a,b,k) - k*log(a/b);
            y0 = imag(w);
            mu_ry = [-y0, y0]
            switch equi
                case 'VF'
                    x_vec0 = linspace(rx(1),rx(2),Nx+1);
                    y_vec0 = linspace(ry(1),ry(2),Ny+1);
                    % use midpoints for training
                    x_vec = midpoints(x_vec0);
                    y_vec = midpoints(y_vec0);

                    halfNy = round(Ny/2);
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
                                %p(:,1) = [vfx(i,j),     vfy(i,j)]';
                                %p(:,2) = [vfx(i,j+1),   vfy(i,j+1)]';
                                %p(:,3) = [vfx(i+1,j+1), vfy(i+1,j+1)]';
                                %p(:,4) = [vfx(i+1,j),   vfy(i+1,j)]';
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
                    VFweights = [flipud(VFweights_hlf); VFweights_hlf]';
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

                case 'cortex'
                    halfNy = round(Ny/2);
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
                    x_vec0 = rx(1) + ecc_x0*(rx(2)-rx(1))/ecc;
                    y_vec0 = ry(1) + ([-fliplr(ecc_y0), ecc_y0(2:halfNy+1)]+ecc)*(ry(2)-ry(1))/(2*ecc);

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
                    VFweights = [flipud(VFweights_hlf); VFweights_hlf]';
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
            x_vec0 = linspace(rx(1),rx(2),Nx+1);
            y_vec0 = linspace(ry(1),ry(2),Ny+1);
            x_vec = midpoints(x_vec0);
            y_vec = midpoints(y_vec0);
            if rotate ~=0
                %x_vec = cumsum([0.25, 0.5*ones(1,Nx/2-1), ones(1,Nx/2)]);
                %x_vec = rx(1) + x_vec/(max(x_vec)+0.5) * (rx(2) - rx(1));
                if rotate < 0
                    rot_mat = [cos(-rotate), sin(-rotate); -sin(-rotate), cos(-rotate)]';
                else
                    rot_mat = [cos(rotate), sin(rotate); -sin(rotate), cos(rotate)]';
                end

                xx0 = [rx(1); rx(2); rx(1); rx(2)];
                yy0 = [ry(1); ry(1); ry(2); ry(2)];
                center = [mean(rx), mean(ry)]
                [xx0, yy0]
                corners = [xx0-center(1), yy0-center(2)]*rot_mat + repmat(center, [4,1])
                if rotate < 0
                    x_range = [min([rx(1); corners(:,1)]), max([rx(2); corners(:,1)])];
                    y_range = [min([ry(1); corners(:,2)]), max([ry(2); corners(:,2)])];
                else
                    x_range = [min(corners(:,1)), max(corners(:,1))];
                    y_range = [min(corners(:,2)), max(corners(:,2))];
                end

                Nxx = round(Nx/diff(rx)*diff(x_range));
                Nyy = round(Ny/diff(ry)*diff(y_range));
                x_r = linspace(x_range(1), x_range(2), Nxx);
                y_r = linspace(y_range(1), y_range(2), Nyy);
                [xx, yy] = ndgrid(x_r, y_r);
                xx = xx(:);
                yy = yy(:);
                if rotate < 0
                    xmean = mean(xx);
                    ymean = mean(yy);
                    T_xy = [xx-xmean, yy-ymean]*rot_mat + repmat([xmean, ymean], [Nxx*Nyy,1]);
                    pick = T_xy(:,1) <= rx(2) & T_xy(:,2) <= ry(2) & T_xy(:,1) >= rx(1) & T_xy(:,2) >= ry(1);
                    T_xy = T_xy(pick,:);
                    Nxy = size(T_xy,1);
                    facs = factor(Nxy);
                    nfac = length(facs);
                    Nxx = 1;
                    for i=1:(nfac-1)
                        Nxx = Nxx*facs(i);
                        if Nxx*facs(i+1) > Nxy/2
                            Nyy = Nxy/Nxx;
                            break;
                        end
                    end
                    % place holder for Nxx*Nyy entries in T
                    x_vec = linspace(rx(1), rx(2), Nxx);
                    y_vec = linspace(ry(1), ry(2), Nyy);
                else
                    T_xy = [xx, yy];
                    l1 = sum((corners(2,:) - corners(1,:)).^2);
                    l2 = sum((corners(3,:) - corners(1,:)).^2);
                    l1_vec = corners(2,:) - corners(1,:);
                    l2_vec = corners(3,:) - corners(1,:);
                    l0_vec = T_xy - repmat(corners(1,:),[Nxx*Nyy,1]);
                    ap = l0_vec*l1_vec';
                    aq = l0_vec*l2_vec';
                    pick = ap > 0 & ap < l1 & aq > 0 & aq < l2;
                    disp('T_xy')
                    rx_flat = rx;
                    ry_flat = ry;
                    rx = x_range;
                    ry = y_range;
                    x_range
                    [min(T_xy(:,1)), max(T_xy(:,1))]
                    ry
                    y_range
                    [min(T_xy(:,2)), max(T_xy(:,2))]
                    T_xy = T_xy(pick,:);
                    disp('T_xy(pick)')
                    [min(T_xy(:,1)), max(T_xy(:,1))]
                    [min(T_xy(:,2)), max(T_xy(:,2))]
                    Nxy = size(T_xy,1);
                    facs = factor(Nxy);
                    nfac = length(facs);
                    Nxx = 1;
                    for i=1:(nfac-1)
                        Nxx = Nxx*facs(i);
                        if Nxx*facs(i+1) > Nxy/2
                            Nyy = Nxy/Nxx;
                            break;
                        end
                    end
                    % place holder for Nxx*Nyy entries in T
                    x_vec = linspace(rx(1), rx(2), Nxx);
                    y_vec = linspace(ry(1), ry(2), Nyy);
                end
            end
        end
        range_x = rx(2)-rx(1);
        range_y = ry(2)-ry(1);
        T = ENtrset('grid',[0, 0, 0, 0, 0],...
            x_vec,...	% VFx
            y_vec,...	% VFy
            linspace(rOD(1),rOD(2),NOD),...	% OD
            tmp1,...				% ORtheta
            linspace(rORr(1),rORr(2),NORr),stream);%,... % ORr

        %sizeI = [range_x, range_y];
        %pd_ratio = 0.5;
        %[xy, ~] = pdisk2(sizeI, Nx*Ny, pd_ratio, stream);
        %counter = 1;
        %while size(xy,1) < Nx*Ny && counter < 100
        %    if size(xy,1)/(Nx*Ny) < pd_ratio
        %        pd_ratio = size(xy,1)/(Nx*Ny);
        %        counter = counter + 1;
        %    end
        %    [xy, ~] = pdisk2(sizeI, Nx*Ny, pd_ratio, stream);
        %    size(xy, 1)
        %end
        [x_, y_] = ndgrid(x_vec, y_vec);
        xy = reshape(cat(3, x_, y_), Nx*Ny, 2);
        xy(:,1) = xy(:,1) + (rand(Nx*Ny,1) - 0.5)*0.0*range_x/Nx;
        xy(:,2) = xy(:,2) + (rand(Nx*Ny,1) - 0.5)*0.0*range_y/Ny;
        T(:,1:2) = repmat(xy, NOR*NOD*NORr, 1);

        if rotate ~= 0
            for i=1:NOD*NORr*NOR
                start = (i-1)*Nxy + 1;
                stop = i*Nxy;
                T(start:stop,1:2) = T_xy;
            end
        end

        dx = mean(diff(x_vec));
        dy = mean(diff(y_vec));
        disp([num2str(prod(G)/(Nx*Ny)), 'neurons per visual field position']);
    else
        disp(['reading VF training points from ', VFpath])
        fID = fopen(VFpath, 'r');
        nxny = fread(fID, 1, 'int')
        data = fread(fID, [nxny,4], 'double');
        x_cortex = data(:,1); %  points on the mu grid
        y_cortex = data(:,2);
        xx_cortex = data(:,1); % mid points
        yy_cortex = data(:,2);
        x_deformed = data(:,3); % deformed T points
        y_deformed = data(:,4);
        mu_rx = fread(fID, 2, 'double')
        mu_ry = fread(fID, 2, 'double')
        df_mu_rx = fread(fID, 2, 'double')
        df_mu_ry = fread(fID, 2, 'double')
        dx = (rx(2)-rx(1))/nxny;
        dy = (ry(2)-ry(1))/nxny;
        vf = fread(fID, [nxny,2], 'double');
        vf_rx = fread(fID, 2, 'double')
        vf_ry = fread(fID, 2, 'double')
        transform = fread(fID, 2, 'double');
        pivot0 = fread(fID, 2, 'double')
        VFweights = fread(fID, nxny, 'double');
        midp = fread(fID, 1, 'int');
        fclose(fID);

        if heteroAlpha ~= 0
            if heteroAlpha == 1
                alpha_v = repmat(VFweights(:), NOD*NOR*NORr,1);
            else
                assert(heteroAlpha == -1);
                alpha_v = repmat(1./VFweights(:), NOD*NOR*NORr,1);
            end
            alpha = alpha_v/sum(alpha_v)*length(alpha_v) * alpha;
        end

        facs = factor(nxny)
        nfac = length(facs);
        assert(nfac > 1);
        fac1 = 1;
        for i=1:(nfac-1)
            fac1 = fac1*facs(i);
            if fac1*facs(i+1) > nxny/2
                fac2 = nxny/fac1;
                break;
            end
        end
        placeholder1 = linspace(1,fac1,fac1);
        placeholder2 = linspace(1,fac2,fac2);
        T = ENtrset('grid', zeros(1,5), ...
            placeholder1,...
            placeholder2,...
            linspace(rOD(1),rOD(2),NOD),...	% OD
            tmp1,...				        % ORtheta
            linspace(rORr(1),rORr(2),NORr),stream);% ORr
        x_vec0 = linspace(rx(1), rx(2), Nx);
        y_vec0 = linspace(ry(1), ry(2), Ny);
        %x_vec0 = linspace(rx(1), rx(2), fac1);
        %y_vec0 = linspace(ry(1), ry(2), fac2);
        xy = zeros(nxny,2);
        switch cortical_VF
            case 'VF'
                [xy(:,1), xy(:,2)] = pol2cart(vf(:,2),vf(:,1));
                xy(:,1) = rx(1) + (xy(:,1) - vf_rx(1))*(rx(2)-rx(1))/(vf_rx(2)-vf_rx(1));
                xy(:,2) = ry(1) + (xy(:,2) - vf_ry(1))*(ry(2)-ry(1))/(vf_ry(2)-vf_ry(1));
            case 'cortex'
                xy(:,1) = rx(1) + (x_deformed - df_mu_rx(1))*(rx(2)-rx(1))/(df_mu_rx(2)-df_mu_rx(1));
                xy(:,2) = ry(1) + (y_deformed - df_mu_ry(1))*(ry(2)-ry(1))/(df_mu_ry(2)-df_mu_ry(1));
        end

        for i=1:NOD*NORr*NOR
            start = (i-1)*nxny + 1;
            stop = i*nxny;
            T(start:stop,1:2) = xy;
        end
        disp('training range');
        [min(xy(:,1)), max(xy(:,1))]
        [min(xy(:,2)), max(xy(:,2))]
        VFweights_hlf = ones(nxny,1);
        disp([num2str(prod(G)/nxny), 'neurons per visual field position']);
    end
    dOD = 2*l;
    disp(['#',num2str(iRange),': dx = ',num2str(dx), ', dy = ', num2str(dy), ', dOD = ', num2str(dOD)]);
    % For non-rectangular cortex shapes, create a suitable Pi here:
    resol = 10;
    if cortical_shape
        manual_LR = ~non_cortical_LR && ~uniform_LR;
        [Pi, W, H, LR, VF, G, qx, Pi_x, Pi_y] = myCortex(stream, G(1), aspectRatio, rx, mu_rx, x_cortex, ry, mu_ry, y_cortex, VFweights_hlf, ecc, a, b, k, resol, nod, rOD*ODabsol, ODnoise, manual_LR, fign, [ENfilename0,'/',ENfilename], cortical_VF);
        fID = fopen([ENfilename0,'/',ENfilename,'-RefVF.bin'],'w');
        fwrite(fID, xx_cortex(:), 'double');
        fwrite(fID, yy_cortex(:), 'double');
        fwrite(fID, VFweights_hlf, 'double');
        fclose(fID);
    else
        Pi = zeros(G);
        Pi_x = x_vec;
        Pi_y = y_vec;
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
        for i=1:iters
            fID = fopen([ENfilename0,'/',ENfilename,'-LR_Pi',num2str(i),'.bin'],'w');
            fwrite(fID, a, 'double');
            fwrite(fID, b, 'double');
            fwrite(fID, k, 'double');
            fwrite(fID, ecc, 'double');
            fwrite(fID, int32(G), 'int');
            fwrite(fID, Pi, 'int');
            fwrite(fID, Pi_x, 'double');
            fwrite(fID, Pi_y, 'double');
            fclose(fID);
        end
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
    T_vec{1} = x_vec0;
    T_vec{2} = y_vec0;
    T_vec{3} = [-l, l];
    T_vec{6} = tmp0;
    T_vec{7} = [r];
    [T_vec{4}, T_vec{5}] = pol2cart(2*tmp0, r);
    %id = struct('VFx',1,'VFy',2,'OD',3,'ORx',4,'ORy',5,'OR',6,'ORr',7);
    %id = struct('VFx',1,'VFy',2,'OD',3);
    % Ranges of stimuli variables (for greyscales, etc.)
    % v = [1 rx;2 ry;3 rOD];
    v = [1 rx Nx+1;2 ry Ny+1;3 rOD NOD;4 -r r 2;5 -r r 2;6 -pi/2 pi/2 NOR+1;7 0 r NORr];
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
    %p = 2^(nG-1);
    p = 1;			% Stencil order (of derivative) ***
    s = stencil(nT);
    %s = {[0 -1 1],[0;-1;1]};	% Stencil list ***
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

    if cortical_shape
        %assert(mod(G(2)/2,2) == 0);
        xscale = (G(1)-1)/W;
        yscale = (G(2)-1)/(2*H);
        disp('dxdy');
        1/xscale
        1/yscale
        if isempty(VFpath)
            xx_cortex = [flipud(xx_cortex); xx_cortex];
            yy_cortex = [-flipud(yy_cortex); yy_cortex];
        end
        xgrid = 1 + (xx_cortex-qx)*xscale;
        disp('xgrid');
        [min(xgrid), max(xgrid)]
        ygrid = 1 + (yy_cortex+H)*yscale;
        disp('ygrid');
        [min(ygrid), max(ygrid)]
    else
        if rotate ~= 0
            xgrid = 1 + (T_xy(:,1)-rx(1))/(rx(2)-rx(1))*(G(1)-1);
            ygrid = 1 + (T_xy(:,2)-ry(1))/(ry(2)-ry(1))*(G(2)-1);
        else
            [xgrid, ygrid] = meshgrid(x_vec, y_vec);
            xgrid = 1 + xgrid/(rx(2)-rx(1))*(G(1)-1);
            ygrid = 1 + ygrid/(ry(2)-ry(1))*(G(2)-1);
        end
    end

    % Initial elastic net: retinotopic with some noise and random, uniform
    % OD and OR.

    if cortical_shape
        switch cortical_VF
            case 'VF'
                if old
                    mu = ENtrset('grid',zeros(1,2),...		% Small noise
                        linspace(rx(1),rx(2),G(1)),...	% VFx
                        linspace(ry(2),ry(2),G(2)),stream);	% VFy
                else
                    mu = reshape(VF, M, 2);
                end
            case 'cortex'
                if ~isempty(VFpath)
                    %figure;
                    if ~isempty(transform)
                        slope0 = transform(1)
                        shrink = transform(2)
                    end
                    if shrink < 1.0
                        W
                        H
                        dx = W/(G(1)-1)
                        dy = 2*H/(G(2)-1)
                        slope = slope0 * 1;
                        mu_xy = zeros(M,2);
                        % output of meshgrid is the transpose of the output of ndgrid
                        if mod(G(2),2) == 0
                            [cart_x, cart_y] = ndgrid(linspace(0,W,G(1)), linspace(dy/2,H,G(2)/2));
                        else

                            [cart_x, cart_y] = ndgrid(linspace(0,W,G(1)), linspace(0,H,(G(2)+1)/2));
                        end
                        cart = cat(3,cart_x, cart_y);
                        disp('tx ty')
                        tx = [min(min(cart(:,:,1))), max(max(cart(:,:,1)))]
                        ty = [min(min(cart(:,:,2))), max(max(cart(:,:,2)))]

                        pivot = [pivot0(1) - mu_rx(1), 0]
                        b = pivot(2) - slope*pivot(1);
                        pivot = reshape([repmat(pivot(1), size(cart_x)), repmat(pivot(2), size(cart_y))], size(cart));
                        dxx = (cart_y - slope*cart_x - b)*slope/(1+slope*slope);
                        dyy = -dxx/slope;
                        cart = (1-shrink)*reshape([dxx, dyy], size(cart)) + cart;
                        %plot(cart(:,:,1), cart(:,:,2), '-*r', 'MarkerSize', 0.2)
                        rad = atan(slope) - atan(slope*shrink);
                        rot_mat = [cos(rad), sin(rad); -sin(rad), cos(rad)]';
                        transformed_cart = reshape(reshape(cart - pivot, [prod(size(cart_x)),2])*rot_mat,size(cart))+pivot;
                        plot(transformed_cart(:,:,1), transformed_cart(:,:,2), '-sg', 'MarkerSize', 0.2)
                        if mod(G(2),2) == 0
                            transformed_cart_reflect = reshape([transformed_cart(:,:,1), - transformed_cart(:,:,2)], size(cart));,
                        else
                            transformed_cart_reflect = reshape([transformed_cart(:,2:end,1), - transformed_cart(:,2:end,2)], [size(cart,1),size(cart,2)-1,2]);,
                        end
                        transformed_cart = cat(2,flip(transformed_cart_reflect,2), transformed_cart);
                        %plot(transformed_cart(:,:,1)+100, transformed_cart(:,:,2), '-ok', 'MarkerSize', 0.2)
                        disp('mu_xy before rxry');
                        tx0 = min(min(transformed_cart(:,:,1)))
                        tx1 = max(max(transformed_cart(:,:,1)))
                        df_mu_rx
                        ty0 = min(min(transformed_cart(:,:,2)))
                        ty1 = max(max(transformed_cart(:,:,2)))
                        df_mu_ry
                        mu_xy(:,1) = reshape(rx(1) + (transformed_cart(:,:,1)-df_mu_rx(1))*(rx(2)-rx(1))/diff(df_mu_rx), [M,1]);
                        mu_xy(:,2) = reshape(ry(1) + (transformed_cart(:,:,2)-df_mu_ry(1))*(ry(2)-ry(1))/diff(df_mu_ry), [M,1]);
                        mu = mu_xy;
                        disp('mu_xy');
                        rx
                        [min(mu(:,1)), max(mu(:,1))]
                        ry
                        [min(mu(:,2)), max(mu(:,2))]
                        %subplot(1,2,2)
                        %hold on
                        %plot(mu_xy(:,1), mu_xy(:,2), '-ok', 'MarkerSize', 0.2);

                        %print(gcf,'-loose','-r600', '-dpng',[ENfilename0,'/',ENfilename,'-transform.png']);
                        %
                    else
                        tmp1 = linspace(rx(1),rx(2),G(1));
                        tmp2 = linspace(ry(1),ry(2),G(2));
                        mu = ENtrset('grid',zeros(1,2),...		% Small noise
                            tmp1,...	% VFx
                            tmp2,stream);	% VFy
                    end
                else
                    tmp1 = linspace(rx(1),rx(2),G(1));
                    tmp2 = linspace(ry(1),ry(2),G(2));
                    mu = ENtrset('grid',zeros(1,2),...		% Small noise
                        tmp1,...	% VFx
                        tmp2,stream);	% VFy
                end

                %dtmp = (tmp1(2)-tmp1(1))/2;
                %tmp1 = linspace(rx(1)-dtmp, rx(2) + dtmp, G(1));
                %w = dipole_ext(ecc,pi/2,a,b,k) - k*log(a/b);
                %xmax = real(w);
                %ymax = imag(w);

                %ylength = 2*ymax/xmax*G(2);
                %yratio_hlf = 2*ymax/ylength;
                %hy = (ry(2)-ry(1))/2/yratio_hlf;

                %tmp2 = linspace(0, 2*hy, G(2));

                %[min(tmp1), max(tmp1)]
                %[min(tmp1), max(tmp2)]
                %disp('mu points range');
                %[min(mu(:,1)), max(mu(:,1))]
                %rx
                %[min(mu(:,2)), max(mu(:,2))]
                %ry
        end
    else
        if rotate > 0
            [xx, yy] = ndgrid(linspace(rx_flat(1),rx_flat(2), G(1)), linspace(ry_flat(1),ry_flat(2), G(2)));
            xx = xx(:);
            yy = yy(:);
            rot_mat = [cos(rotate), sin(rotate); -sin(rotate), cos(rotate)]';
            xmean = mean(xx)
            ymean = mean(yy)
            mu_xy = [xx-xmean, yy-ymean]*rot_mat + repmat([xmean, ymean], [G(1)*G(2),1]);
            mu = mu_xy;
            disp('mu_xy')
            x_range
            [min(mu_xy(:,1)), max(mu_xy(:,1))]
            [min(T_xy(:,1)), max(T_xy(:,1))]
            y_range
            [min(mu_xy(:,2)), max(mu_xy(:,2))]
            [min(T_xy(:,2)), max(T_xy(:,2))]
        else
            init_ratio = 0;
            range_x = rx(2)-rx(1);
            range_y = ry(2)-ry(1);
            noise_mu = 1.0;
            mu = ENtrset('grid',[range_x/Nx*noise_mu, range_y/Ny*noise_mu],...		% Small noise
                linspace(rx(1)+range_x*init_ratio,rx(2)-range_x*init_ratio,G(1)),...	% VFx
                linspace(ry(1)+range_y*init_ratio,ry(2)-range_y*init_ratio,G(2)),stream);	% VFy
%             tmp = rand(M,2);
%             tmp(:,1) = rx(1) + tmp(:,1) * (rx(2) - rx(1));
%             tmp(:,2) = ry(1) + tmp(:,2) * (ry(2) - ry(1));
%             mu = tmp;
        end
    end
    %    mu = reshape(gridVF, M, 2);
    if uniform_LR
        LR(:) = mean(rOD);
        mu = [mu reshape(LR,M,1)];		% OD
        %mu = [mu, ENtrset('uniform',rOD,M,stream)];
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
    %OR = zeros(size(OR));
    OR = ENtrset('uniform',[-pi/2 pi/2;0 r], M,stream);
    OR(:,2) = r/2;
    mu = [mu OR];
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
                myV1replay(G,aspectRatio,bc,ENlist,v,1,T,T_vec,Pi,murange,id);
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
                'Nx rx dx Ny ry dy NOD rOD dOD l NOR rORt rORr r seed v xgrid ygrid ' ...
                'N D L M G bc p s T T_vec Pi S DD knot A LL mu Kin Kend iters Ksched '...
                'alpha beta annrate max_it max_cyc min_K tol method '...
                'W normcte betanorm']);
            if strcmp(ENproc,'saveplot')
                ENlist = struct('mu',mu,'stats',struct(...
                    'K',NaN,'E',[NaN NaN NaN],'time',[NaN NaN NaN],...
                    'cpu','','code',1,'it',0));
                myV1replay(G,aspectRatio,bc,ENlist,v,1,T,T_vec,Pi,murange,id);
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
        [mu,stats] = ENtr_ann(exchange_nm,Tr,S,Pi,mu,Ksched(ENcounter),alpha,betanorm,ibeta,...
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
                    myV1replay(G,aspectRatio,bc,ENlist,v,ENcounter+1,T,T_vec,Pi,murange,id);
                end
            case {'save','saveplot'}
                save(sprintf('%s-%s%04d.mat',ENfilename0,ENfilename,ENcounter),'mu','stats','murange');
                if strcmp(ENproc,'saveplot')
                    myV1replay(G,aspectRatio,bc,struct('mu',mu,'stats',stats),v,1,T,T_vec,Pi,murange,id);
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
                'ENlist murange id xgrid ygrid ' ...
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
                'ENlist murange id xgrid ygrid ' ...
                'N D L M G bc p s T T_vec Pi S DD knot A LL Kin Kend iters Ksched '...
                'alpha beta annrate max_it max_cyc min_K tol method '...
                'W normcte betanorm']);
            delete([ENfilename0, '-', ENfilename, '????.mat']);
        otherwise
            % Do nothing.
    end

    % Plot some statistics for the objective function value and computation time
    switch ENproc
        case {'varplot','saveplot'}
            myV1replay(G,aspectRatio,bc,ENlist,v,1,T,T_vec,Pi,murange,id,[],[],[100, 102]);
        otherwise
            % Do nothing.
    end
    disp(['ratio of L/R = ', num2str(sum(mu(logical(Pi),id.OD) < 0)/sum(mu(logical(Pi),id.OD) > 0))]);
else
    disp('data exist');
    load([ENfilename0,'-',ENfilename,'.mat']);
end
if ~exist('figlist', 'var')
    if plots
        %figlist = [1,2,3,4,5,7,15,16,20,21,100,102,34, 40, 41, 50, 54, 60];
        figlist = [1,2,4,5,6,15,16,34,50,60,100,102];
        %figlist = [1];
        % less figures
        %figlist = [1,4,100,102,34,60];
    else
        figlist = [];
    end
end
statsOnly = true;
if l == 0
    figlist(figlist == 1) = [];
    figlist(figlist == 2) = [];
    figlist(figlist == 13) = [];
    figlist(figlist == 14) = [];
    figlist(figlist == 15) = [];
    figlist(figlist == 34) = [];
    figlist(figlist == 40) = [];
    figlist(figlist == 50) = [];
    figlist(figlist == 54) = [];
    figlist(figlist == 60) = [];
    statsOnly = false;
end
if r == 0
    figlist(figlist == 2) = [];
    figlist(figlist == 3) = [];
    figlist(figlist == 4) = [];
    figlist(figlist == 6) = [];
    figlist(figlist == 7) = [];
    figlist(figlist == 16) = [];
    figlist(figlist == 34) = [];
    figlist(figlist == 41) = [];
    figlist(figlist == 50) = [];
    figlist(figlist == 54) = [];
    figlist(figlist == 60) = [];
    statsOnly = false;
end
right_open = cortical_shape;
stats = myV1stats(stream,G,aspectRatio,bc,ENlist,v,plotting,T,T_vec,Pi,murange,id,[],[ENfilename0,'/',ENfilename,'.png'],figlist,statsOnly,right_open,separateData,xgrid,ygrid,iRange,ENfilename);
if saveLR
    for i = 1:iters
        fID = fopen([ENfilename0,'/',ENfilename,'-LR_Pi',num2str(i),'.bin'],'a');
        fwrite(fID, ENlist(i).mu(:,id.OD), 'double');
        fclose(fID);
    end
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
function s = stencil(nT)
s = {};
if nT == 4
    q = [0, 0, 0, 0, -25, 48, -36, 16, -3]/12;
    s = [s, q, q'];
end
if nT == 3
    q = [0, 0, -3, -10, 18, -6, 1]/12;
    s = [s, q, q'];
end
if nT == 2
    q = [0, 0, -3, 4, -1]/2;
    s = [s, q, q'];
end
if nT == 1
    q = [0 -1 1];
    s = [s, q, q'];
end
end
