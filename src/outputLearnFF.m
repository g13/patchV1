% connection strength heatmaps % better to choose from testLearnFF 
function outputLearnFF(isuffix0, isuffix, osuffix, fdr, LGN_switch, mix)
	if nargin < 5
		LGN_switch = true;
	end
	if nargin < 6
		mix = true;
	end
	fdr = [fdr, '/'];
	st = 1; %0 for temporal, 1 for spatial, 2 for both

	%V1_pick = [1,10,100,999,1000];
	ns = 10;

	rng(1390843)
	if ~isempty(isuffix0)
	    isuffix0 = ['_', isuffix0];
	end
	if ~isempty(isuffix)
	    isuffix = ['_', isuffix];
	end
	if ~isempty(osuffix)
	    osuffix = ['_', osuffix];
	end
	thres = 0.0;
    orient_thres = 0.8;
	nstep = 1000;
	step0 = 1;
	nt_ = 0;
	nbins = 20;
    nit0 = 20;

	f_sLGN = ['sLGN', osuffix, '.bin']
	learnDataFn = ['learnData_FF', osuffix, '.bin']

	fLGN_vpos = ['LGN_vpos', isuffix0, '.bin'];
	LGN_V1_id_fn = ['LGN_V1_idList', isuffix, '.bin']
	fLGN_switch = ['LGN_switch', isuffix, '.bin'];

	fid = fopen(learnDataFn, 'r');
	dt = fread(fid, 1, 'float')
	fclose(fid);
	
	fid = fopen(fLGN_vpos, 'r');
	nLGN = fread(fid, 1, 'uint') % # ipsi-lateral LGN 
	nLGN_I = fread(fid, 1, 'uint') % # ipsi-lateral LGN 
	fseek(fid, 5*4, 0); % 5 float constants
    LGN_vpos = fread(fid,[(nLGN+nLGN_I), 2], 'float');
    LGN_type = fread(fid, nLGN, 'uint');
    types = unique(LGN_type);
    ntype = length(types);
	fseek(fid, (nLGN+nLGN_I)*2*4, 0); % skip LGN_vpos in polar
	doubleOnOff = fread(fid, 1, 'int')
	fclose(fid);
	
	% read the constants first only 
	fid = fopen(f_sLGN, 'r');
	nt = fread(fid, 1, 'uint');
	nV1 = fread(fid, 1, 'uint');
	max_LGNperV1 = fread(fid, 1, 'uint')
	sRatio = fread(fid, 1, 'float')
	nLearnFF = fread(fid, 1, 'uint')
	gmaxLGN = fread(fid, nLearnFF, 'float')
	gmax0 = gmaxLGN(1)*sRatio; % TODO, separate E and I
	fclose(fid);

	if nt_ > nt || nt_ == 0
		nt_ = nt;
	end
	if step0 > nt_ || step0 == 0
		step0 = 1;
	end
	range_nt = nt_-step0 +1;
	if range_nt == nt
		rtime = '';
	else
		rtime = ['-t', num2str(step0/nt*100,'%.0f'),'_',num2str(nt_/nt*100,'%.0f'),'%'];
	end
	if nstep > range_nt || nstep == 0
	    nstep = range_nt;
	end
	step0
	nt_
	nstep
	% read connection id
	sid = fopen(LGN_V1_id_fn, 'r');
	LGN_V1_ID = zeros(max_LGNperV1, nV1);
	nLGN_V1 = zeros(nV1,1);
	fread(sid, 1, 'uint'); % nV1
	for i = 1:nV1
	    nLGN_V1(i) = fread(sid, 1, 'uint');
	    assert(nLGN_V1(i) <= max_LGNperV1);
	    if nLGN_V1(i) > 0
	        LGN_V1_ID(1:nLGN_V1(i),i) = fread(sid, nLGN_V1(i), 'uint')  + 1;
	    end
	end
	fclose(sid);

	if ~exist('V1_pick', 'var') 
		V1_pick = randi(768,[ns,1]);
	else
		ns = length(V1_pick);
	end
	disp(V1_pick); 

    orient = zeros(nV1,1);
	fid = fopen(f_sLGN, 'r');
	fseek(fid, 6*4, 0); % skip till time
	fseek(fid, max_LGNperV1*nV1*int64(nt_-1)*4, 0); % skip till time
	sLGN = fread(fid, [max_LGNperV1, nV1], 'float');
    fclose(fid);

    for iV1 = 1:nV1
        all_id = LGN_V1_ID(1:nLGN_V1(iV1),iV1);
        all_type = LGN_type(all_id);
        all_s = sLGN(1:nLGN_V1(iV1),iV1);

        on_s = all_s(all_type == 4);
        on_id = all_id(all_type == 4);
        onPick = on_id(on_s >= max(on_s) * orient_thres);
        on_pos = mean(LGN_vpos(onPick,:), 1);

        off_s = all_s(all_type == 5);
        off_id = all_id(all_type == 5);
        offPick = off_id(off_s >= max(off_s) * orient_thres);
        off_pos = mean(LGN_vpos(offPick,:), 1);

        orient(iV1) = atan2(off_pos(2)-on_pos(2), on_pos(1)-off_pos(1)); % spin around as the imagesc
        if orient(iV1) < 0
            orient(iV1) = orient(iV1) + 2*pi;
        end
    end
	f = figure('PaperPosition',[0, 0, 4, 4]);
    histogram(orient*180/pi, 'BinEdges', linspace(0,360,12));
	set(f, 'OuterPosition', [.1, .1, 8, 6]);
	set(f, 'innerPosition', [.1, .1, 8, 6]);
	%saveas(f, [fdr, 'tLGN_V1-',num2str(iV1), osuffix,rtime], 'fig');
	saveas(f, [fdr, 'ori-LGN_V1', osuffix,rtime, '.png']);

	if LGN_switch
		fid = fopen(fLGN_switch, 'r');
		nStatus = fread(fid, 1, 'uint')
		status = fread(fid, nStatus*6, 'float'); % activated percentage
		statusDur = fread(fid, nStatus, 'float'); % duration
		reverse = fread(fid, nStatus, 'int');
		fclose(fid);
		reverse'
		statusDur'
		typeInput = [sum(statusDur.*(1-reverse)), sum(statusDur.*reverse)]/sum(statusDur);
	    nit = nStatus + 1
	else
	    nit = nit0
		typeInput = ones(ntype, 1);
	end
	if nit <= 1
		nit = 2
	end

    nrow = double(idivide(int32(nit+nit0-1),int32(nit0)));
	qt = int32(floor(linspace(step0, nt_, nit)))
	qtt = int32(floor(linspace(1,nstep,nit)));
	for iq = 1:ns
	    iV1 = V1_pick(iq)
	    %disp(nLGN_V1(iV1));
	    %disp(LGN_V1_ID(1:nLGN_V1(iV1),iV1)');
	    
	    if st == 2 || st == 1
	        sLGN = zeros(nLGN, nit);
	        fid = fopen(f_sLGN, 'r');
	        fseek(fid, 6*4, 0); % skip till time
	    
	        % skip times
	        %ht = round(nt/2);
	        %it = [0, ht-1, nt-1 - (ht+1)]
			
	        fseek(fid, max_LGNperV1*nV1*int64(step0-1)*4, 0); % skip till time
	        data = fread(fid, [max_LGNperV1, nV1], 'float');
	        sLGN(LGN_V1_ID(1:nLGN_V1(iV1),iV1),1) = data(1:nLGN_V1(iV1),iV1);

	        it = diff(qt)-1;
	        for j = 1:nit-1
	            if it(j) > 0
	                fseek(fid, max_LGNperV1*nV1*int64(it(j))*4, 0); % skip till time
	            end
	            data = fread(fid, [max_LGNperV1, nV1], 'float');
	            sLGN(LGN_V1_ID(1:nLGN_V1(iV1),iV1),j+1) = data(1:nLGN_V1(iV1),iV1);
	        end
	        fclose(fid);
	        
			if doubleOnOff == 0
	        	f = figure('PaperPosition',[0, 0, nit, (2-mix)]);
				set(f, 'PaperUnit', 'inches');
				nLGN_1D = sqrt(double(nLGN))
	    	    sLGN = reshape(sLGN, [nLGN_1D, nLGN_1D, nit]);
				gmax = max(abs(sLGN(:)));
	    	    if mix
	    	        clims = [-1, 1];
	    	        for i = 1:nit
	    	            subplot(1,nit+1,i)
	    	            stmp = sLGN(:,:,i);
	    	            offPick = LGN_type == 5;
	    	            stmp(offPick) = -stmp(offPick);

						local_max = max(abs(stmp(:)));
	    	            stmp = stmp./gmax;
	    	            stmp(abs(stmp)<local_max/gmax*thres) = 0;

	    	            imagesc(stmp', clims);
	    	            colormap('gray');
	    	            daspect([1,1,1]);
	    	            set(gca,'YDir','reverse')
	    	            title(['t', num2str(double(qt(i))/nt*100,'%.0f'),'%-n',num2str(sum(sum(stmp>0))),'-p',num2str(gmax/gmax0*100,'%.0f'),'%'], 'FontSize', 6);
	    	            if i == nit
							ax = subplot(1, nit+1, nit+1);
	    	            	im = imagesc(stmp, clims);
							im.Visible = 0;
							ax.Visible = 0;
	    	                colorbar;
	    	            	colormap('gray');
	    	            end
	    	        end
	    	    else
	    	        clims = [0, 1];
	    	        for itype = 1:ntype
	    	            for i = 1:nit
	    	                subplot(ntype,nit+1,(itype-1)*(nit+1) + i)
	    	                stmp = sLGN(:,:,i);
							if i == 1
								stmp
							end
	    	                stmp(LGN_type ~= types(itype)) = 0;

							local_max = max(abs(stmp(:)));
	    	            	stmp(stmp<local_max*thres) = 0;
	    	            	stmp = stmp./gmax;

	    	                imagesc(stmp', clims);
	    	            	daspect([1,1,1]);
	    	                set(gca,'YDir','reverse')
	    	                if itype == 1
								title(['t', num2str(double(qt(i))/nt*100,'%.0f'),'%-n',num2str(sum(sum(stmp>0))),'-p',num2str(gmax/gmax0*100,'%.0f'),'%'], 'FontSize', 6);
	    	                end
	    	                if i == 1
	    	                    ylabel(['type: ', num2str(types(itype))]);
	    	                end
	    	                if i == nit
								ax = subplot(1, nit+1, nit+1);
	    	            		im = imagesc(stmp, clims);
								im.Visible = 0;
								ax.Visible = 0;
	    	            		colormap('gray');
	    	                    colorbar;
	    	                end
	    	            end
	    	        end
	    	    end
			else
	        	f = figure('PaperPosition',[0, 0, nit0, ntype*nrow]);
				set(f, 'PaperUnit', 'inches');
				assert(doubleOnOff == 1);
				nLGN_1D = sqrt(double(nLGN/2))
				sLGN = reshape(sLGN, [nLGN_1D*2, nLGN_1D, nit]);
				gmax = max(abs(sLGN(:)));
	    	    clims = [0, 1];
	    	    
	    	    for itype = 1:ntype
                    row = 1;
	    	        for i = 1:nit
                        iplot = (row-1)*ntype*(nit0+1) + (itype-1)*(nit0+1);
                        if i > nit0
                            iplot = iplot + mod(i-1, nit0)+1;
                        else
                            iplot = iplot + i;
                        end
	    	            subplot(nrow*ntype,nit0+1,iplot)
	    	            stmp = sLGN(itype:2:(nLGN_1D*2),:,i);
						local_max = max(abs(stmp(:)));
	    	            stmp(stmp<local_max*thres) = 0;
	    	            stmp = stmp./gmax;
	    	            imagesc([1 nLGN_1D], [1,nLGN_1D],stmp', clims);
	    	        	daspect([1,1,1]);
	    	            set(gca,'YDir','reverse')
	    	            if itype == 1
							title(['t', num2str(double(qt(i))/nt*100,'%.0f'),'%-n',num2str(sum(sum(stmp>0))),'-p',num2str(gmax/gmax0*100,'%.0f'),'%'], 'FontSize', 6);
                        else
                            if i == nit
                                title([num2str(orient(iV1)*180/pi, '%.0f'), 'deg']);
                            end
	    	            end
	    	            if mod(i, nit0) == 1
	    	                ylabel(['type: ', num2str(types(itype))]);
	    	            end
                        if mod(i, nit0) == 0
                            row = row + 1;
                        end
	    	            if i == nit
							ax = subplot(1, nit0+1, nit0+1);
	    	        		im = imagesc(stmp, clims);
							im.Visible = 0;
							ax.Visible = 0;
	    	        		colormap('gray');
	    	                colorbar;
	    	            end
	    	        end
	    	    end
			end
			set(f, 'OuterPosition', [.1, .1, nit+2, 4]);
			set(f, 'innerPosition', [.1, .1, nit+2, 4]);
			if mix && doubleOnOff ~= 1
	        	%saveas(f, [fdr,'sLGN_V1-',num2str(iV1), osuffix, '-mix',rtime], 'fig');
	        	saveas(f, [fdr,'sLGN_V1-',num2str(iV1), osuffix, '-mix',rtime,'.png']);
			else
	        	%saveas(f, [fdr,'sLGN_V1-',num2str(iV1), osuffix, '-sep',rtime], 'fig');
	        	saveas(f, [fdr,'sLGN_V1-',num2str(iV1), osuffix, '-sep',rtime,'.png']);
			end
	    end
	    if st == 2 || st == 0
	        
	        tstep = int64(round(range_nt/nstep))
	        it = step0:tstep:nt_;
	        nstep = length(it)
	        tLGN = zeros(max_LGNperV1, nstep);
	        

	        fid = fopen(f_sLGN, 'r');
	        fseek(fid, 6*4, 0); % skip till time
	        fseek(fid, max_LGNperV1*nV1*int64(step0-1)*4, 0); % skip till time
	        data = fread(fid, [max_LGNperV1, nV1], 'float');
	        tLGN(:,1) = data(:,iV1);
	        
	        for j = 2:nstep
	            fseek(fid, max_LGNperV1*nV1*int64(tstep-1)*4, 0); % skip till time
	            data = fread(fid, [max_LGNperV1, nV1], 'float');
	            tLGN(:,j) = data(:,iV1);
	        end
	        fclose(fid);
	       	gmax = max(tLGN(:));
	        f = figure('PaperPosition',[.1 .1 8 6]);
			
			if LGN_switch
				subplot(21,3,3*10 + [1,2])
				hold on
				status_t = 0;
				for i = 1:nStatus
					current_nt = round(statusDur(i)*1000/dt);
					current_t = (1:current_nt)*dt;
					plot(status_t + current_t, zeros(current_nt,1) + reverse(i), 'k');
					status_t = status_t + statusDur*1000;
				end
				set(gca,'visible','off','XColor','none','YColor','none','xtick',[],'ytick',[]);
			end
				
			for i = 1:ntype
	    		subplot(ntype,3, 3*(i-1) + [1,2])
	        	plot(it*dt, tLGN(LGN_type(LGN_V1_ID(1:nLGN_V1(iV1),iV1)) == types(i),:)./gmax*100, '-');
				title(['type', num2str(types(i)), ' input takes ' num2str(typeInput(i)*100, '%.1f'), ' %']);
	        	ylabel('strength % of max');
				if i == ntype
	        		xlabel('ms');
				end
			end
	        edges = linspace(0,100,nbins);
	        for i = 1:nit
	            subplot(nit,3,3*i)
				hold on
				for j = 1:ntype
	            	histogram(tLGN(LGN_type(LGN_V1_ID(1:nLGN_V1(iV1),iV1)) == types(j), qtt(i))./gmax*100, 'BinEdges', edges, 'FaceAlpha', 0.5);
				end
	        end
			set(f, 'OuterPosition', [.1, .1, 8, 6]);
			set(f, 'innerPosition', [.1, .1, 8, 6]);
	        %saveas(f, [fdr, 'tLGN_V1-',num2str(iV1), osuffix,rtime], 'fig');
	        saveas(f, [fdr, 'tLGN_V1-',num2str(iV1), osuffix,rtime, '.png']);
	    end
	end
end
