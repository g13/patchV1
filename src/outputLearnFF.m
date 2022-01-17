% connection strength heatmaps % better to choose from testLearnFF 
function outputLearnFF(isuffix0, isuffix, osuffix, res_fdr, data_fdr, fig_fdr, LGN_switch, mix, st, examSingle, use_local_max)
	if nargin < 10
		disp('no default argument available');
		return
	else
		if nargin < 11
			use_local_max = 1;
		end
	end

	fig_fdr = [fig_fdr, '/'];
	res_fdr = [res_fdr, '/'];
	data_fdr = [data_fdr, '/'];

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
	%%%% HERE %%%%
	thres_out = 0.5; % under which ratio of max LGN connection strength will not be used to calculate the orientation of RF, neither will be counted toward the number of connection in spatial figure's title.
	nstep = 1000; % total steps to sample from the trace of weight evolution.
	step0 = 1; % starting time step
	nt_ = 0; % ending time step
	nbins = 20; % bins for histogram
    nit0 = 20; % number of snapshot for the spatial figure and histogram in the temporal figure
	ns = 10; % number for V1 neurons to be sampled.
	%V1_pick = [203,752,365,360,715,467,743]; % specify the IDs of V1 neurons to be sampled. If set, ns will be ignored.
	%%%%%%%%%%%%  

	f_sLGN = [data_fdr, 'sLGN', osuffix, '.bin']
	learnDataFn = [data_fdr, 'learnData_FF', osuffix, '.bin']
	V1_frFn = [data_fdr, 'max_fr', osuffix, '.bin']

	fLGN_vpos = [res_fdr, 'LGN_vpos', isuffix0, '.bin'];
	LGN_V1_id_fn = [res_fdr, 'LGN_V1_idList', isuffix, '.bin']
	fLGN_switch = [res_fdr,'LGN_switch', isuffix, '.bin'];

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
	gmaxLGN = fread(fid, nLearnFF, 'float')*sRatio
	fclose(fid);

	fid = fopen(V1_frFn, 'r');
	if fid == -1
		fr = zeros(nV1,1);
	else
		fr = fread(fid, nV1, 'double');
		fclose(fid);
	end
	

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
		V1_pick = randi(nV1,[ns,1]);
		%V1_pick = randi(nV1,[ns,1]);
	else
		ns = length(V1_pick);
	end
	disp(V1_pick); 

	%%%%%%%%%%%%%%%%%% HERE %%%%%%%%%%%%%%%%%%
    orient = zeros(nV1,1);
	fid = fopen(f_sLGN, 'r');
	fseek(fid, 6*4, 0); % skip till time
	fseek(fid, max_LGNperV1*nV1*int64(nt_-1)*4, 0); % skip till time
	sLGN = fread(fid, [nV1, max_LGNperV1], 'float')'; % transposed
    fclose(fid);

	nLGN_1D = sqrt(double(nLGN/2))
	onS = zeros(nV1,1);
	offS = zeros(nV1,1);
    for iV1 = 1:nV1
        all_id = LGN_V1_ID(1:nLGN_V1(iV1),iV1);
        all_type = LGN_type(all_id);
        all_s = sLGN(1:nLGN_V1(iV1),iV1);

        on_s = all_s(all_type == 4);
        on_id = all_id(all_type == 4);
		sPick = on_s >= max(on_s) * thres_out;
        onPick = on_id(sPick);
		onS(iV1) = sum(on_s(sPick));
        on_pos = mean(LGN_vpos(onPick,:), 1);

        off_s = all_s(all_type == 5);
        off_id = all_id(all_type == 5);
		sPick = off_s >= max(off_s) * thres_out;
        offPick = off_id(sPick);
		offS(iV1) = sum(off_s(sPick));
        off_pos = mean(LGN_vpos(offPick,:), 1);

        orient(iV1) = atan2(off_pos(2)-on_pos(2), on_pos(1)-off_pos(1)); % spin around as the imagesc
        if orient(iV1) < 0
            orient(iV1) = orient(iV1) + 2*pi;
        end
    end

	f = figure('PaperPosition',[0, 0, 8, 16]);
	subplot(2,1,1)
    histogram(onS-offS, 20);
	xlabel('sOn-sOff')
	subplot(2,1,2)
    histogram(orient*180/pi, 'BinEdges', linspace(0,360,12));
	xlabel('OP (deg)')

	set(f, 'OuterPosition', [.1, .1, 8, 12]);
	set(f, 'innerPosition', [.1, .1, 8, 12]);
	saveas(f, [fig_fdr, 'stats-LGN_V1', osuffix,rtime, '.png']);
	%%%%%%%%%%%%%%%%%% HERE %%%%%%%%%%%%%%%%%%

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
	    %disp(nLGN_V1(iV1));
	    %disp(LGN_V1_ID(1:nLGN_V1(iV1),iV1)');
	    
	if st == 2 || st == 1
	    sLGN_all = zeros(nLGN, nit, ns);
	    fid = fopen(f_sLGN, 'r');
	    fseek(fid, 6*4, 0); % skip till time
	    
	    % skip times
	    %ht = round(nt/2);
	    %it = [0, ht-1, nt-1 - (ht+1)]
		
	    fseek(fid, max_LGNperV1*nV1*int64(step0-1)*4, 0); % skip till time
	    data = fread(fid, [nV1, max_LGNperV1], 'float')'; % transposed

	    for iq = 1:ns
	        iV1 = V1_pick(iq);
	        sLGN_all(LGN_V1_ID(1:nLGN_V1(iV1),iV1),1,iq) = data(1:nLGN_V1(iV1),iV1);
        end

	    it = diff(qt)-1;
	    for j = 1:nit-1
	        if it(j) > 0
	            fseek(fid, max_LGNperV1*nV1*int64(it(j))*4, 0); % skip till time
	        end
			data = fread(fid, [nV1, max_LGNperV1], 'float')'; % transposed
	        for iq = 1:ns
	            iV1 = V1_pick(iq);
	            sLGN_all(LGN_V1_ID(1:nLGN_V1(iV1),iV1),j+1,iq) = data(1:nLGN_V1(iV1),iV1);
            end
	    end
	    fclose(fid);
	    for iq = 1:ns
	        iV1 = V1_pick(iq);
            sLGN = sLGN_all(:,:,iq);
			gmax = max(abs(sLGN(:)));
            if gmax == 0
                continue;
            end
			if doubleOnOff == 0
	        	f = figure('PaperPosition',[0, 0, nit, (2-mix)]);
				set(f, 'PaperUnit', 'inches');
	    	    sLGN = reshape(sLGN, [nLGN_1D, nLGN_1D, nit]);
	    	    if mix
	    	        clims = [-1, 1];
	    	        for i = 1:nit
	    	            subplot(1,nit+1,i)
	    	            stmp = sLGN(:,:,i);
	    	            offPick = LGN_type == 5;
	    	            stmp(offPick) = -stmp(offPick);

						local_max = max(abs(stmp(:)));
	    	            stmp = stmp./gmax;
	    	            stmp(abs(stmp)<local_max/gmax*thres_out) = 0;

	    	            imagesc(stmp', clims);
	    	            colormap('gray');
	    	            daspect([1,1,1]);
	    	            set(gca,'YDir','reverse')
	    	            title(['t', num2str(double(qt(i))/nt*100,'%.0f'),'%-n',num2str(sum(sum(stmp>0))),'-p',num2str(gmax/gmaxLGN*100,'%.0f'),'%'], 'FontSize', 6);
	    	            if i == nit
							ax = subplot(1, nit+1, nit+1);
	    	            	im = imagesc(stmp, clims);
							im.Visible = 0;
							ax.Visible = 0;
	    	                colorbar;
	    	            	colormap('gray');
                            if i == nit
                                title([num2str(orient(iV1)*180/pi, '%.0f'), 'deg, ', num2str(fr(iV1),'%.2f'), 'Hz']);
                            end
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
	    	            	stmp = stmp./gmax;

	    	                imagesc(stmp', clims);
	    	            	daspect([1,1,1]);
	    	                set(gca,'YDir','reverse')
	    	                if itype == 1
								title(['t', num2str(double(qt(i))/nt*100,'%.0f'),'%-n',num2str(sum(sum(stmp>0))),'-p',num2str(gmax/gmaxLGN*100,'%.0f'),'%'], 'FontSize', 6);
	    	                end
	    	                if i == 1
	    	                    ylabel(['type: ', num2str(types(itype))]);
	    	                end
	    	                if i == nit
								ax = subplot(1, nit+1, nit+1);
	    	            		im = imagesc(stmp, clims);
								im.Visible = 0;
								ax.Visible = 0;
                                title([num2str(orient(iV1)*180/pi, '%.0f'), 'deg, ', num2str(fr(iV1),'%.2f'), 'Hz']);
	    	            		colormap('gray');
	    	                    colorbar;
	    	                end
	    	            end
	    	        end
	    	    end
			else
	        	f = figure('PaperPosition',[0, 0, nit0, ntype*nrow], 'Resize', 'off');
				set(f, 'PaperUnit', 'inches');
				assert(doubleOnOff == 1);
				sLGN = reshape(sLGN, [nLGN_1D*2, nLGN_1D, nit]);
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
	    	            stmp0 = sLGN(itype:2:(nLGN_1D*2),:,i);
						local_max = max(abs(stmp0(:)));
						if use_local_max == 1
							stmp = stmp0./local_max;
						else
							stmp = stmp0./gmax;
						end
	    	            imagesc([1 nLGN_1D], [1,nLGN_1D],stmp', clims);
	    	        	daspect([1,1,1]);
	    	            set(gca,'YDir','reverse')
						set(gca,'YTickLabel', []);
						set(gca,'XTickLabel', []);
						local_nCon = sum(sum(stmp0>=thres_out*gmax));
	    	            if itype == 1
							title(['t', num2str(double(qt(i))/nt*100,'%.0f'),'%-n',num2str(local_nCon),'-p',num2str(local_max/gmaxLGN*100,'%.0f'),'%'], 'FontSize', 5);
                        else
							title(['n',num2str(local_nCon),'-p',num2str(local_max/gmaxLGN*100,'%.0f'),'%'], 'FontSize', 5);
	    	            end
	    	            if mod(i, nit0) == 1
	    	                ylabel(['type: ', num2str(types(itype))]);
	    	            end
	    	            if i == nit
							subplot(nrow*ntype, nit0+1, iplot+1);
							stmp = stmp0./gmax;
							stmp(stmp < thres_out) = 0;
							imagesc([1 nLGN_1D], [1,nLGN_1D], stmp', clims);
	    	        		daspect([1,1,1]);
							fpos = get(gca, 'Position');
	    	            	set(gca,'YDir','reverse')
							set(gca,'YTickLabel', []);
							set(gca,'XTickLabel', []);
							title(num2str(local_nCon,'%.0f'), 'FontSize', 5)
	    	        		colormap('gray');
	    	                colorbar;
							set(gca, 'Position', fpos);
	    	            end
                        if mod(i, nit0) == 0
                            row = row + 1;
                        end
	    	        end
	    	    end
                %suptitle([num2str(nLGN_1D), 'x', num2str(nLGN_1D), 'x2: ',num2str(orient(iV1)*180/pi, '%.0f'), 'deg, (thres:',num2str(thres_out*100,'%.0f'),'%), ', num2str(fr(iV1),'%.2f'), 'Hz']);
			end
			set(f, 'OuterPosition', [.1, .1, nit+2, 4]);
			set(f, 'innerPosition', [.1, .1, nit+2, 4]);
			if mix && doubleOnOff ~= 1
	        	%saveas(f, [fig_fdr,'sLGN_V1-',num2str(iV1), osuffix, '-mix',rtime], 'fig');
	        	saveas(f, [fig_fdr,'sLGN_V1-',num2str(iV1), osuffix, '-mix',rtime,'.png']);
			else
	        	%saveas(f, [fig_fdr,'sLGN_V1-',num2str(iV1), osuffix, '-sep',rtime], 'fig');
	        	saveas(f, [fig_fdr,'sLGN_V1-',num2str(iV1), osuffix, '-sep',rtime,'.png']);
			end
	    end
    end

	if st == 2 || st == 0
	    tstep = int64(round(range_nt/nstep))
	    it = step0:tstep:nt_;
	    nstep = length(it)
		qtt = int32(floor(linspace(1,nstep,nit)));
	    tLGN_all = zeros(max_LGNperV1, nstep, ns);

	    fid = fopen(f_sLGN, 'r');
	    fseek(fid, 6*4, 0); % skip till time
	    fseek(fid, max_LGNperV1*nV1*int64(step0-1)*4, 0); % skip till time
		data = fread(fid, [nV1, max_LGNperV1], 'float')'; % transposed
	    tLGN_all(:,1,:) = data(:,V1_pick);
	    
	    for j = 2:nstep
	        fseek(fid, max_LGNperV1*nV1*int64(tstep-1)*4, 0); % skip till time
			data = fread(fid, [nV1, max_LGNperV1], 'float')'; % transposed
	        tLGN_all(:,j,:) = data(:,V1_pick);
	    end
	    fclose(fid);

	    for iq = 1:ns
	        iV1 = V1_pick(iq);
            tLGN = tLGN_all(:,:,iq);
	        gmax = max(tLGN(:));
            if gmax == 0
                continue;
            end
	        gmin = min(tLGN(:));
			if examSingle
				f = figure('PaperPosition',[.1 .1 8 8]);
				for i = 1:nLGN_V1(iV1)
					ip = LGN_V1_ID(i,iV1);
					this_type = mod(LGN_type(ip),2);
					sat = (tLGN(i,end)-gmin)/gmax;
					val = 1.0;
					if this_type == 0
						assert(mod(ip,2) == 1);
						hue = 0;
						ip = 2*fix(ip/(2*nLGN_1D))*nLGN_1D + mod((ip+1)/2-1, nLGN_1D)+1;
					else
						assert(mod(ip,2) == 0);
						hue = 2/3;
						ip = (2*ceil(ip/2/nLGN_1D)-1)*nLGN_1D + mod(ip/2-1, nLGN_1D)+1;
					end
					hsv = [hue, sat, val];	
					subplot(2*nLGN_1D, nLGN_1D, ip)
	        		plot(it*dt, tLGN(i,:)./gmax*100, '-', 'Color', hsv2rgb(hsv));
					ylim([0, 100])
					set(gca,'YTickLabel', []);
					set(gca,'XTickLabel', []);
				end
				saveas(f, [fig_fdr, 'tLGN_V1_single-',num2str(iV1), osuffix,rtime, '.png']);
			end
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
				ylim([0, 100])
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
				if i == nit
					xlabel(['strength % of max0, on:off=', num2str(onS(iV1)/offS(iV1),'%.1f')]);
				end
	        end
			set(f, 'OuterPosition', [.1, .1, 8, 6]);
			set(f, 'innerPosition', [.1, .1, 8, 6]);
	        %saveas(f, [fig_fdr, 'tLGN_V1-',num2str(iV1), osuffix,rtime], 'fig');
	        saveas(f, [fig_fdr, 'tLGN_V1-',num2str(iV1), osuffix,rtime, '.png']);
	    end
	end
end
