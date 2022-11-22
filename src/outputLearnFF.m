% connection strength heatmaps % better to choose from testLearnFF 
function outputLearnFF(isuffix0, isuffix, osuffix, res_fdr, setup_fdr, data_fdr, fig_fdr, LGN_switch, mix, st, examSingle, use_local_max)
	step0 = 1; % starting time step
	nt_ = 0; % ending time step
	if nargin < 10
		disp('no default argument available');
		return
	else
		if nargin < 11
		    use_local_max = 1
		end
	end

	fig_fdr = [fig_fdr, '/'];
	res_fdr = [res_fdr, '/'];
	setup_fdr = [setup_fdr, '/'];
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
    top_thres = 0.8; % threshold to count connections of max strength, fluctuates below max
	thres_out = 0.2; % under which ratio of max LGN connection strength will not be used to plot spatial RF
	os_out = 0.5; % under which ratio of max LGN connection strength will not be used to calculate the orientation and its selectivity. 
	nstep = 1000; % total steps to sample from the trace of weight's temporal evolution.
	nbins = 20; % bins for histogram
    nit0 = 20; % number of snapshot for the spatial figure and histogram in the temporal figure
	ns = 10; % number for V1 neurons to be sampled.
	%V1_pick = [203,752,365,360,715,467,743]; % specify the IDs of V1 neurons to be sampled. If set, ns will be ignored.
    nop = 12;
	%%%%%%%%%%%%  

	f_sLGN = [data_fdr, 'sLGN', osuffix, '.bin']
	learnDataFn = [data_fdr, 'learnData_FF', osuffix, '.bin']
	V1_frFn = [data_fdr, 'max_fr', osuffix, '.bin']

	fV1_allpos = [res_fdr, 'V1_allpos', isuffix0, '.bin'];
	fLGN_vpos = [res_fdr, 'LGN_vpos', isuffix0, '.bin'];

	LGN_V1_id_fn = [setup_fdr, 'LGN_V1_idList', isuffix, '.bin']
	fLGN_switch = [setup_fdr,'LGN_switch', isuffix, '.bin'];
	fConnectome = [setup_fdr, 'connectome_cfg', isuffix, '.bin'];

	fid = fopen(fLGN_vpos, 'r');
	nLGN = fread(fid, 1, 'uint') % # ipsi-lateral LGN 
	nLGN_I = fread(fid, 1, 'uint') % # ipsi-lateral LGN 
	fseek(fid, 2*4, 'cof'); % max_ecc and x0
    xspan = fread(fid, 1, 'float');
	fseek(fid, 2*4, 'cof'); % y0 and yspan
    LGN_vpos = fread(fid,[(nLGN+nLGN_I), 2], 'float');
    LGN_type = fread(fid, nLGN, 'uint');
    types = unique(LGN_type);
    ntype = length(types);
	fseek(fid, (nLGN+nLGN_I)*2*4, 'cof'); % skip LGN_vpos in polar
	doubleOnOff = fread(fid, 1, 'int')
	nStage = fread(fid, 1, 'int')
	input_width = fread(fid, nStage*2, 'int')/2
	fclose(fid);
	
	% read the constants first only 
	fid = fopen(f_sLGN, 'r');
	nt = fread(fid, 1, 'uint');
	sampleInterval = fread(fid, 1, 'uint');
	dt = fread(fid, 1, 'float')
	nV1 = fread(fid, 1, 'uint');
	max_LGNperV1 = fread(fid, 1, 'uint')
	sRatio = fread(fid, 1, 'float')
	nLearnFF = fread(fid, 1, 'uint')
	gmaxLGN = fread(fid, nLearnFF, 'float')*sRatio
	FF_InfRatio = fread(fid, 1, 'float');
	nskip = (8 + nLearnFF);
	capacity = max_LGNperV1*sRatio*FF_InfRatio/gmaxLGN(1)/top_thres
	fclose(fid);

	fid = fopen(fV1_allpos, 'r'); % format follows patch.cu, search for the same variable name
	nblock = fread(fid, 1, 'uint');
	%blockSize = fread(fid, 1, 'uint');
	fclose(fid);

	fid = fopen(fConnectome, 'r');
	fread(fid,2,'uint');
	mE = fread(fid,1,'uint');
	blockSize = fread(fid,1,'uint');
	mI = blockSize - mE;
	synPerCon = fread(fid,4,'float'); % synapse per cortical connection in float!
	synPerFF = fread(fid,2,'float'); % synapse per FF connection in float!
	fclose(fid);

	fid = fopen(V1_frFn, 'r');
	if fid == -1
		fr = zeros(nV1,1);
	else
		fr = fread(fid, nV1, 'double');
		fclose(fid);
	end
	

	%tstep
	if nt_ > nt || nt_ == 0
		nt_ = nt;
	end
	if step0 > nt_ || step0 == 0
		step0 = 1;
	end
		
	range_nt = nt_-step0 +1;
	if range_nt == nt
		rtime = ''
	else
		rtime = ['-t', num2str(step0/nt*100,'%.0f'),'_',num2str(nt_/nt*100,'%.0f'),'%']
	end
	if nstep > range_nt || nstep == 0
	    nstep = range_nt;
	end
	step0
	nt_
	nstep

	%sample_step
	if sampleInterval > 1
		dt = sampleInterval*dt;
		nt_float = nt/sampleInterval;
		nt = floor(nt/sampleInterval);

		nt_ = round(nt_/sampleInterval);
		if nt_ >nt 
			nt_ = nt;
		end

		step0 = round(step0/sampleInterval);
		if step0 > nt_ || step0 == 0
			step0 = 1;
		end

		range_nt = nt_-step0 +1;
		if nstep > range_nt || nstep == 0
		    nstep = range_nt;
		end
		step0
		nt_
		nstep
	end


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

	nLGN_1D = sqrt(double(nLGN/2))
	%min_dis = xspan/(nLGN_1D-1);
	min_dis = 0;

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

	if nit > nt_ - step0+1
		nit = nt_ - step0+1;
	end
    qt = int32(floor(linspace(step0, nt_, nit)));

	%%%%%%%%%%%%%%%%%% HERE %%%%%%%%%%%%%%%%%%
	fid = fopen(f_sLGN, 'r');
	fseek(fid, nskip*4, 'cof'); % skip till time
    f = figure('PaperPosition',[0, 0, 4, nit*0.75]*2);

    x = ((1:nLGN_1D/2)-0.5)*sqrt(2);
    x = x + 0.01;
    x = [0, x];
    [xLGN, yLGN] = meshgrid(1:nLGN_1D, 1:nLGN_1D);
    xLGN = xLGN-nLGN_1D/2-0.5;
    yLGN = yLGN-nLGN_1D/2-0.5;
    rLGN = sqrt(xLGN.*xLGN + yLGN.*yLGN);
    disp('rLGN shape = ')
    disp(size(rLGN));
    op_edges = linspace(0,360,nop);
	x_op = (op_edges(1:nop-1) + op_edges(2:nop))/2;
	lims = zeros(2,2);
    lims(:,1) = inf;
    lims(:,2) = -inf;
	it = diff(qt)-1;
    max20 = zeros(max_LGNperV1/2, nit, 2);
    min0 = zeros(max_LGNperV1/2, nit, 2);
    picked = false(nLGN_1D,nLGN_1D,nV1,2);
    s_on = zeros(max_LGNperV1/2, nV1);
    s_off = zeros(max_LGNperV1/2, nV1);
    s_Don = zeros(length(x)-1,1);
    s_Doff = zeros(length(x)-1,1);
    n0 = 0;
    for i=1:nit
        if i == 1
	        fseek(fid, max_LGNperV1*nV1*int64(step0-1)*4, 'cof'); % skip till time
        else
            if it(i-1)>0
	            fseek(fid, max_LGNperV1*nV1*int64(it(i-1))*4, 'cof'); % skip till time
            end
        end
	    sLGN = fread(fid, [nV1, max_LGNperV1], 'float')'; % transposed

        sLGN_grid = zeros(nLGN, nV1);
	    for j = 1:nV1
	        sLGN_grid(LGN_V1_ID(1:nLGN_V1(j),j),j) = sLGN(1:nLGN_V1(j),j);
        end
        sLGN_grid = reshape(sLGN_grid, [nLGN_1D*2, nLGN_1D, nV1]);
        sLGN_on = sLGN_grid(1:2:(nLGN_1D*2),:,:);
        sLGN_off = sLGN_grid(2:2:(nLGN_1D*2),:,:);
        if i == 1
            picked(:, :, :, 1) = sLGN_on > 0;
            picked(:, :, :, 2) = sLGN_off > 0;
        end
        
	    for j = 1:nV1
            if sum(picked(:,:,j,:)) > 0
                q_on = sLGN_on(:,:,j);
                s_on(:,j) = reshape(q_on(picked(:,:,j,1)), [max_LGNperV1/2,1]);
                q_off = sLGN_off(:,:,j);
                s_off(:,j) = reshape(q_off(picked(:,:,j,2)), [max_LGNperV1/2,1]);
            else
                s_on(:,j) = 0;
                s_off(:,j) = 0;
            end
        end
        max20(:,i,1) = histcounts(sum(s_on>gmaxLGN*top_thres, 1), 'BinEdges', 0:max_LGNperV1/2);
        max20(:,i,2) = histcounts(sum(s_off>gmaxLGN*top_thres, 1), 'BinEdges', 0:max_LGNperV1/2);
        
        min0(:,i,1) = histcounts(sum(s_on==0, 1), 'BinEdges', 0:max_LGNperV1/2);
        min0(:,i,2) = histcounts(sum(s_off==0, 1), 'BinEdges', 0:max_LGNperV1/2);

		if i == nit
            nmax = sum(s_on > gmaxLGN*top_thres, 1);
            avg_on_max = mean(nmax(nmax > 0));
            nmax = sum(s_off > gmaxLGN*top_thres, 1);
            avg_off_max = mean(nmax(nmax > 0));
            nmin = sum(s_on == 0, 1);
            avg_on_min = mean(nmin(nmin > 0));
            nmin = sum(s_off == 0, 1);
            avg_off_min = mean(nmin(nmin > 0));
        end

        sLGN_on = mean(sLGN_on,3);
        sLGN_off = mean(sLGN_off,3);
        for j=1:length(x)-1
            pick = (rLGN > x(j) & rLGN < x(j+1));
            s_Don(j) = mean(sLGN_on(pick));
            s_Doff(j) = mean(sLGN_off(pick));
        end
        subplot(nit,4,4*(i-1)+4)
        hold on
        plot(x(2:end), s_Don, '-*r');
        plot(x(2:end), s_Doff, '-*b');
        ylim([0,inf])
        if i==1
	        title('str over dis')
	        ylabel('#avg str')
        end
        if i==nit
	        xlabel('dis (unit LGN)')
		else
			set(gca, 'XTickLabel', []);
        end

        [onS, offS, os, overlap, orient] = determine_os_str(LGN_vpos, LGN_V1_ID, LGN_type, sLGN, nV1, nLGN_V1, min_dis, os_out, nLGN_1D);
        subplot(nit,4,4*(i-1)+1)
		hold on
        h=histogram(onS-offS, 20);
        if h.BinEdges(1) < lims(1,1)
            lims(1,1) = h.BinEdges(1);
        end
        if h.BinEdges(end) > lims(1,2)
            lims(1,2) = h.BinEdges(end);
        end
        if i==1
	        title('On-Off balance')
	        xlabel('sOn-sOff')
	        ylabel('#V1')
        end

	    subplot(nit,4,4*(i-1)+2)
        [counts, ~, bin] = histcounts(orient*180/pi, 'BinEdges', op_edges);
        for j=1:11
            counts(j) = counts(j)*mean(os(bin == j));
        end
        bar(x_op, counts, 'FaceColor', 'b');
        if i==1
	        title('OP dist')
	        ylabel('#V1 weighted by os')
        end
        if i==nit
	        xlabel('OP (deg)')
		else
			set(gca, 'XTickLabel', []);
        end

	    subplot(nit,4,4*(i-1)+3)
		hold on
        h=histogram(os, 11);
        if h.BinEdges(1) < lims(2,1)
            lims(2,1) = h.BinEdges(1);
        end
        if h.BinEdges(end) > lims(2,2)
            lims(2,2) = h.BinEdges(end);
        end
        if i==1
	        title('OS dist')
	        ylabel('#V1')
        end
        if i==nit
			xlabel('dis/(Ron+Roff)')
        end
    end
    disp('data collected');
    for i=1:nit
	    subplot(nit,4,4*(i-1)+1)
        xlim(lims(1,:));
		if i<nit
			set(gca, 'XTickLabel', []);
		end
	    subplot(nit,4,4*(i-1)+3)
        xlim(lims(2,:));
		if i<nit
			set(gca, 'XTickLabel', []);
		end
    end

	set(f, 'OuterPosition', [.1, .1, 4, nit*0.75]*2);
	set(f, 'innerPosition', [.1, .1, 4, nit*0.75]*2);
	saveas(f, [fig_fdr, 'tOS-dist', osuffix,rtime, '.png']);
    close(f);
    disp('tOS-dist finished');

	f = figure('PaperPosition',[0, 0, 6, 6]);
	clims = [0, 1];
	subplot(2,2,1)
    counts = max20(:,:,1);
	imagesc([1, nit], [1, max_LGNperV1/2], counts./max(counts), clims);
	set(gca,'YDir','normal')
	colormap('gray');
    ylabel(['# > ', num2str(top_thres*100,'%.0f'), '% max']);
    title(['on (', num2str(avg_on_max,'%.2f'), '/', num2str(capacity,'%.2f'),')']);

	subplot(2,2,2)
    counts = max20(:,:,2);
	imagesc([1, nit], [1, max_LGNperV1/2], counts./max(counts), clims);
	colormap('gray');
    title(['off (', num2str(avg_off_max,'%.2f'), '/', num2str(capacity,'%.2f'),')']);
	set(gca,'YDir','normal')

	subplot(2,2,3)
    counts = min0(:,:,1);
	imagesc([1, nit], [1, max_LGNperV1/2], counts./max(counts), clims);
	colormap('gray');
    ylabel('# == 0');
    title(['on: ', num2str(avg_on_min,'%.2f')]);
    xlabel('sample time');
	set(gca,'YDir','normal')

	subplot(2,2,4)
    counts = min0(:,:,2);
	imagesc([1, nit], [1, max_LGNperV1/2], counts./max(counts), clims);
	colormap('gray');
    title(['off: ', num2str(avg_off_min,'%.2f')]);
	set(gca,'YDir','normal')

	set(f, 'OuterPosition', [.1, .1, max_LGNperV1/2, nit*0.75]/4);
	set(f, 'innerPosition', [.1, .1, max_LGNperV1/2, nit*0.75]/4);
	saveas(f, [fig_fdr, 'max_min_tDist', osuffix,rtime, '.png']);
    close(f);
    disp('max_min_tDist finished');

	fseek(fid, max_LGNperV1*nV1*int64(nt_-1)*4 + nskip*4, 'bof'); % skip till time
	sLGN = fread(fid, [nV1, max_LGNperV1], 'float')'; % transposed
    fclose(fid);

    [onS, offS, os, overlap, orient] = determine_os_str(LGN_vpos, LGN_V1_ID, LGN_type, sLGN, nV1, nLGN_V1, min_dis, os_out, nLGN_1D);

	epick = zeros(mE*nblock,1);
	ipick = zeros(mI*nblock,1);
	for i = 1:nblock
		epick(((i-1)*mE+1):(i*mE)) = (i-1)*blockSize + (1:mE);
		ipick(((i-1)*mI+1):(i*mI)) = (i-1)*blockSize + mE + (1:mI);
	end

	f = figure('PaperPosition',[0, 0, 6, 6]);
	subplot(2,2,1)
	dS = onS - offS;
    [~, dS_edges, ~] = histcounts(dS, nbins);
    [countsE, ~, ~] = histcounts(dS(epick), 'BinEdges', dS_edges);
    [countsI, ~, ~] = histcounts(dS(ipick), 'BinEdges', dS_edges);
	x_ds = (dS_edges(1:nbins) + dS_edges(2:nbins+1))/2;
    hold on
    b = bar(x_ds, countsE, 'FaceColor', 'r');
    b = bar(x_ds, countsI, 'FaceColor', 'b');
    hold off
	legend('E', 'I')
	xlabel('sOn-sOff')
	ylabel('#V1')
    alpha(.5);

	subplot(2,2,2)
    [countsE, ~, binE] = histcounts(orient(epick)*180/pi, 'BinEdges', linspace(0,360,nop));
    [countsI, ~, binI] = histcounts(orient(ipick)*180/pi, 'BinEdges', linspace(0,360,nop));
    hold on
    b = bar(x_op, countsE, 'FaceColor', 'r');
    b = bar(x_op, countsI, 'FaceColor', 'b');
    hold off
	xlabel('OP (deg)')
	ylabel('#V1')
    alpha(.5);
	
	subplot(2,2,4)
	osE = os(epick);
	osI = os(ipick);
	overlapE = overlap(epick);
	overlapI = overlap(ipick);
    ocountsE = countsE;
    ocountsI = countsI;
    for i=1:(nop-1)
        countsE(i) = countsE(i)*mean(osE(binE == i));
        countsI(i) = countsI(i)*mean(osI(binI == i));
        ocountsE(i) = ocountsE(i)*(1-mean(overlapE(binE == i)));
        ocountsI(i) = ocountsI(i)*(1-mean(overlapI(binI == i)));
    end
    hold on
    bar(x_op, countsE, 'FaceColor', 'r');
    bar(x_op, countsI, 'FaceColor', 'b');
    bar(x_op, ocountsE, 'FaceColor', 'm');
    bar(x_op, ocountsI, 'FaceColor', 'c');
    hold off
	xlabel('OP (deg)')
	ylabel('#V1 weighted by os')
    alpha(.5);

	subplot(2,2,3)
	osEdges = linspace(0,1,nop);
    [countsE, ~, ~] = histcounts(osE, 'BinEdges', osEdges);
    [countsI, ~, ~] = histcounts(osI, 'BinEdges', osEdges);
    [ocountsE, ~, ~] = histcounts(overlapE, 'BinEdges', osEdges);
    [ocountsI, ~, ~] = histcounts(overlapI, 'BinEdges', osEdges);
	x_os = (osEdges(1:nop-1) + osEdges(2:nop))/2;
    hold on
    bar(x_os, countsE, 'FaceColor', 'r');
    bar(x_os, countsI, 'FaceColor', 'b');
    bar(x_os, ocountsE, 'FaceColor', 'm');
    bar(x_os, ocountsI, 'FaceColor', 'c');
    hold off
	xlabel('dis/(Ron+Roff) or overlap')
	ylabel('#V1')
    alpha(.5);

	set(f, 'OuterPosition', [.1, .1, 6, 6]);
	set(f, 'innerPosition', [.1, .1, 6, 6]);
	saveas(f, [fig_fdr, 'stats-LGN_V1', osuffix,rtime, '.png']);
    close(f);
    disp('stats finished');
	%%%%%%%%%%%%%%%%%% HERE %%%%%%%%%%%%%%%%%%

	if st == 2 || st == 1
	    sLGN_all = zeros(nLGN, nit, ns);
	    fid = fopen(f_sLGN, 'r');
	    fseek(fid, nskip*4, 'cof'); % skip till time
	    
	    % skip times
	    %ht = round(nt/2);
	    %it = [0, ht-1, nt-1 - (ht+1)]
	    for j = 1:nit
	        if j == 1
	            fseek(fid, max_LGNperV1*nV1*int64(step0-1)*4, 'cof'); % skip till time
            else
	            if it(j-1) > 0
	                fseek(fid, max_LGNperV1*nV1*int64(it(j-1))*4, 'cof'); % skip till time
	            end
            end
	        data = fread(fid, [nV1, max_LGNperV1], 'float')'; % transposed
	    	for i=1:ns
	            iV1 = V1_pick(i);
	            sLGN_all(LGN_V1_ID(1:nLGN_V1(iV1),iV1),j,i) = data(1:nLGN_V1(iV1),iV1);
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
							stmp = stmp0./gmax;
						else
							stmp = stmp0./gmaxLGN;
						end
	    	            imagesc([1 nLGN_1D], [1,nLGN_1D],stmp', clims);
	    	        	daspect([1,1,1]);
	    	            set(gca,'YDir','reverse')
						set(gca,'YTickLabel', []);
						set(gca,'XTickLabel', []);
						local_nCon = sum(sum(stmp0>=thres_out*gmaxLGN));
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
            close(f);
	    end
    end
    disp('sLGN finished');

	if st == 2 || st == 0
		tstep = int64(round(range_nt/nstep))
		it = step0:tstep:nt_;
	    nstep = length(it)
		qtt = int32(floor(linspace(1,nstep,nit)));
	    tLGN_all = zeros(max_LGNperV1, nstep, ns);

	    fid = fopen(f_sLGN, 'r');
	    fseek(fid, nskip*4, 'cof'); % skip till time
	    fseek(fid, max_LGNperV1*nV1*int64(step0-1)*4, 'cof'); % skip till time
		data = fread(fid, [nV1, max_LGNperV1], 'float')'; % transposed
	    tLGN_all(:,1,:) = data(:,V1_pick);
	    
	    for j = 2:nstep
	        fseek(fid, max_LGNperV1*nV1*int64(tstep-1)*4, 'cof'); % skip till time
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
			if use_local_max ~= 1
				gmax = gmaxLGN;
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
                close(f);
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
            close(f)
	    end
	end
    disp('tLGN finished');
end


function [onS, offS, os, overlap, orient] = determine_os_str(pos, id, type, s, n, m, min_dis, os_out, nLGN_1D)
    orient = zeros(n,1);
    overlap = zeros(n,1);
    os = zeros(n,1);
	onS = zeros(n,1);
	offS = zeros(n,1);
    for i = 1:n
        all_id = id(1:m(i),i);
        all_type = type(all_id);
        all_s = s(1:m(i),i);

        sLGN = zeros(nLGN_1D*nLGN_1D*2,1);
	    sLGN(all_id) = all_s; 
		sLGN = reshape(sLGN, [nLGN_1D*2, nLGN_1D]);
	    on_s = sLGN(1:2:(nLGN_1D*2),:);
	    off_s = sLGN(2:2:(nLGN_1D*2),:);
        overlap_pick = and(on_s>0, off_s>0);
        max_s = max(on_s, off_s);
        min_s = min(on_s, off_s);
        smaller_size = min(sum(on_s>0,'all'), sum(off_s>0,'all'));
        overlap(i) = sum(min_s(overlap_pick)./max_s(overlap_pick),'all')/smaller_size;

        on_s = all_s(all_type == 4);
        on_id = all_id(all_type == 4);
		sPick = on_s > max(on_s) * os_out;
        onPick = on_id(sPick);
		onS(i) = sum(on_s);
        on_pos = mean(pos(onPick,:), 1);

        off_s = all_s(all_type == 5);
        off_id = all_id(all_type == 5);
		sPick = off_s > max(off_s) * os_out;
        offPick = off_id(sPick);
		offS(i) = sum(off_s);
        off_pos = mean(pos(offPick,:), 1);

        dis_vec = [on_pos(1)-off_pos(1), off_pos(2)-on_pos(2)];
		on_off_dis = sqrt(dis_vec*dis_vec');
		if on_off_dis <= min_dis/2
            orient(i) = nan;
            os(i) = 0;
        else
			orient(i) = atan2(off_pos(2)-on_pos(2), on_pos(1)-off_pos(1)); % spin around as the imagesc
        	if orient(i) < 0
        	    orient(i) = orient(i) + 2*pi;
        	end
		    proj = dis_vec./on_off_dis;

		    if length(onPick) > 1
		    	rel_pos = pos(onPick,:) - on_pos;
            	on_dis = sqrt(sum(rel_pos.*rel_pos,2)); % std of dist to center
		    	cos_on = (rel_pos./on_dis)*proj';
            	proj_on = on_dis.*cos_on; % ??
		    	max_on_p = max(proj_on(proj_on>=0));
		    	max_on_m = max(abs(proj_on(proj_on<=0)));
		    	r_on = max_on_p*(max_on_p/max_on_m);
		    	if isnan(r_on)
		    		r_on = min_dis/2;
		    	end
		    else
		    	r_on = min_dis/2;
		    end

		    if length(offPick) > 1
		    	rel_pos = pos(offPick,:) - off_pos;
            	off_dis = sqrt(sum(rel_pos.*rel_pos,2));
		    	cos_off = (rel_pos./off_dis)*proj';
            	proj_off = off_dis.*cos_off;
		    	max_off_p = max(proj_off(proj_off>=0));
		    	max_off_m = max(abs(proj_off(proj_off<=0)));
		    	r_off = max_off_p*(max_off_p/max_off_m);
		    	if isnan(r_off)
		    		r_off = min_dis/2;
		    	end
		    else
		    	r_off = min_dis/2;
		    end

            os(i) = on_off_dis/(r_on+r_off);
        end
    end
end
