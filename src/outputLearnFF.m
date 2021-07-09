% connection strength heatmaps % better to choose from testLearnFF 
function outputLearnFF(isuffix, osuffix, fdr, LGN_switch, mix)
	if nargin < 4
		LGN_switch = true;
	end
	if nargin < 5
		mix = true;
	end
	fdr = [fdr, '/'];
	st = 2; %0 for temporal, 1 for spatial, 2 for both
	V1_pick = [1,10,100,1000];
	rng(1390842)
	ns = 10;
	if ~isempty(isuffix)
	    isuffix = ['_', isuffix];
	end
	if ~isempty(osuffix)
	    osuffix = ['_', osuffix];
	end
	thres = 0.0
	nstep = 0
	nbins = 20;
    nit0 = 20;

	f_sLGN = ['sLGN', osuffix, '.bin']
	LGN_V1_id_fn = ['LGN_V1_idList', isuffix, '.bin']
	fLGN_vpos = ['LGN_vpos', isuffix, '.bin'];
	fLGN_switch = ['LGN_switch', isuffix, '.bin'];
	learnDataFn = ['learnData_FF', osuffix, '.bin']

	fid = fopen(learnDataFn, 'r');
	dt = fread(fid, 1, 'float')
	fclose(fid);
	
	fid = fopen(fLGN_vpos, 'r');
	nLGN = fread(fid, 1, 'uint') % # ipsi-lateral LGN 
	nLGN_I = fread(fid, 1, 'uint') % # ipsi-lateral LGN 
	fseek(fid, (5 + (nLGN+nLGN_I)*5)*4, 0);
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

	if ~exist('nstep', 'var')
	    nstep = 0;
	end
	if nstep > nt || nstep == 0
	    nstep = nt;
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

	fid = fopen(fLGN_vpos, 'r');
    fseek(fid, (7+2*nLGN)*4, 0);
    LGN_type = fread(fid, nLGN, 'uint');
    fclose(fid);
    types = unique(LGN_type);
    ntype = length(types)

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

	if ~exist('V1_pick', 'var') 
		V1_pick = randi(768,[ns,1]);
	end
	disp(V1_pick); 
    nrow = double(idivide(int32(nit+nit0-1),int32(nit0)));
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
	        qt = int32(floor(linspace(1, nt, nit)));
	        it = diff([0, qt])-1;
	        for j = 1:nit
	            if it(j) > 0
	                fseek(fid, max_LGNperV1*nV1*int64(it(j))*4, 0); % skip till time
	            end
	            data = fread(fid, [max_LGNperV1, nV1], 'float');
	            sLGN(LGN_V1_ID(1:nLGN_V1(iV1),iV1),j) = data(1:nLGN_V1(iV1),iV1);
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
	    	            stmp = stmp./gmax;
	    	            stmp(abs(stmp)<thres) = 0;
	    	            imagesc(stmp, clims);
	    	            colormap('gray');
	    	            daspect([1,1,1]);
						axis image
	    	            %set(gca,'YDir','normal')
	    	            title(['t', num2str(double(qt(i))/nt*100,'%.0f'),'%-n',num2str(sum(sum(sLGN(:,:,i)>thres))),'-p',num2str(gmax/gmax0*100,'%.0f'),'%'], 'FontSize', 6);
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
	    	                stmp = stmp./gmax;
	    	                stmp(stmp<thres) = 0;
	    	                imagesc(stmp, clims);
	    	            	daspect([1,1,1]);
							axis image
	    	                %set(gca,'YDir','normal')
	    	                if itype == 1
								title(['t', num2str(double(qt(i))/nt*100,'%.0f'),'%-n',num2str(sum(sum(sLGN(:,:,i)>thres))),'-p',num2str(gmax/gmax0*100,'%.0f'),'%'], 'FontSize', 6);
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
	    	            stmp = stmp./gmax;
	    	            stmp(stmp<thres) = 0;
	    	            imagesc(stmp, clims);
	    	        	daspect([1,1,1]);
						axis image
	    	            %set(gca,'YDir','normal')
	    	            if itype == 1
							title(['t', num2str(double(qt(i))/nt*100,'%.0f'),'%-n',num2str(sum(sum(stmp>0))),'-p',num2str(gmax/gmax0*100,'%.0f'),'%'], 'FontSize', 6);
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
	        	saveas(f, [fdr,'sLGN_V1-',num2str(iV1), osuffix, '-mix'], 'fig');
	        	saveas(f, [fdr,'sLGN_V1-',num2str(iV1), osuffix, '-mix','.png']);
			else
	        	saveas(f, [fdr,'sLGN_V1-',num2str(iV1), osuffix, '-sep'], 'fig');
	        	saveas(f, [fdr,'sLGN_V1-',num2str(iV1), osuffix, '-sep','.png']);
			end
	    end
	    if st == 2 || st == 0
	        
	        tstep = int64(round(nt/nstep))
	        it = 1:tstep:nt;
	        nstep = length(it)
	        tLGN = zeros(max_LGNperV1, nstep);
	        

	        fid = fopen(f_sLGN, 'r');
	        fseek(fid, 6*4, 0); % skip till time
	        
	        for j = 1:nstep
	            if j > 1
	                fseek(fid, max_LGNperV1*nV1*int64(tstep-1)*4, 0); % skip till time
	            end
	            data = fread(fid, [max_LGNperV1, nV1], 'float');
	            tLGN(:,j) = data(:,iV1);
	        end
	        fclose(fid);
	       	gmax = max(tLGN(:));
	        qt = int32(floor(linspace(1,nstep,nit)));
	        f = figure('PaperPosition',[.1 .1 8 6]);
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
	            	histogram(tLGN(LGN_type(LGN_V1_ID(1:nLGN_V1(iV1),iV1)) == types(j), qt(i))./gmax*100, 'BinEdges', edges, 'FaceAlpha', 0.5);
				end
	        end
			set(f, 'OuterPosition', [.1, .1, 8, 6]);
			set(f, 'innerPosition', [.1, .1, 8, 6]);
	        saveas(f, [fdr, 'tLGN_V1-',num2str(iV1), osuffix], 'fig');
	        saveas(f, [fdr, 'tLGN_V1-',num2str(iV1), osuffix, '.png']);
	    end
	end
end
