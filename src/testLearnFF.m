%nt_ = 800000; % choose time steps to plot
nt_ = 800000; % choose time steps to plot
clear iV1
targetfr = 5
iV1 = randi(768,1);
nsub = 3;
thres = 0.5;  % as a percent of gmax
suffix = 'lFF';
if ~isempty(suffix) 
    suffix = ['_', suffix];
end
learnDataFn = ['learnData_FF', suffix, '.bin']
rawDataFn = ['rawData', suffix, '.bin']
LGN_V1_id_fn = ['LGN_V1_idList', suffix, '.bin']
f_sLGN = ['sLGN', suffix, '.bin']
f_LGNoutput = ['outputB4V1', suffix, '.bin']
%%

fid = fopen(f_sLGN, 'r');
nt = fread(fid, 1, 'uint');
nV1 = fread(fid, 1, 'uint');
max_LGNperV1 = fread(fid, 1, 'uint')
sRatio = fread(fid, 1, 'float')
nLearnFF = fread(fid, 1, 'uint')
gmaxLGN = fread(fid, nLearnFF, 'float')
gmax = gmaxLGN(1)*sRatio; % TODO, separate E and I
fclose(fid);
thres = gmax*thres;

fid = fopen(learnDataFn, 'r');
dt = fread(fid, 1, 'float')
nt = fread(fid, 1, 'uint')
nLGN = fread(fid, 1, 'uint')
mE = fread(fid, 1, 'uint')
mI = fread(fid, 1, 'uint')
blockSize = mE + mI
nblock = fread(fid, 1, 'uint');
nE = nblock*mE
nI = nblock*mI
nV1 = nE + nI
rawData = fread(fid, 1, 'uint')
max_LGNperV1 = fread(fid, 1, 'uint')
nLearnTypeFF_E = fread(fid, 1, 'uint')
nLearnTypeFF_I = fread(fid, 1, 'uint')
nLearnTypeFF = nLearnTypeFF_I + nLearnTypeFF_E;
size_t = int64(10*4);
if nt_ > nt
    nt_ = nt;
end
disp(['plotting time: ', num2str(dt*nt_)]);
assert(max_LGNperV1 > 0);
if rawData
    disp('will be reading from rawData for V1 spikes and gFF');
    rid = fopen(rawDataFn, 'r');
    % skip some constants
    fread(rid, 1, 'float')
    fread(rid, 1, 'uint') 
    fread(rid, 1, 'uint')
    haveH = fread(rid, 1, 'uint')
    ngTypeFF = fread(rid, 1, 'uint')
    ngTypeE = fread(rid, 1, 'uint')
    ngTypeI = fread(rid, 1, 'uint')
end
disp('const read');

%%
sid = fopen(LGN_V1_id_fn, 'r');
LGN_V1_ID = zeros(max_LGNperV1, nV1);
nLGN_V1 = zeros(nV1,1);
for i = 1:nV1
    nLGN_V1(i) = fread(sid, 1, 'uint');
    assert(nLGN_V1(i) <= max_LGNperV1);
    if nLGN_V1(i) > 0
        LGN_V1_ID(1:nLGN_V1(i),i) = fread(sid, nLGN_V1(i), 'uint') + 1;
    end
end
fclose(sid);
disp('LGN_V1 conn ID read');

if ~exist('iV1', 'var')
    %% find a test neuron
    nsp = zeros(nV1,1);
    % read V1 spikes only
    for i = 1:nt_
        if rawData 
            V1_sInfo = fread(rid, nV1, 'float');
            fseek(rid, (1+(1+haveH)*(ngTypeFF + ngTypeE + ngTypeI))*nV1*4, 0); % skip
        else
            fseek(fid, (1+nLearnTypeFF)*nLGN*4, 0); % skip
            V1_sInfo = fread(fid, nV1, 'float');
            skip_size = nV1 + nV1*max_LGNperV1 + 2*(nE*nLearnTypeFF_E + nI*nLearnTypeFF_I) + (nE+nV1) ;
            fseek(fid, skip_size*4, 0); % skip
        end
        assert(sum(V1_sInfo >=0) == nV1);
        assert(sum(V1_sInfo > 0 & V1_sInfo < 1) == 0);
        assert(sum(V1_sInfo < 2) == nV1);
        nsp = nsp + floor(V1_sInfo);
    end
    if rawData
        fclose(rid);
        rid = fopen(rawDataFn, 'r'); % reopen
        fseek(rid, 7*4, 0); % skip
    else
        fclose(fid);
        fid = fopen(learnDataFn', 'r'); %reopen
        fseek(fid, 10*4, 0); % skip
    end
    %% with i spikes
    disp(max(nsp));
    i = targetfr;
    if i > max(nsp)
        i = max(nsp);
    end
    disp(['network spiked ', num2str(sum(nsp)), ' times']);
    if sum(nsp) > 0
        iV1 = [];
        ei = 0;
        while (length(iV1) == 0 || ei == 0) && i <= max(nsp)
            iV1 = find(nsp == i, 1);
            % detemine EI
            if length(iV1)
                iblock = idivide(int32(iV1-1),int32(blockSize));
                if mod(iV1-1,blockSize) < mE
                    j = mE*iblock + (iV1 - blockSize*iblock)
                    EI = 'E'
                    ei = 1;
                else
                    j = mI*iblock + (iV1 - blockSize*iblock - mE)
                    EI = 'I'
                    ei = 0;
                end
            end
            i = i + 1;
        end
        if length(iV1) == 0
            iV1 = randi(mE, 1);
        end
    else
        iV1 = randi(nV1, 1);
    end
    nsp(iV1)
    disp(['this V1 firing rate ', num2str(nsp(iV1)/(nt_*dt/1000)), 'Hz']);
end

% detemine EI
iblock = idivide(int32(iV1-1),int32(blockSize));
if mod(iV1-1,blockSize) < mE
    j = mE*iblock + (iV1 - blockSize*iblock)
    EI = 'E'
    ei = 1;
else
    j = mI*iblock + (iV1 - blockSize*iblock - mE)
    EI = 'I'
    ei = 0;
end

oid = fopen(f_LGNoutput, 'r');
fseek(oid, 3*4, 0);
LGNfr = zeros(nLGN, nt_);
for it = 1:nt_
    if it > 1
        fseek(oid, nLGN*4*4, 0);
    end
    LGNfr(:, it) = fread(oid, nLGN, 'float');
end
fclose(oid);

if nsub > nLGN_V1(iV1)
    nsub = nLGN_V1(iV1);
end
disp(['number of LGN connections: ', num2str(nLGN_V1(iV1))]);
iLGN = LGN_V1_ID(1:nLGN_V1(iV1), iV1);

iLGN_sInfo = zeros(nt_, nLGN_V1(iV1)); % LGN spike
iLTP = zeros(nt_, nLGN_V1(iV1)); %r1
% cortical
iLTD = zeros(nt_,1); %o1
iTrip = zeros(nt_,1); %o2
iAvg = zeros(nt_,1); %fr average
sLGN = zeros(nt_, nLGN_V1(iV1)); % connection strength over time
sLGN_final = zeros(nV1,1);
nLGN_final = zeros(nV1,1);
avg_total = zeros(nV1,1); % not needed

iV1_sInfo = zeros(nt_,1);
igFF = zeros(nt_,1);

nsp = zeros(nV1,1);
nsp_LGN = zeros(nLGN,1);
fstatus = 0;
if rawData 
    iv = zeros(nt_,1);
end
iV1

tsp_LGN = [];
id_LGN = [];
id = (1:nLGN)';
for i = 1:nt_
    LGN_sInfo = fread(fid, nLGN, 'float');
    nsp_ = floor(LGN_sInfo);
    id_LGN = [id_LGN; id(LGN_sInfo > 0)];
    tsp_LGN = [tsp_LGN; (i + (LGN_sInfo(nsp_>0) - nsp_(nsp_>0)))*dt];
    nsp_LGN = nsp_LGN + nsp_;
    vLTP = fread(fid, [nLGN, nLearnTypeFF], 'float');
    assert(size(vLTP,1) == nLGN);
    assert(size(vLTP,2) == nLearnTypeFF);
    for k=1:nLGN_V1(iV1)
        iLTP(i,k) = vLTP(iLGN(k),1);
    end
    size_t = size_t + nLGN*(1+nLearnTypeFF)*4;
    if rawData 
        V1_sInfo = fread(rid, nV1, 'float');
        assert(length(V1_sInfo) == nV1);
        v = fread(rid, nV1, 'float');
        assert(length(v) == nV1);
        iv(i) = v(iV1);
        gFF = fread(rid, nV1, 'float'); % only the first type of gFF is needed, thus only the first type is read
        assert(length(gFF) == nV1);
        fstatus = fseek(rid, ((1+haveH)*(ngTypeFF + ngTypeE + ngTypeI) - 1)*nV1*4, 0); % skip
        if fstatus == -1
            disp(ferror(rid));
            break;
        end
    else
        V1_sInfo = fread(fid, nV1, 'float');
        gFF = fread(fid, nV1, 'float');
        assert(length(gFF) == nV1);
        size_t = size_t + nV1*2*4;
    end
    assert(sum(V1_sInfo < 0) == 0);
    nsp = nsp + floor(V1_sInfo);
    LGN_V1_s = fread(fid, [max_LGNperV1, nV1], 'float');
    if i==nt_
        sLGN_final = sum(LGN_V1_s);
        nLGN_final = sum(LGN_V1_s>thres);
    end
    assert(size(LGN_V1_s,1) == max_LGNperV1);
    assert(size(LGN_V1_s,2) == nV1); size_t = size_t + nV1*max_LGNperV1*4;
    vLTD_E = fread(fid, [nE, nLearnTypeFF_E], 'float');
    assert(size(vLTD_E,1) == nE);
    assert(size(vLTD_E,2) == nLearnTypeFF_E);
    vTripE = fread(fid, [nE, nLearnTypeFF_E], 'float');
    assert(size(vTripE,1) == nE);
    assert(size(vTripE,2) == nLearnTypeFF_E);

    if nLearnTypeFF_I > 0
        vLTD_I = fread(fid, [nI, nLearnTypeFF_I], 'float');
        assert(size(vLTD_I,1) == nI);
        assert(size(vLTD_I,2) == nLearnTypeFF_I);
        vTripI = fread(fid, [nI, nLearnTypeFF_I], 'float');
        assert(size(vTripI,1) == nI);
        assert(size(vTripI,2) == nLearnTypeFF_I);
    end

    size_t = size_t + 2*(nE*nLearnTypeFF_E + nI*nLearnTypeFF_I)*4;
    vAvg = zeros(nV1, 1); 
    data = fread(fid, [2,nE], 'float');
    vAvg(1:nE) = data(1,:)';
    vAvg((nE+1):nV1) = fread(fid, nI, 'float');
    size_t = size_t + (nE+nV1)*4;

    if ei == 1
        iLTD(i) = vLTD_E(j,1);
        iTrip(i) = vTripE(j,1);
    else
        if nLearnTypeFF_I > 0
            iLTD(i) = vLTD_I(j,1);
            iTrip(i) = vTripI(j,1);
        end
    end
    iAvg(i) = vAvg(iV1);
    for k = 1:nLGN_V1(iV1)
        iLGN_sInfo(i,k) = LGN_sInfo(iLGN(k));
        sLGN(i,k) = LGN_V1_s(k, iV1);
    end
    iV1_sInfo(i) = V1_sInfo(iV1);
    igFF(i) = gFF(iV1);
    avg_total = avg_total + vAvg;
end

fclose(fid);
disp('V1 firing across population in Hz:')
disp([min(nsp), mean(nsp), max(nsp)]./(nt_*dt/1000));
disp('firing rate across LGN in Hz')
disp([min(nsp_LGN), mean(nsp_LGN), max(nsp_LGN)]./(nt_*dt/1000));

f = figure;
titl = 'LGNspike_scatter';
plot(tsp_LGN, id_LGN, '*k');
saveas(f, titl, 'fig');

if rawData
    fclose(rid);
end
if fstatus == 0
    format long
    disp(size_t);
    %[~, id] = sort(sum(floor(iLGN_sInfo), 1), 'descend'); % pick the LGNs that fires the most
    %pLGN = id(1:nsub);
    %pLGN = randi(nLGN_V1(iV1), nsub,1)
    %nLGN_1D = int32(sqrt(nLGN));
    pLGN = int32([6*16+7, 7*16+7, 8*16+7]);
    %pLGN = [1,2,3]
    disp(iLGN(pLGN)');
    f = figure;
    it = 1:nt_; % end points, spike times counts from start points, (iAvg as well, for now, TODO)
    t = it*dt; 
    v1_pick = iV1_sInfo > 0;
    nV1sp = sum(v1_pick);
    disp(['V1 fired ', num2str(nV1sp), ' times']);
    for j = 1:nsub
        i = pLGN(j);
        subplot(nsub,1,j);
        hold on
        yyaxis left
        lines = [];
        labels = {};
        gl = plot(t, igFF, ':g');
        lines = [lines, gl];
        labels = [labels, 'LGN cond'];
        xlim([0,t(end)]);
        max_igFF = max(igFF);
        ylim([0,max_igFF*1.1]);
        ss = sLGN(:,i)/gmax*max_igFF;
        gs = plot(t, ss, '-b');
        lines = [lines, gs];
        labels = [labels, 'LGN strength'];
        max_gFr = max(LGNfr(iLGN(pLGN(j)),:));
        gfr = plot(t, LGNfr(iLGN(pLGN(j)),:)./max_gFr*max_igFF, '-y');
        lines = [lines, gfr];
        labels = [labels, 'LGN fr'];
        xlim([0,t(end)]);

        pick = iLGN_sInfo(:,i) > 0;
        if sum(pick) > 0
            gsp = plot((it(pick)-1+iLGN_sInfo(pick,i)' - floor(iLGN_sInfo(pick,i)'))*dt, zeros(sum(pick),1) + ss(pick), '*g', 'MarkerSize', 2.0);
            lines = [lines, gsp];
            labels = [labels, 'LGN tsp'];
        end

        if nV1sp > 0
            v1sp = plot((it(v1_pick)-1+iV1_sInfo(v1_pick)' - floor(iV1_sInfo(v1_pick)'))*dt, zeros(nV1sp,1) + ss(v1_pick), '*k', 'MarkerSize', 2.0);
            lines = [lines, v1sp];
            labels = [labels, 'V1 tsp'];
        end

        if j == 1
            legend(lines, labels, 'Location', 'northwest');
        end

        yyaxis right
        lines = [];
        labels = {};
        if rawData 
            vl = plot(t, iv, ':k');
            lines = [lines, vl];
            labels = [labels, 'voltage'];
        end
        %if nV1sp > 0
        %    v1sp = plot((it(v1_pick)-1+iV1_sInfo(v1_pick)' - floor(iV1_sInfo(v1_pick)'))*dt, ones(nV1sp,1), '*k', 'MarkerSize', 2.0);
        %    lines = [lines, v1sp];
        %    labels = [labels, 'V1 tsp'];
        %end
        r1 = plot(t, iLTP(:,i), '-g');
        lines = [lines, r1];
        labels = [labels, 'r1'];

        %ymax = max(iLTP(:,i));
        %pick = iLGN_sInfo(:,i) > 0;
        %if sum(pick) > 0
        %    gsp = plot((it(pick)-1+iLGN_sInfo(pick,i)' - floor(iLGN_sInfo(pick,i)'))*dt, zeros(sum(pick),1) + ymax, '*g', 'MarkerSize', 2.0);
        %    lines = [lines, gsp];
        %    labels = [labels, 'LGN tsp'];
        %end

        o1 = plot(t, iLTD, '-m');
        lines = [lines, o1];
        labels = [labels, 'o1'];

        o2 = plot(t, iTrip, '-r');
        lines = [lines, o2];
        labels = [labels, 'o2'];

        avgl = plot(t-dt, iAvg, '-k');
        lines = [lines, avgl];
        labels = [labels, '<fr>'];
        if j == 1
            legend(lines, labels, 'Location', 'northwest');
        end
    end
    titl = ['check-learnFF-', num2str(iV1), EI];
    title(titl)
    %%
    saveas(f, titl, 'fig');
end

fid = fopen(['LGN_fr', suffix, '.bin'], 'r');
fread(fid, 2, 'uint');
LGN_fr = fread(fid,[nLGN, nt_], 'float');
fclose(fid);
f = figure;
subplot(2,2,1)
%hold on
%h = histogram(nsp_LGN./(nt_*dt/1000), 50);
%histogram(mean(LGN_fr'), h.BinEdges)
%title('LGN FR dist.');

fid = fopen(f_sLGN, 'r');
nt = fread(fid, 1, 'uint');
nV1 = fread(fid, 1, 'uint');
max_LGNperV1 = fread(fid, 1, 'uint')
sRatio = fread(fid, 1, 'float')
nLearnFF = fread(fid, 1, 'uint')
gmaxLGN = fread(fid, nLearnFF, 'float')
gmax = gmaxLGN(1)*sRatio; % TODO, separate E and I

thres = gmax*thres;
sLGN = zeros(nLGN, 2, nV1);
data = fread(fid, [max_LGNperV1, nV1], 'float');
for i=1:nV1
    sLGN(LGN_V1_ID(1:nLGN_V1(i),i),1,i) = data(:,i);
end
fseek(fid, max_LGNperV1*nV1*(nt-2)*4, 0); % skip till end 
data = fread(fid, [max_LGNperV1, nV1], 'float');
for i=1:nV1
    sLGN(LGN_V1_ID(1:nLGN_V1(i),i),2,i) = data(:,i);
end
fclose(fid);
% normalize
sLGN = sLGN - repmat(mean(sLGN), [nLGN,1]);
max_sLGN = max(abs(sLGN));
%take pearson corr coef
sLGN = sLGN./(repmat(max_sLGN, [nLGN,1]));
spat_corr = mean(sLGN(:,1,:).*sLGN(:,2,:));
plot(nsp./(nt_*dt/1000), squeeze(spat_corr), '*g');
xlabel('V1 fr Hz')
ylabel('spatial pearson corr')

% scatter plot of the spiked neurons (red)
subplot(2,2,2)
hold on
id = find(nsp == 0);
plot(id, ones(length(id), 1).*nsp(id)./(nt_*dt/1000), '*b', 'MarkerSize', 2.0);
id = find(nsp > 0);
plot(id, ones(length(id), 1).*nsp(id)./(nt_*dt/1000), '*r',  'MarkerSize', 2.0);
grid on
subplot(2,2,3)
plot(nsp./(nt_*dt/1000), sLGN_final, '*b');
xlabel('V1 fr Hz')
ylabel('sLGN sum streng Hz')
subplot(2,2,4)
plot(nsp./(nt_*dt/1000), nLGN_final, '*b');
xlabel('V1 fr Hz')
ylabel(['nLGN with s > ', num2str(thres)])
saveas(f, 'hist', 'fig');
