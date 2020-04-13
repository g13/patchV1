%LGN_V1_id_fn = 'LGN_V1_idList.bin';
LGN_V1_id_fn = 'LGN_V1_idList_lFF.bin';
nt_ = 1000; % choose time steps to plot
%%
fid = fopen('learnData_FF.bin', 'r');
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
assert(nt_ <= nt);
disp(['plotting time: ', num2str(dt*nt_)]);
assert(max_LGNperV1 > 0);
if rawData
    disp('will be reading from rawData for V1 spikes and gFF');
    rid = fopen('rawData.bin', 'r');
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
    %disp(nLGN_V1(i));
    if nLGN_V1(i) > 0
        LGN_V1_ID(1:nLGN_V1(i),i) = fread(sid, nLGN_V1(i), 'uint');
        %disp(LGN_V1_ID(1:nLGN_V1(i),i)');
    end
end
fclose(sid);
disp('LGN_V1 conn ID read');

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
    rid = fopen('rawData.bin', 'r'); % reopen
    fseek(rid, 7*4, 0); % skip
else
    fclose(fid);
    fid = fopen('learnData_FF.bin', 'r'); %reopen
    fseek(fid, 10*4, 0); % skip
end
%% with i spikes
i = 5;
disp(['network spiked ', num2str(sum(nsp)), ' times']);
if sum(nsp) > 0
    iV1 = [];
    ei = 0;
    while (length(iV1) == 0 || ei == 0) && i <= max(nsp)
        iV1 = find(nsp == i, 1);
        i = i+1; % if no one 
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
    end
    if i > max(nsp) 
        iV1 = randi(nV1, 1);
    end
else
    iV1 = randi(nV1, 1);
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

iV1
nsp(iV1)

nsub = 3;
if nsub > nLGN_V1(iV1)
    nsub = nLGN_V1(iV1);
end
disp(nLGN_V1(iV1));
iLGN = LGN_V1_ID(1:nLGN_V1(iV1), iV1)+1;
disp(iLGN');

iLGN_sInfo = zeros(nt_, nLGN_V1(iV1)); % LGN spike
iLTP = zeros(nt_, nLGN_V1(iV1)); %r1
% cortical
iLTD = zeros(nt_,1); %o1
iTrip = zeros(nt_,1); %o2
iAvg = zeros(nt_,1); %fr average
sLGN = zeros(nt_, nLGN_V1(iV1)); % connection strength over time
avg_total = zeros(nV1,1); % not needed

iV1_sInfo = zeros(nt_,1);
igFF = zeros(nt_,1);

nsp = zeros(nV1,1);
nsp_LGN = zeros(nLGN,1);
fstatus = 0;
if rawData 
    iv = zeros(nt_,1);
end

for i = 1:nt_
    LGN_sInfo = fread(fid, nLGN, 'float');
    assert(length(LGN_sInfo) == nLGN);
    assert(sum(LGN_sInfo < 0) == 0);
    nsp_LGN = nsp_LGN + floor(LGN_sInfo);
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
    assert(size(LGN_V1_s,1) == max_LGNperV1);
    assert(size(LGN_V1_s,2) == nV1);
    size_t = size_t + nV1*max_LGNperV1*4;
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
disp('nsp: V1')
disp([min(nsp), mean(nsp), max(nsp)]);
disp('     LGN')
disp([min(nsp_LGN), mean(nsp_LGN), max(nsp_LGN)]);
if rawData
    fclose(rid);
end
if fstatus == 0
    format long
    disp(size_t);
    [~, id] = sort(sum(floor(iLGN_sInfo), 1), 'descend'); % pick the LGNs that fires the most
    pLGN = id(1:nsub);
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
        plot(t, igFF, ':g');
        xlim([0,t(end)]);
        ymax = max(igFF);
        pick = iLGN_sInfo(:,i) > 0;
        if sum(pick) > 0
            plot((it(pick)-1+iLGN_sInfo(pick,i)' - floor(iLGN_sInfo(pick,i)'))*dt, zeros(sum(pick),1) + ymax, '*g', 'MarkerSize', 1.0);
        end
        plot(t, sLGN(:,i), '-b');
        xlim([0,t(end)]);

        yyaxis right
        if rawData 
            plot(t, iv, ':k');
        end
        if nV1sp > 0
            plot((it(v1_pick)-1+iV1_sInfo(v1_pick)' - floor(iV1_sInfo(v1_pick)'))*dt, ones(nV1sp,1), '*k', 'MarkerSize', 1.0);
        end
        plot(t, iLTP(:,i), '-g');
        plot(t, iLTD, '-m');
        plot(t, iTrip, '-y');
        plot(t-dt, iAvg, '-k');
    end
    titl = ['check-learnFF-', num2str(iV1), EI];
    title(titl)
    %%
    saveas(f, titl, 'fig');
end
% scatter plot of the spiked neurons (red)
avg_total = avg_total./nt_;
f = figure;
subplot(2,1,1)
histogram(avg_total);
subplot(2,1,2)
hold on
id = find(avg_total == 0);
plot(id, ones(length(id), 1).*avg_total(id), '*b', 'MarkerSize', 2.0);
id = find(avg_total > 0);
plot(id, ones(length(id), 1).*avg_total(id), '*r',  'MarkerSize', 2.0);
grid on
saveas(f, 'hist', 'fig');
