% connection heatmaps
fLGN_vpos = 'LGN_vpos_lFF.bin'
f_sLGN = 'sLGN.bin'
LGN_V1_id_fn = 'LGN_V1_idList_lFF.bin';

fid = fopen(fLGN_vpos, 'r');
nLGN = fread(fid, 1, 'uint') % # ipsi-lateral LGN 
fread(fid, 1, 'uint'); % skip contra-lateral LGN all from the ipsi eye
fread(fid, 1, 'float'); % skip max-ecc

% not needed 
x0 = fread(fid, 1, 'float'); % x0
xspan = fread(fid, 1, 'float'); % xspan
y0 = fread(fid, 1, 'float'); % y0
yspan = fread(fid, 1, 'float'); % yspan
LGN_x = fread(fid, nLGN, 'float');
LGN_y = fread(fid, nLGN, 'float');

nLGN_1D = sqrt(double(nLGN))

% read connection id
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

sLGN = zeros(nLGN_1D, nLGN_1D, 3);

fid = fopen(f_sLGN, 'r');
nt = fread(fid, 1, 'uint');
nV1 = fread(fid, 1, 'uint');
max_LGNperV1 = fread(fid, 1, 'uint')

% sample connection
while iV1 > 768 % make sure is excitatory 
    iV1 = randi(nV1, 1);
end
% better to choose from testLearnFF
iV1 = 91


ht = round(nt/2);
% skip times
it = [0, ht, nt-1 - (ht+2)]
for j = 1:3
    if it(j) > 0
        fseek(fid, max_LGNperV1*nV1*it(j)*4, 0); % skip till time
    end
    data = fread(fid, [max_LGNperV1, nV1], 'float');
    for i = 1:nLGN_V1(iV1)
        iLGN = LGN_V1_ID(i, iV1);
        [row, col] = ind2sub([nLGN_1D, nLGN_1D], iLGN);
        sLGN(row,col,j) = data(i,iV1);
    end
end
fclose(fid);
gmax = 0.02
f = figure();
subplot(1,3,1)
%heatmap(sLGN(:,:,1));
imagesc(sLGN(:,:,1)./gmax);
colorbar;
title('initial');

subplot(1,3,2)
%heatmap(sLGN(:,:,2));
imagesc(sLGN(:,:,2)./gmax);
colorbar;
title('halfway');

subplot(1,3,3)
%heatmap(sLGN(:,:,3)); % old matlab version may not have heatmap function
%imagesc(sLGN(:,:,3)./gmax*255);
imagesc(sLGN(:,:,3)./gmax);
%daspect([1,1,1]);
colorbar;
title(['final LGN->V1 #', num2str(iV1)]);
saveas(f, ['LGN_V1_',num2str(iV1)], 'fig');

disp(sum(sum(abs(sLGN(:,:,2) - sLGN(:,:,1))))); % total half change
disp(sum(sum(abs(sLGN(:,:,3) - sLGN(:,:,1))))); % total change
