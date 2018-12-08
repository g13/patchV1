n = 1024;
% n = 160*128;
run_t = 25;
nstep = 200;
dt = run_t/nstep;
ngE = 2;
ngI = 1;
nstep = nstep + 1;
t = (0:(nstep-1)) * dt;
%%
figure;
vid = fopen('v_ictorious.bin','r');
v = fread(vid,[n,nstep],'double');
fclose(vid);
subplot(2,3,1)
plot(t, v);
ylim([-2/3,1]);

gEid = fopen('gE_nerous.bin','r');
gE = fread(gEid,[n,ngE*nstep],'double');
fclose(gEid);
gE = reshape(gE,[n,ngE,nstep]);
subplot(2,3,2)
hold on
plot(t, squeeze(gE(:,1,:)));
plot(t, squeeze(gE(:,2,:)));

gIid = fopen('gI_berish.bin','r');
gI = fread(gIid,[n,ngI*nstep],'double');
fclose(gIid);
gI = reshape(gI,[n,ngI,nstep]);
subplot(2,3,3)
plot(t, squeeze(gI(:,1,:)));
%%
vid = fopen('v_CPU.bin','r');
v = fread(vid,[n,nstep],'double');
fclose(vid);
subplot(2,3,4)
plot(t, v);
ylim([-2/3,1]);

gEid = fopen('gE_CPU.bin','r');
gE = fread(gEid,[n*ngE,nstep],'double');
fclose(gEid);
gE = reshape(gE,[n,ngE,nstep]);
%gE = permute(gE,[2,1,3]);       
subplot(2,3,5)
hold on
plot(t, squeeze(gE(:,1,:)));
plot(t, squeeze(gE(:,2,:)));

gIid = fopen('gI_CPU.bin','r');
gI = fread(gIid,[n*ngI,nstep],'double');
fclose(gIid);
gI = reshape(gI,[n,ngI,nstep]);
%gI = permute(gI,[2,1,3]);
subplot(2,3,6)
plot(t, squeeze(gI(:,1,:)));