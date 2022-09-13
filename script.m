%% script
workpath = 'D:\scratch\patchV1\';
fdr = 'test';
if ~exist([workpath, fdr], 'dir')
    mkdir([workpath, fdr]);
end
status = copyfile('*.m', [workpath, fdr]);
if status == 1
    cwd = cd([workpath, fdr]);
end