%% script
new = true;
workpath = 'D:\scratch\patchV1\';
fdr = 'test';
if exist([workpath, fdr], 'dir')
	disp(['deleting contents in ', fdr]);
	delete([workpath, fdr,'/*'])
else
    if new
	    mkdir([workpath, fdr]);
    end
end
copyfile('*.m', [workpath, fdr]);

cwd = cd([workpath, fdr]);

parameters;