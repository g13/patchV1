import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
from grating import *
from ext_signal import *

stage = 2

if stage == 2:
    # setup stage II retinal waves 
    nseq = 40
    video_fn = f'{nseq}_retinal_wave_II'
    reverse = False
    resolution = 128
    seed = 3536653
    speed = 4.0 # degree/s
    range_deg = 10.0 # eccentricity from the original
    SF = np.array([1/20, 1/20])
    center = np.pi/2
    wing = np.pi/2
    amp = np.array([1.0,1.0])
    overlap = 1
    sharpness = np.array([-1,-1])
    phase = np.zeros(2) # initialize, make sure size is correct
    frameRate = 30
    
    np.random.seed(seed)
    orient = np.array([np.pi*1/4, np.pi*3/4, np.pi*5/4, np.pi*7/4])
    #orient = np.array([np.pi*1/4, np.pi*3/4])
    #orient = np.random.rand(nseq)*2*np.pi
    #orient = np.random.permutation(np.arange(nseq)/nseq*np.pi*2)
    # video file has x-axis flipped orientation
    flipped_orient = 2*np.pi - orient
    fig = plt.figure('orient dist.', figsize = (8,4), dpi =300)
    ax = fig.add_subplot(121, polar=True)
    edges = np.linspace(0, 2*np.pi, 30)
    print(edges)
    counts, _ = np.histogram(flipped_orient, bins=edges)
    print(counts)
    centers = (edges[:-1] + edges[1:])/2
    radi = max(min(counts),1)
    nbins = 10
    ax.bar(centers, counts, width = 2*np.pi*radi/nbins*0.6, bottom=radi)
    ax = fig.add_subplot(122, polar=True)
    sliced = np.arange(nseq)
    nsliced = sliced.size
    ax.plot(flipped_orient[sliced], np.arange(nsliced)+5,'*', ms = 2.0)
    #orient = np.pi/4
    print(f'wave orientation: {orient*180/np.pi}')
    virtual_LGN = 0
    
    
    # In[33]:
    
    
    # generate retinal waves 
    stimulus_fn = video_fn + '.bin'
    
    TF = SF*speed
    dDis = (1-overlap)*2*wing/(np.pi*2)/SF[0]
    waveLength0 = np.sum(2*wing/(np.pi*2)/SF)
    burst_period = 2*wing/(np.pi*2)/SF[1] + dDis
    waveSF = 1.0/(waveLength0 + range_deg*2 - (waveLength0 - burst_period))
    waveTF = waveSF*speed
    dphase = dDis*waveSF*2*np.pi
    extend_finish = 2*range_deg*np.sqrt(2) - (1/waveSF - burst_period)
    fullDis = 1/waveSF + extend_finish
    time = np.zeros(nseq) + fullDis/speed
    phase[0] = -np.pi*2*(range_deg*np.sqrt(2)*waveSF) - np.pi
    phase[1] = phase[0] - dphase
    print(f'waveLength0 = {waveLength0} >= burst_period = {burst_period} >= dDis = {dDis}')
    print(f'TF = {TF}, SF = {SF}, speed = {speed} deg/s')
    print(f'waveLength = {1/waveSF}, waveTF = {waveTF} Hz')
    print(f'phase = {phase*180/np.pi} deg')
    print(f'full width of bar = {1/SF * wing/np.pi} degrees, time = {time}')
    print(f'frames to repeat = {1/waveSF/speed*frameRate:.1f}, frames for finish = {extend_finish/speed*frameRate:.1f}')
    
    generate_retinal_wave(amp, SF, TF, waveSF, waveTF, orient, phase, sharpness, resolution, video_fn, time, frameRate = frameRate, ecc = range_deg, gtype='drifting', neye = 1, bar = True, center = center, wing = wing, virtual_LGN = virtual_LGN, nrepeat = 1, reverse = reverse)
    
# setup stage III retinal waves 
if stage == 3:
    nseq = 40
    nrepeat = 3 # of each seq
    video_fn = f'{nseq}x{nrepeat}_retinal_wave_2'
    reverse = False
    resolution = 128
    seed = 3536653
    speed = 5.0 # degree/s
    range_deg = 5.0 # eccentricity from the original
    SF = np.array([1/8, 1/16])
    center = np.pi/2
    wing = np.pi/2
    amp = np.array([1.0,1.0])
    overlap = 0
    sharpness = np.array([-1,-1])
    phase = np.zeros(2) # initialize, make sure size is correct
    frameRate = 30
    
    np.random.seed(seed)
    #orient = np.array([np.pi*1/4, np.pi*1/2, np.pi*3/4, np.pi])
    #orient = np.array([np.pi*1/4, np.pi*3/4])
    #orient = np.random.rand(nseq)*2*np.pi
    orient = np.random.permutation(np.arange(nseq)/nseq*np.pi*2)
    # video file has x-axis flipped orientation
    flipped_orient = 2*np.pi - orient
    fig = plt.figure('orient dist.', figsize = (8,4), dpi =300)
    ax = fig.add_subplot(121, polar=True)
    edges = np.linspace(0, 2*np.pi, 30)
    print(edges)
    counts, _ = np.histogram(flipped_orient, bins=edges)
    print(counts)
    centers = (edges[:-1] + edges[1:])/2
    radi = max(min(counts),1)
    ax.bar(centers, counts, width = 2*np.pi*radi/nbins*0.6, bottom=radi)
    ax = fig.add_subplot(122, polar=True)
    sliced = np.arange(nseq)
    nsliced = sliced.size
    ax.plot(flipped_orient[sliced], np.arange(nsliced)+5,'*', ms = 2.0)
    #orient = np.pi/4
    print(f'wave orientation: {orient*180/np.pi}')
    virtual_LGN = 0
    
    # generate retinal waves 
    stimulus_fn = video_fn + '.bin'
    
    TF = SF*speed
    dDis = (1-overlap)*2*wing/(np.pi*2)/SF[0]
    waveLength0 = np.sum(2*wing/(np.pi*2)/SF)
    burst_period = 2*wing/(np.pi*2)/SF[1] + dDis
    waveSF = 1.0/(waveLength0 + range_deg*2 - (waveLength0 - burst_period))
    waveTF = waveSF*speed
    dphase = dDis*waveSF*2*np.pi
    extend_finish = 2*range_deg*np.sqrt(2) - (1/waveSF - burst_period)
    fullDis = 1/waveSF*nrepeat + extend_finish
    time = np.zeros(nseq) + fullDis/speed
    phase[0] = -np.pi*2*(range_deg*np.sqrt(2)*waveSF) - np.pi
    phase[1] = phase[0] - dphase
    print(f'waveLength0 = {waveLength0} >= burst_period = {burst_period} >= dDis = {dDis}')
    print(f'TF = {TF}, SF = {SF}, speed = {speed} deg/s')
    print(f'waveLength = {1/waveSF}, waveTF = {waveTF} Hz')
    print(f'phase = {phase*180/np.pi} deg')
    print(f'full width of bar = {1/SF * wing/np.pi} degrees, time = {time}')
    print(f'frames to repeat = {1/waveSF*nrepeat/speed*frameRate:.1f}, frames for finish = {extend_finish/speed*frameRate:.1f}')
    
    generate_retinal_wave(amp, SF, TF, waveSF, waveTF, orient, phase, sharpness, resolution, video_fn, time, frameRate = frameRate, ecc = range_deg, gtype='drifting', neye = 1, bar = True, center = center, wing = wing, virtual_LGN = virtual_LGN, nrepeat = nrepeat, reverse = reverse)
