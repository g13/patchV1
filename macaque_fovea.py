import numpy as np
import matplotlib.pyplot as plt
from construct import *
from sys import getsizeof
def construct_macaque_fovea(physical_area, nblock, neuron_per_block, seed, use_sobol3d = False):
    pos = np.empty((3,nblock,neuron_per_block), dtype=float)
    print(f'taking up {getsizeof(pos)/1024/1024} Mb')
    # left ellipse
    aspect_ratio = 1.6
    left_radius = 1.0
    ly = left_radius * 0.8
    dx = aspect_ratio*np.sqrt(left_radius*left_radius - ly*ly)
    lx = left_radius*aspect_ratio - dx
    lx0 = left_radius*aspect_ratio
    ly0 = 0
    langle = ellipse_angle(left_radius,aspect_ratio,ly)
    lphase = np.pi - langle
    langle = 2*langle
    lseg_area = ellipse_lrseg_area(lphase, lphase+langle, left_radius, aspect_ratio)
   
    # right circle
    _rx = 0.8
    vol_aspect_ratio = 2.8
    rx = _rx + lx
    ryl = _rx * vol_aspect_ratio
    ry_ratio = 1.0 # asymmetric to y axis 
    right_radius = 3
    ry = [-ryl/(ry_ratio+1), ryl*ry_ratio/(ry_ratio+1)]
    rx0 = rx - np.sqrt(right_radius*right_radius - ryl*ryl/4)
    ry0 = ry[0] + ryl/2
    rphase = -np.arcsin(ryl/2/right_radius)
    rangle = -2*rphase
    rseg_area = seg_area(rangle, right_radius)
    
    left_top_point = np.array([lx, ly])
    right_top_point = np.array([rx, ry[1]])
    right_bottom_point = np.array([rx, ry[0]])
    left_bottom_point = np.array([lx, -ly])
    mid_area = poly_area([lx, rx, rx, lx], [ly, ry[1], ry[0], -ly], 4)
    
    area = lseg_area + rseg_area + mid_area
    area_per_block = area/nblock
    scale_ratio =  np.sqrt(physical_area/area)
    ### now rescale and plot
    ly = ly * scale_ratio
    left_radius = left_radius * scale_ratio
    dx = aspect_ratio*np.sqrt(left_radius*left_radius - ly*ly)
    lx = aspect_ratio*left_radius - dx
    lx0 = aspect_ratio*left_radius
    ly0 = ly0 * scale_ratio
    langle = ellipse_angle(left_radius,aspect_ratio,ly)
    lphase = np.pi - langle
    langle = 2*langle
    lseg_area = ellipse_lrseg_area(lphase, lphase+langle, left_radius, aspect_ratio)
    left_seg = CircularSegment(lx0, ly0, aspect_ratio, left_radius, lphase, langle, -1)
    lseg = left_seg.generator
    
    rx = _rx * scale_ratio
    ryl = rx * vol_aspect_ratio 
    rx = rx + lx
    right_radius = right_radius * scale_ratio
    ry = [-ryl/(ry_ratio+1), ryl*ry_ratio/(ry_ratio+1)]
    rx0 = rx - np.sqrt(right_radius*right_radius - ryl*ryl/4)
    ry0 = ry[0] + ryl/2
    rphase = -np.arcsin(ryl/2/right_radius)
    rangle = -2*rphase
    rseg_area = seg_area(rangle, right_radius)
    right_seg = CircularSegment(rx0, ry0, 1.0, right_radius, rphase, rangle, 1)
    rseg = right_seg.generator
    
    rslope = ry0/(rx-lx)
    
    left_top_point = np.array([lx, ly])
    right_top_point = np.array([rx, ry[1]])
    right_bottom_point = np.array([rx, ry[0]])
    left_bottom_point = np.array([lx, -ly])
    mid_area = poly_area([lx, rx, rx, lx], [ly, ry[1], ry[0], -ly], 4)
    left_top_line = LineSegment(left_top_point, right_top_point)
    left_bottom_line = LineSegment(left_bottom_point, right_bottom_point)
    top_line = left_top_line.generator
    tslope = 1/left_top_line.inverted_slope
    bottom_line = left_bottom_line.generator
    bslope = 1/left_bottom_line.inverted_slope
    
    l2 = left_radius*left_radius
    r2 = right_radius*right_radius
    lr_phase = -np.arcsin(ly/right_radius)
    lr_angle = -2*lr_phase
    lr_x0 = lx - np.sqrt(right_radius*right_radius - ly*ly)
    left_right_seg = CircularSegment(lr_x0, 0, 1.0, right_radius, lr_phase, lr_angle, 1)
    lr_seg = left_right_seg.generator
    
    area = lseg_area + rseg_area + mid_area
    print(f'area {area} = {lseg_area} + {rseg_area} + {mid_area}')
    area_per_block = area/nblock
    print('area per block =', area_per_block)
    
    fig = plt.figure('shape', dpi = 600)
    ax = fig.add_subplot(121)
    nLeftSeg = 100
    nRightSeg = 100
    nTopLine = 100
    nBottomLine = 100
    draw(nLeftSeg, lseg, ax, 0.1)
    draw(nRightSeg, rseg, ax, 0.1)
    ax.plot(left_bottom_point[0],left_bottom_point[1],'*r', ms = 0.3)
    ax.plot(left_top_point[0],left_top_point[1],'*b', ms = 0.3)
    ax.plot(right_bottom_point[0],right_bottom_point[1],'or', ms = 0.3)
    ax.plot(right_top_point[0],right_top_point[1],'ob', ms = 0.3)
    draw(nTopLine, top_line, ax, 0.1)
    draw(nBottomLine, bottom_line, ax, 0.1)
    ax.set_aspect('equal')
    #ax.set_xlim(2, 2.3)
    #ax.set_ylim(-3.2, -2.0)
    
    draw(nLeftSeg, lr_seg, ax, 0.1)
    mid_rseg_area = seg_area(lr_angle, right_radius)
    rarea = mid_area + rseg_area - mid_rseg_area
    larea = area - rarea
    
    r_nblock = rarea/area*nblock
    
    l_nblock = nblock - r_nblock
    
    nr_nblock = np.round(r_nblock).astype(int)
    nl_nblock = np.round(l_nblock).astype(int)
    cl = np.sqrt(area_per_block)
    print('characteristic_length, cl =', cl)
    print(f'left nblock: {nl_nblock}, right nblock: {nr_nblock}')
    if nl_nblock < 2:
        print('at least 2 blocks for left segment')
        return
    
    if np.mod(nl_nblock,2) == 1:
        print('make even nblock for the left segment')
        return
    
    ##########################################################
    #   construct the right area
    ##########################################################
    
    cos_rtheta = np.sqrt(1/(1+rslope*rslope))
    print(f'cos(theta) of {cos_rtheta}')
    #cos_cl = cl
    cos_cl = cl/cos_rtheta
    lc = np.linalg.norm([rx0-lr_x0, ry0])
    rring = lc/cos_cl
    nring = np.round(rring).astype(int)
    print('modulus of ring = ', rring-nring)
    cos_cl = lc/nring
    rring = lc/cos_cl
    nring = np.int(rring)
    print('corrected cl = ', cos_cl, 'nring =', nring)
    if rring-nring > 0:
        nring += 1
    redrawOnly = False
    if ~redrawOnly:
        seg_ring_area = np.empty(nring+1)
        poly_ring_area = np.empty(nring)
        ring_area = np.empty(nring)
        correction = np.empty(nring)
        cor = np.zeros(2)
        slice_angle = np.empty(nring+1)
        slice_phase = np.empty(nring+1)
        slangle = np.empty((2,nring))
        x0 = np.empty(nring+1)
        y0 = np.empty(nring+1)
        p0 = np.empty((2,nring+1))
        p1 = np.empty((2,nring+1))
        slice_phase[0] = lr_phase
        slice_angle[0] = lr_angle
        x0[0] = lr_x0
        y0[0] = 0
        seg_ring_area[0] = mid_rseg_area
        p0[:,0] = left_top_point
        p1[:,0] = left_bottom_point
        tolerance_sig = 1e-16
        iteration = 0
        biteration = 0
        ring_seg = np.empty(nring, dtype=object)
    blockCount = 0
    for i in range(nring):
        if ~redrawOnly:
            modulus_percent = 0.5
            correction[i] = 0
            dab = 0.01
            count = 0
            #while (count < 1):
            while (modulus_percent > tolerance_sig):
                if rring-nring > 0:
                    ratio = ((rring-nring)+i+correction[i])/rring
                else:
                    ratio = (1+i+correction[i])/rring
                    
                x0[i+1] = lr_x0 + (rx0-lr_x0)*ratio
                y0[i+1] = ry0*ratio
                a = 1+bslope*bslope
                b2 = -ly-bslope*lx
                b = 2*(bslope*(b2-y0[i+1])-x0[i+1])
                c = x0[i+1]*x0[i+1] - r2 + np.power(b2-y0[i+1],2)
                ltx0 = (-b+np.sqrt(b*b-4*a*c))/2/a
                lty0 = bslope*ltx0+b2
                yl = lty0-y0[i+1]
                slice_phase[i+1] = np.arcsin(yl/right_radius)
                p1[:,i+1] = [ltx0, lty0]
                
                a = 1+tslope*tslope
                b1 = ly-tslope*lx
                b = 2*(tslope*(b1-y0[i+1])-x0[i+1])
                c = x0[i+1]*x0[i+1] -r2  + np.power(b1-y0[i+1],2)
                ltx0 = (-b+np.sqrt(b*b-4*a*c))/2/a
                lty0 = tslope*ltx0+b1
                yl = lty0-y0[i+1]
                slice_angle[i+1] = np.arcsin(yl/right_radius) - slice_phase[i+1]
                p0[:,i+1] = [ltx0, lty0]
                
                seg_ring_area[i+1] = seg_area(slice_angle[i+1], right_radius)        
    
                poly_ring_area[i] = poly_area([p0[0,i], p0[0,i+1], p1[0,i+1], p1[0,i]], \
                                           [p0[1,i], p0[1,i+1], p1[1,i+1], p1[1,i]],4)
                ring_area[i] = seg_ring_area[i+1] + poly_ring_area[i] - seg_ring_area[i]
                current = ring_area[i]/area_per_block
                if count == 0:
                    target = np.round(current)
                modulus_percent = (ring_area[i]%area_per_block)/area_per_block
                if modulus_percent>0.5:
                    modulus_percent = 1-modulus_percent
                if count > 0:
                    if current == old:
                        break
                    dab = (target - current)/(current - old)*old_dab
                correction[i] += dab
                old_dab = dab
                old = current
                count += 1
                #print(old, '->', old_dab, '->', current, '->', dab, '->', target)
            iteration += count
            #print(count, 'iterations for slice')
    
        n = (i+1)*nLeftSeg
        ring_seg[i] = CircularSegment(x0[i+1], y0[i+1], 1.0, right_radius, slice_phase[i+1], slice_angle[i+1], 1)
        ax.plot(p0[0,i+1], p0[1,i+1], '*', ms=0.1)
        ax.plot(p1[0,i+1], p1[1,i+1], '*', ms=0.1)
        draw(n, ring_seg[i].generator, ax, 0.1)
        nslice_block = np.round(ring_area[i]/area_per_block).astype(int)
        x = np.empty((2,nslice_block-1))
        y = np.empty((2,nslice_block-1))
        if ~redrawOnly:
            for j in range(2):
                if j == 0:
                    pp = p0
                    k = 0
                else:
                    pp = p1
                    k = nslice_block-2
                percent = 0.5
                dab = 0.01
                count = 0
                slangle0 = slice_angle[i]/nslice_block
                target = area_per_block
                while (percent > tolerance_sig):
                    slangle[j, i] = slangle0 * (1+cor[j])
                    x[0,k] = np.cos(slice_angle[i]/2 - slangle[j, i]) * right_radius
                    y[0,k] = np.sqrt(r2 - x[0,k]*x[0,k])
                    x[0,k] = x0[i] + x[0,k]
                    y[0,k] = y0[i] + (1-2*j)*y[0,k]
                    x[1,k] = x[0,k] + x0[i+1]-x0[i]
                    y[1,k] = y[0,k] + y0[i+1]-y0[i]
                    left_seg_block = seg_area(slangle[j, i], right_radius)
                    mid_sec_block = poly_area([pp[0,i], pp[0,i+1], x[1,k], x[0,k]], \
                                                [pp[1,i], pp[1,i+1], y[1,k], y[0,k]],4)
                    srangle = 2*np.arcsin(np.linalg.norm(pp[:,i+1]-[x[1,k], y[1,k]])/2/right_radius)
                    right_seg_block = seg_area(srangle, right_radius)
                    current = mid_sec_block+right_seg_block-left_seg_block
                    percent = np.abs(current-target)/target
                    if count > 0:
                        if current == old:
                            break
                        dab = (target - current)/(current - old)*old_dab
                    cor[j] += dab
                    old_dab = dab
                    old = current
                    count += 1
                    #print(old, '->', old_dab, '->', current, '->', dab, '->', target)
                biteration += count
                #ax.plot(x[0],y[0],'*', ms = 0.1)
                #ax.plot(x[1],y[1],'o', ms = 0.1)
                
        
        slangle0 = (slice_angle[i]-slangle[0,i]-slangle[1,i])/(nslice_block-2)
        for j in range(nslice_block-3):
            this_angle = slice_angle[i]/2 - slangle[0,i] - (j+1)*slangle0
            sign = np.sign(this_angle)
            x[0,j+1] = np.cos(this_angle) * right_radius
            y[0,j+1] = np.sqrt(r2 - x[0,j+1]*x[0,j+1])
            x[0,j+1] = x0[i] + x[0,j+1]
            y[0,j+1] = y0[i] + sign*y[0,j+1]
            x[1,j+1] = x[0,j+1] + x0[i+1]-x0[i]
            y[1,j+1] = y[0,j+1] + y0[i+1]-y0[i]
        #print(x,y)
        #print(nslice_block)
        ax.plot(x, y, lw = 0.1)
        
        
        #for j in [0, nslice_block//2, nslice_block-1]:
        for j in range(nslice_block):
            if j==0:
                this_phase = slice_angle[i]/2 - slangle[0,i]
                this_angle = slangle[0,i]
            else:
                if j==nslice_block-1:
                    this_phase = slice_phase[i]
                    this_angle = slangle[1,i]
                else:
                    this_phase = slice_angle[i]/2 - slangle[0,i] - j*slangle0
                    this_angle = slangle0
            if i==0:
                left_curve = left_right_seg.copy(this_phase, this_angle)
            else:
                left_curve = ring_seg[i-1].copy(this_phase, this_angle)
            if i==nring-1:
                right_curve = right_seg.copy(this_phase, this_angle)
            else:
                right_curve = ring_seg[i].copy(this_phase, this_angle)
            if j==0:
                top_curve = LineSegment(p0[:,i], p0[:,i+1])
            else:
                top_curve = LineSegment(np.array([x[0,j-1],y[0,j-1]]), np.array([x[1,j-1], y[1,j-1]]))
                
            if j==nslice_block-1:
                bottom_curve = LineSegment(p1[:,i], p1[:,i+1])
                lcurve = [bottom_curve, left_curve, top_curve]
                rcurve = [right_curve]
            else:
                bottom_curve = LineSegment(np.array([x[0,j],y[0,j]]), np.array([x[1,j], y[1,j]]))
                lcurve = [left_curve, top_curve]
                rcurve = [bottom_curve, right_curve]
                
            if j!=0 and j!=nslice_block-1:
                lefty = np.array([y[0,j], y[0,j-1], y[1,j-1]])
                righty = np.array([y[0,j], y[1,j], y[1,j-1]])
            else:
                if j==0:
                    lefty = np.array([y[0,j], p0[1,i], p0[1,i+1]]) 
                    righty = np.array([y[0,j], y[1,j], p0[1,i+1]])
                else:
                    lefty = np.array([p1[1,i+1], p1[1,i], y[0, j-1], y[1,j-1]]) 
                    righty = np.array([p1[1,i+1], y[1,j-1]])
                    
            if use_sobol3d:
                temp_pos = generate_pos_3d(lcurve, rcurve, area_per_block, lefty, righty, neuron_per_block, seed+blockCount)
                pos[:, blockCount,:] = temp_pos
            else:
                temp_pos = generate_pos_2d(lcurve, rcurve, area_per_block, lefty, righty, neuron_per_block, seed+blockCount)
                pos[:2, blockCount,:] = temp_pos
            ax.plot(temp_pos[0,:], temp_pos[1,:], ',', ms = 0.3)
            blockCount += 1
    
    print('avg iterations =', iteration/nring)
    print('avg biterations =', biteration/nring)
    print(f'{blockCount} blocks constructed')
    #ax.set_ylim(0.5,1)
    #ax.set_xlim(0,0.5)
    
    ###########################################################
    #    construct the left area
    #fig = plt.figure('shape', dpi = 600)
    #ax = fig.add_subplot(121)
    #draw(nLeftSeg, lseg, ax, 0.1)
    #draw(nLeftSeg, lr_seg, ax, 0.1)
    #ax.set_aspect('equal')
    ###########################################################
    
    rstripe = ly/cl
    nstripe = np.round(rstripe).astype(int)+1
    print('upper estimate', nstripe, 'stripes')
    stripe_area = np.empty(nstripe)
    nline = nstripe - 1
    pll = np.empty((2, nline*2 + 1)) # point on the left contour of the left seg
    plr = np.empty((2, nline*2 + 1)) # point on the right contour of the left seg
    la = np.empty(nstripe)
    ra = np.empty(nstripe)
    lcor = np.empty(nstripe)
    literation = 0
    i = 0
    sangle = langle
    qangle = lr_angle
    lefted = larea/2
    while (np.sum(stripe_area[:i])/area_per_block < nl_nblock/2 and lefted > 0):
        percent = 0.5
        dab = 0.01
        count = 0
        tolerance_sig = 1e-16
        lcor[i] = 0
        sangle = langle - np.sum(la[:i])*2
        qangle = lr_angle - np.sum(ra[:i])*2
        #print(i, sangle*180/np.pi, qangle*180/np.pi)
        dangle1 = 1/nstripe*langle/2
        dangle1 = (sangle- dangle1)/2 
        dangle2 = 1/nstripe*lr_angle/2
        dangle2 = (qangle- dangle2)/2
        dl = cl/np.cos((dangle1+dangle2)/2)
        #print(np.cos((dangle1+dangle2)/2), dl)
        dangle1 = 2*np.arcsin(dl/2/left_radius)
        dangle2 = dangle1/langle * lr_angle
        while (percent > tolerance_sig):
            la[i] = (1+lcor[i])*dangle1
            angle1 = lphase+np.sum(la[:i])
            angle2 = angle1+la[i]
            r = polar(left_radius, aspect_ratio, angle2)
            pll[1,i] = np.sin(angle2)*r
            pll[0,i] = np.cos(angle2)*r + lx0
            #print('i=',i,'count=',count)
            left_seg_block = ellipse_lrseg_area(angle1, angle2, left_radius, aspect_ratio)
    
            ra[i] = (1+lcor[i])*dangle2
            plr[1,i] = np.sin(lr_angle/2-np.sum(ra[:i+1]))*right_radius
            plr[0,i] = np.sqrt(r2 - plr[1,i]*plr[1,i]) + lr_x0
            right_seg_block = seg_area(ra[i], right_radius)
            if i==0:
                mid_sec_block = poly_area([pll[0,i], plr[0,i], left_top_point[0]], \
                                        [pll[1,i], plr[1,i], left_top_point[1]],3) # the top mid sec is a triangle
            else:
                mid_sec_block = poly_area([pll[0,i], plr[0,i], plr[0,i-1], pll[0,i-1]], \
                                        [pll[1,i], plr[1,i], plr[1,i-1], pll[1,i-1]],4)
                
            stripe_area[i] = mid_sec_block+right_seg_block+left_seg_block
            current = stripe_area[i]/area_per_block
            if count == 0:
                target = np.round(current)
                if target < 1:
                    target = 1
                this_area = target*area_per_block
                lefted = larea/2 - np.sum(stripe_area[:i])
                if this_area > lefted or (this_area > 1.5*(lefted-this_area)):
                    target = lefted/area_per_block
                    print(f'stripe #{i} forced to contain {target} blocks')
                    lefted = 0
                    break
            
            percent = np.abs((current-target)/target)
            if count > 0:
                if current == old:
                    break
                dab = (target - current)/(current - old)*old_dab
            lcor[i] += dab
            old_dab = dab
            old = current
            count += 1
            #print(nline,'-',i,'-',sum(lcor[:i]),lcor[i])
            #print('#', i, count,':', old, '->', current ,'->', target, ':', old_dab, '->', dab)
            #ax.plot([pll[0,i], plr[0,i]], [pll[1,i], plr[1,i]], ls=':', lw = 0.1*count)    
        literation += count
        #if (np.sum(la[:i+1]) > langle/2):
        #    nline = i+1
        #    print('im here')
        #    break
        if lefted > 0:
            ax.plot([pll[0,i], plr[0,i]], [pll[1,i], plr[1,i]], lw = 0.1)    
        #print(f'stripe area #{i} = {mid_sec_block}+{right_seg_block}+{left_seg_block}, larea = {larea}') 
        #print(i, stripe_area[i]/area_per_block)
        i+=1
    #print(lcor)
    nstripe = i
    nline = nstripe - 1
    print(f'{nline} lines top, {sum(stripe_area[:nstripe]/area_per_block)} blocks, total {nl_nblock} blocks')
    pll[0, nline] = 0
    pll[1, nline] = 0
    plr[0, nline] = right_radius + lr_x0
    plr[1, nline] = 0
    ax.plot([pll[0,nline], plr[0,nline]], [pll[1,nline], plr[1,nline]], lw = 0.1)
    for i in range(1, nline+1):
        pll[0, nline+i] = pll[0, nline-i]
        pll[1, nline+i] = -pll[1, nline-i]
        plr[0, nline+i] = plr[0, nline-i]
        plr[1, nline+i] = -plr[1, nline-i]
        ax.plot([pll[0,nline+i], plr[0,nline+i]], [pll[1,nline+i], plr[1,nline+i]], lw = 0.1)
    ####################
        
    def get_midp(p0, p1, n):
        midp = np.empty((2, n-1))
        r = np.arange(1,n)/n
        midp[0,:] = p0[0] + (p1[0]-p0[0])*r
        midp[1,:] = p0[1] + (p1[1]-p0[1])*r
        return midp
    def get_startp(p0, p1, n):
        startp = p0 + (p1-p0)/n
        return startp
    
    ########### BUILD block within stripes ############
    stripe_block = np.empty(nstripe, dtype=int)
    stripe_block[0] = np.round(stripe_area[0]/area_per_block).astype(int)
    pp = np.empty((nline,2), dtype=object)
    for i in range(1, nstripe):
        stripe_block[i] = np.round(stripe_area[i]/area_per_block).astype(int)
        startp_p = get_startp(pll[:,i-1], plr[:,i-1], stripe_block[i])
        startp = get_startp(pll[:,i], plr[:,i], stripe_block[i])
        midp_p = np.empty((2,stripe_block[i]-1))
        midp = np.empty((2,stripe_block[i]-1))
        percent = 0.5
        count = 0
        pcor = 0
        dab = 0.5
        angle1 = lphase+np.sum(la[:i])
        angle2 = angle1 + la[i]
        block_lseg_area = ellipse_lrseg_area(angle1, angle2, left_radius, aspect_ratio)
        assert(block_lseg_area > 0)
        target = area_per_block
        
        while (percent > tolerance_sig):
            leftp = pll[:,i] + (startp - pll[:,i]) * (1+pcor)
            leftp_p = pll[:,i-1] + (startp_p - pll[:,i-1]) * (1+pcor)
            block_sec_area = poly_area([leftp[0], leftp_p[0], pll[0,i-1], pll[0,i]], [leftp[1], leftp_p[1], pll[1,i-1], pll[1,i]], 4)
            block_area = block_sec_area + block_lseg_area
            current = block_area
            assert(current > 0)
            percent = np.abs((current-target)/target)
            if count > 0:
                if current == old:
                    break
                dab = (target - current)/(current - old)*old_dab
            pcor += dab
            old_dab = dab
            old = current
            count += 1
        literation += count
            
        if stripe_block[i] > 1:
            midp[:,0] = leftp
            midp_p[:,0] = leftp_p
            
        if stripe_block[i] >2:
            percent = 0.5
            count = 0
            pcor = 0
            dab = 0.01
            block_rseg_area = seg_area(ra[i], right_radius)
            target = area_per_block
            startp = get_startp(plr[:,i], pll[:,i], stripe_block[i])
            startp_p = get_startp(plr[:,i-1], pll[:,i-1], stripe_block[i])
            while (percent > tolerance_sig):
                rightp = plr[:,i] + (startp - plr[:,i]) * (1+pcor)
                rightp_p = plr[:,i-1] + (startp_p - plr[:,i-1]) * (1+pcor)
                block_sec_area = poly_area([rightp[0], rightp_p[0], plr[0,i-1], plr[0,i]], [rightp[1], rightp_p[1], plr[1,i-1], plr[1,i]], 4)
                block_area = block_sec_area + block_rseg_area
                current = block_area
                percent = np.abs((current-target)/target)
                if count > 0:
                    if current == old:
                        break
                    dab = (target - current)/(current - old)*dab
                pcor += dab
                old_dab = dab
                old = current
                count += 1
            literation += count
            midp[:,-1] = rightp
            midp_p[:,-1] = rightp_p
            midp[:,1:-1] = get_midp(leftp, rightp, stripe_block[i]-2)
            midp_p[:,1:-1] = get_midp(leftp_p, rightp_p, stripe_block[i]-2)
        pp[i-1,0] = np.hstack((np.reshape(pll[:,i-1],(2,1)), midp_p, np.reshape(plr[:,i-1],(2,1))))
        pp[i-1,1] = np.hstack((np.reshape(pll[:,i],(2,1)), midp, np.reshape(plr[:,i],(2,1))))
        for j in range(stripe_block[i]-1): 
            # top part
            ax.plot([midp[0,j], midp_p[0,j]], [midp[1,j], midp_p[1,j]], lw = 0.1)
            # bottom part
            ax.plot([midp[0,j], midp_p[0,j]], [-midp[1,j], -midp_p[1,j]], lw = 0.1)
    print("individual blocks constructed")
    # rain rands
    for i in range(nstripe):
        for j in range(stripe_block[i]):
            #print(i,j)
            if i > 0:
                if j == 0:
                    left_curve = left_seg.copy(np.pi-langle/2+np.sum(la[:i]), la[i])       
                else:
                    left_curve = LineSegment(pp[i-1,1][:,j], pp[i-1,0][:,j])
                    
                if j == stripe_block[i]-1:
                    right_curve = left_right_seg.copy(lr_angle/2-np.sum(ra[:i+1]), ra[i])
                else:
                    right_curve = LineSegment(pp[i-1,1][:,j+1], pp[i-1,0][:,j+1])
                    
                bottom_curve = LineSegment(pp[i-1,1][:,j+1], pp[i-1,1][:,j])
                top_curve =    LineSegment(pp[i-1,0][:,j+1], pp[i-1,0][:,j])
                lefty = np.array([pp[i-1,1][1,j+1], pp[i-1,1][1,j], pp[i-1,0][1,j]])
                righty = np.array([pp[i-1,1][1,j+1], pp[i-1,0][1,j+1],  pp[i-1,0][1,j]])
                                  
                lcurve = [bottom_curve, left_curve]
                rcurve = [right_curve, top_curve]
            else:
                bottom_curve = LineSegment(plr[:, i], pll[:, i])
                left_curve = left_seg.copy(np.pi-langle/2, la[0])
                right_curve = left_right_seg.copy(lr_angle/2-ra[0], ra[0])
                lcurve = [bottom_curve, left_curve]
                rcurve = [right_curve]
                lefty = np.array([plr[1,i], pll[1,i], left_top_point[1]])
                righty = np.array([plr[1,i], left_top_point[1]])
                
            if use_sobol3d:
                temp_pos = generate_pos_3d(lcurve, rcurve, area_per_block, lefty, righty, neuron_per_block, seed+blockCount)
                pos[:, blockCount,:] = temp_pos
            else:
                temp_pos = generate_pos_2d(lcurve, rcurve, area_per_block, lefty, righty, neuron_per_block, seed+blockCount)
                pos[:2, blockCount,:] = temp_pos
            ax.plot(temp_pos[0,:], temp_pos[1,:], ',', ms = 0.3)
            blockCount += 1  
    
    top_is_bottom = np.array([1,-1])
    for i in range(nstripe):
        for j in range(stripe_block[i]):
            #print(i,j)
            if i > 0:
                if j == 0:
                    left_curve = left_seg.copy(np.pi+langle/2-np.sum(la[:i+1]), la[i])       
                else:
                    left_curve = LineSegment(pp[i-1,1][:,j]*top_is_bottom, pp[i-1,0][:,j]*top_is_bottom)
                    
                if j == stripe_block[i]-1:
                    right_curve = left_right_seg.copy(-lr_angle/2+np.sum(ra[:i]), ra[i])
                else:
                    right_curve = LineSegment(pp[i-1,1][:,j+1]*top_is_bottom, pp[i-1,0][:,j+1]*top_is_bottom)
                    
                bottom_curve = LineSegment(pp[i-1,1][:,j+1]*top_is_bottom, pp[i-1,1][:,j]*top_is_bottom)
                top_curve =    LineSegment(pp[i-1,0][:,j+1]*top_is_bottom, pp[i-1,0][:,j]*top_is_bottom)
                lefty = np.array([-pp[i-1,0][1,j], -pp[i-1,1][1,j], -pp[i-1,1][1,j+1]])
                righty = np.array([-pp[i-1,0][1,j], -pp[i-1,0][1,j+1], -pp[i-1,1][1,j+1]])
                                  
                lcurve = [left_curve, bottom_curve]
                rcurve = [top_curve, right_curve]
            else:
                reversed_i = 2*nstripe-2
                bottom_curve = LineSegment(plr[:, reversed_i], pll[:, reversed_i])
                left_curve = left_seg.copy(np.pi+langle/2-la[0], la[0])
                right_curve = left_right_seg.copy(-lr_angle/2, ra[0])
                lcurve = [left_curve, bottom_curve]
                rcurve = [right_curve]
                lefty = np.array([left_bottom_point[1], pll[1, reversed_i], plr[1, reversed_i]])
                righty = np.array([left_bottom_point[1], plr[1, reversed_i]])
            if use_sobol3d:
                temp_pos = generate_pos_3d(lcurve, rcurve, area_per_block, lefty, righty, neuron_per_block, seed+blockCount)
                pos[:, blockCount,:] = temp_pos
            else:
                temp_pos = generate_pos_2d(lcurve, rcurve, area_per_block, lefty, righty, neuron_per_block, seed+blockCount)
                pos[:2, blockCount,:] = temp_pos
            ax.plot(temp_pos[0,:], temp_pos[1,:], ',', ms = 0.3)
            blockCount += 1  
            
    pll[0, nline*2] = pll[0, 0]
    pll[1, nline*2] = -pll[1, 0]
    plr[0, nline*2] = plr[0, 0]
    plr[1, nline*2] = -plr[1, 0]
    ax.plot(pll[0,:nline*2+1],pll[1,:nline*2+1],'*', ms = 0.1)
    ax.plot(plr[0,:nline*2+1],plr[1,:nline*2+1],'o', ms = 0.1)
    ax.plot([pll[0,2*nline], plr[0,2*nline]], [pll[1,2*nline], plr[1,2*nline]], lw = 0.1)
    #ax.set_xlim(0,0.5)
    #ax.set_ylim(-1,1)
    if not use_sobol3d:
        pos[2, :, :] = np.reshape(rand.uniform(nblock*neuron_per_block),(1,nblock,neuron_per_block))*0.01
        
    print('avg iterations', literation/(nline+2))
    print(f'{blockCount} constructed')
    ax.set_xlabel('mm')
    ax.set_ylabel('mm')
    ax.set_title('2D physical layout of macaque V1 neurons')
    return pos, fig
