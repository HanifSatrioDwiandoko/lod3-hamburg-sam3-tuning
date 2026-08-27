# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 22:01:49 2020

@author: wpc
"""

import math
import numpy as np
import matplotlib.pyplot as plt 
import copy
# import os
import joblib
# import skimage
from PIL import Image


def window_rect_complex(H,para_label): # label--> 'window','balcony','door'......
    p_component=[]
    num_floor=0
    h_sp=H
    
    eachfloor=para_label['eachfloor']
    for i in range(len(eachfloor)):
        num_floor+=eachfloor[str(i)]['floor']['sf']
        # print (num_floor)
        for nf in range(1,eachfloor[str(i)]['floor']['sf']+1):
            h_sp-=eachfloor[str(i)]['floor']['hf']
            h_gap=h_sp+eachfloor[str(i)]['h_gap']
            # print ('nf',nf,h_sp)
            w_sp=eachfloor[str(i)]['w_bound']
            nv=-1
            nm=0
            num_eachwindow=eachfloor[str(i)]['num_eachwindow']
            edge=eachfloor[str(i)]['edge']
            nwindow=eachfloor[str(i)]['nwindow']
            if eachfloor[str(i)]['num_vertices']>1:
                for n in range(nwindow):
                    if nm<n<num_eachwindow[str(nv+1)]['sw']+nm:
                        w_sp+=num_eachwindow[str(nv+1)]['ww']+edge['d_intra'][str(2*(nv+1))]
                    elif n==num_eachwindow[str(nv+1)]['sw']+nm:
                        nm+=num_eachwindow[str(nv+1)]['sw']
                        # print ('aaaaaaaaa',nm,edge['d_inter'],num_eachwindow)
                        w_sp+=edge['d_inter'][str(2*(nv+1)+1)]+num_eachwindow[str(nv+1)]['ww']
                        nv+=1
                    p_component.append(((w_sp,h_gap),num_eachwindow[str(nv+1)]['ww'],num_eachwindow[str(nv+1)]['hw']))
                    
            else:
                for n in range(nwindow):
                    w_sp=eachfloor[str(i)]['w_bound']+(num_eachwindow['0']['ww']+edge['d_intra'][str(2*(nv+1))])*n
                    p_component.append(((w_sp,h_gap),num_eachwindow['0']['ww'],num_eachwindow['0']['hw'])) 
                    
    return p_component

def transfd(dshape,z=0.5):
    verts=[]
    for window in dshape:
        # vert=[]
        pos,w,h=window
        x,y=pos
        vert=[(x,z,y),(x+w,z,y),(x+w,z,y+h),(x,z,y+h),(x,z,y)] #front
        vert2=[(x,z,y),(x+w,z,y),(x+w,0,y),(x,0,y),(x,z,y)] #bottom
        vert3=[(x,z,y),(x,0,y),(x,0,y+h),(x,z,y+h),(x,z,y)] #left
        vert4=[(x+w,z,y),(x+w,0,y),(x+w,0,y+h),(x+w,z,y+h),(x+w,z,y)] #right
        vert5=[(x,0,y),(x+w,0,y),(x+w,0,y+h),(x,0,y+h),(x,0,y)] # behind
        vert6= [(x,z,y+h),(x+w,z,y+h),(x+w,0,y+h),(x,0,y+h),(x,z,y+h)] # top
        # if lab=='facade':
        #     verts.append([vert2,vert3,vert4,vert5,vert6])
        # elif lab=='window' or lab=='balcony':
        verts.append([vert,vert2,vert3,vert4,vert5,vert6])
        # elif lab=='corrid':
        #     verts.append([vert,vert2,vert3,vert4,vert6])
    
    return verts

# def get_vertices(p_component):
#     for c in p_component:
        # verts=transfd(p_component,0.5,'window')

def rotator(vertex, ux, uy, origin_of_rotation):
    "Place a local (width, thickness, height) vertex into world space, offset by `vertex[0]` (local width-position) along the wall's real unit direction vector (ux, uy)."
    vertex = [float(vertex[0]), float(vertex[1]), float(vertex[2])]
    rotated = [None, None, vertex[2]+origin_of_rotation[2]]

    # vertex is (width-position along the wall, embrasure depth, height) as built
    # by transfd(). Only the first component is an offset ALONG the wall, so the
    # depth must not be mixed into it: the original sqrt(x^2 + depth^2) folded the
    # 0.5m depth into the along-wall direction, which does not recess the opening
    # at all and instead skews it sideways by up to 0.5m (worst at x=0, where the
    # opening and the hole cut in the wall ended up 0.5m apart). Openings are
    # placed flush with the wall plane, so the hole matches the opening exactly.
    l = vertex[0]
    rotated[0] = l*ux + origin_of_rotation[0]
    rotated[1] = l*uy + origin_of_rotation[1]

    return rotated



def wall_axis(modelpts):
    """
    The wall's true horizontal axis: origin (x,y), unit direction (ux,uy) and
    length, for mapping a facade layout's local width-coordinate into world space.

    The endpoints are the two points furthest apart in the XY plane -- i.e. the
    wall's real horizontal extent. They are then ordered low-radius -> high-radius
    (radius = sqrt(x^2+y^2)) so that local x=0 lands on the same corner that
    rect_im.order_points() puts at the LEFT edge of the rectified facade image
    (it assigns image-left from argmin(radius + z) / argmax(z - radius), i.e.
    left = low radius). Keeping the two consistent is what makes local x=0
    correspond to the start of the wall.

    The previous approach picked both endpoints with s = sqrt(x^2+y^2) + z. In a
    projected CRS (EPSG:25832 here) sqrt(x^2+y^2) is ~5.96e6 m and varies by only
    a couple of metres along one wall, while z varies by up to ~30m -- so s was
    dominated by height: argmin returned the LOWEST vertex and argmax the HIGHEST,
    which say nothing about where the wall starts or which way it runs. On walls
    whose lowest vertex is not the starting corner, the origin landed part-way
    along the wall and pushed the whole window grid off the far end (measured up
    to 13.6m past the wall on tile 6632). Where the two extremes were nearly
    coincident it also produced a near-zero, arbitrary direction vector.
    """
    xy = np.array([(p[0], p[1]) for p in modelpts], dtype=float)
    c = xy.mean(axis=0)
    i1 = int(np.argmax(((xy - c) ** 2).sum(axis=1)))
    i2 = int(np.argmax(((xy - xy[i1]) ** 2).sum(axis=1)))
    A, B = xy[i1], xy[i2]
    if np.hypot(A[0], A[1]) > np.hypot(B[0], B[1]):
        A, B = B, A
    L = float(np.linalg.norm(B - A))
    if L < 1e-6:
        return A, 1.0, 0.0, 0.0
    u = (B - A) / L
    return A, float(u[0]), float(u[1]), L


def get_rotation(modelpts, points_to_rotate):
    zs = [p[2] for p in modelpts]

    origin_xy, ux, uy, L = wall_axis(modelpts)
    x1, y1 = float(origin_xy[0]), float(origin_xy[1])
    z1 = min(zs)

    origin_coords = (x1,y1,z1)
    # print (origin_coords)
    new_w = []
    for point_to_rotate in points_to_rotate:
        new_f = []
        for v in point_to_rotate:
            new_v = []
            for p in v :
            # print (v)
                rotated_point = rotator(p, ux, uy, origin_coords)
                new_v.append(rotated_point)
            new_f.append(new_v)
        new_w.append(new_f)
    return new_w

def transpos(LinearRing):
    listPoints=[]
    lr=LinearRing.split()
    assert(len(lr) % 3 == 0)
    for i in range(0, len(lr), 3):
        listPoints.append((float(lr[i]), float(lr[i+1]), float(lr[i+2])))
    return listPoints

def wall_height_profile(modelpts):
    """
    Build a piecewise real roof-height-vs-distance-along-wall profile from a
    wall's true 3D corner points, using the same 2-corner rotation axis
    get_rotation() uses. Most walls are a simple uniform rectangle and this
    profile is flat; on a wall where two building wings of different height
    are stored as one WallSurface (a stepped roofline), the profile captures
    that the true roof is lower over part of the wall's horizontal span.
    Returns None if the wall is too small/degenerate to have a meaningful axis.
    """
    zs = [p[2] for p in modelpts]
    zmin = min(zs)

    # Same axis get_rotation() places the openings along, so the profile's
    # distance-along-wall coordinate matches the openings' local x.
    origin_xy, ux, uy, L = wall_axis(modelpts)
    if L < 0.5:
        return None
    x1, y1 = float(origin_xy[0]), float(origin_xy[1])

    # Rasterise every EDGE of the ring and keep the highest z seen at each
    # position along the wall -- i.e. the polygon's true upper envelope.
    #
    # Sampling only the vertices (as this did originally) is wrong whenever the
    # wall's BASE carries vertices at horizontal positions where its top edge
    # has none: the only sample at such a position is a bottom vertex, so the
    # profile reports the wall as nearly zero height there and every opening in
    # that stretch is clipped away. Walls with a stepped or subdivided base are
    # common here, and on tile 6632 this wrongly deleted 973 legitimate windows
    # (1033 openings clipped, of which only 60 were genuinely above the roof).
    # Interpolating along the edges samples the real top edge at every position
    # it spans, so the envelope is correct regardless of where vertices sit.
    proj = [(((x - x1) * ux + (y - y1) * uy), z) for x, y, z in modelpts]
    buckets = {}
    n = len(proj)
    for i in range(n):
        d0, z0 = proj[i]
        d1, z1 = proj[(i + 1) % n]
        steps = min(4000, max(2, int(abs(d1 - d0) / 0.05) + 1))
        for k in range(steps):
            t = k / (steps - 1.0)
            d = round((d0 + (d1 - d0) * t) / 0.05) * 0.05
            z = z0 + (z1 - z0) * t
            if d not in buckets or z > buckets[d]:
                buckets[d] = z

    profile = sorted(buckets.items())
    profile_xs = [p[0] for p in profile]
    profile_zs = [p[1] for p in profile]
    return profile_xs, profile_zs, zmin


def roof_height_at(profile, x):
    "Interpolated true wall height (above the wall base) at distance x along the wall."
    profile_xs, profile_zs, zmin = profile
    return np.interp(x, profile_xs, profile_zs) - zmin


def clip_to_roofline(p_component, profile, label, tolerance=0.3):
    """
    Drop any opening whose top edge would sit above the wall's true (possibly
    stepped) roofline at its horizontal position, instead of letting it overshoot
    past the real roof edge when the wall's overall H comes from its tallest point.
    """
    if profile is None:
        return p_component
    kept = []
    dropped = 0
    for (w_sp, h_gap), ww, hw in p_component:
        if h_gap + hw <= roof_height_at(profile, w_sp) + tolerance:
            kept.append(((w_sp, h_gap), ww, hw))
        else:
            dropped += 1
    if dropped:
        print('clip_to_roofline: dropped', dropped, 'of', len(p_component),
              'openings for label', label, '-- above the true stepped roofline')
    return kept


def get_coord(para, modelpts):

    W=para['W']
    H=para['H']
    para_set=para['para_set']
    modelinf={}
    # modelinf['W']=W
    # modelinf['H']=H
    profile = wall_height_profile(modelpts)
    for label in list(para_set.keys()):
        p_component=window_rect_complex(H,para_set[label])
        p_component=clip_to_roofline(p_component, profile, label)

        p_verts=transfd(p_component)
        if len(p_verts) > 0:
            rotated_p_component = get_rotation(modelpts, p_verts)
            modelinf[label]=rotated_p_component

        else:
            modelinf[label]= []

    return modelinf

def addfig(ax,c,a,verts):
    for vert in verts:
        # poly = mpl3.art3d.Poly3DCollection(vert,facecolors=np.random.choice(['r']), alpha=1)
        poly = mpl3.art3d.Poly3DCollection(vert[0:4],facecolors=c, alpha=a)
        ax.add_collection3d(poly)
        
        
if __name__ == '__main__':
    para_path = r'save_para'
    imgname = 'tex_2536667'
    para = joblib.load(para_path+'/'+imgname+'.'+'pkl')
    LinearRing = '2.5496955937E7 6672468.666 16.46 2.5496956309E7 6672460.868 16.46 2.5496956349E7 6672460.027 16.46 2.5496956355E7 6672459.907 16.46 2.5496956355E7 6672459.907 34.044 2.5496956349E7 6672460.027 33.98 2.5496956349E7 6672460.027 38.28 2.5496956309E7 6672460.868 38.719 2.5496955937E7 6672468.666 38.724 2.549695591E7 6672469.231 38.725 2.549695591E7 6672469.231 16.46 2.5496955937E7 6672468.666 16.46'
    modelpts = transpos(LinearRing)
    modelinf = get_coord(para, modelpts)
    
    import mpl_toolkits.mplot3d as mpl3
    fig = plt.figure()
    ax = mpl3.Axes3D(fig)
    
    addfig(ax,['b'],1,modelinf['window'])
    # addfig(ax,['purple'],1,vertsbal)
    # addfig(ax,['w'],0.6,vertsf1)
    # addfig(ax,['w'],0.6,vertsf2)
    # addfig(ax,['w'],0.6,vertsf3)
     
    # for vert in verts:
    #     poly = mpl3.art3d.Poly3DCollection(vert,facecolors=np.random.choice(['r']), alpha=1)
    #     ax.add_collection3d(poly)
        
    # for vert in vertsbal:
    #     poly = mpl3.art3d.Poly3DCollection(vert,facecolors=np.random.choice(['g']), alpha=0.6)
    #     ax.add_collection3d(poly)
        
    # for vert in vertsf:
    #     poly = mpl3.art3d.Poly3DCollection(vert,facecolors=np.random.choice(['y']), alpha=0.6)
    #     ax.add_collection3d(poly)
    x1,y1,z1 =(25496956.355, 6672459.907, 16.46)
    ax.set_xlim3d(left=x1-5,right=x1+20)
    ax.set_ylim3d(bottom=y1-5, top=y1+20)
    ax.set_zlim3d(bottom=z1-2,top=z1+12)
    # ax.set_aspect(1)
    plt.show()
    plt.close()




















