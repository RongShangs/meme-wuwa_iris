"""A shared waveform endpoint, morphed into the actual first/last vertical pupils."""
from pathlib import Path
from functools import lru_cache
import cv2
import numpy as np
from render import image, over_rgb, warp_rgba_premultiplied


def smooth(t):
    t=np.clip(t,0.,1.)
    return t*t*(3-2*t)


def palette(iris, last):
    def sample(a):
        h,w=a.shape[:2];y,x=np.mgrid[:h,:w]
        radius=((x-w/2)/(w/2))**2+((y-h/2)/(h/2))**2
        selection=(a[:,:,3]>200)&(radius>.16)&(radius<.65)&(y<h*.48)
        return np.median(a[selection,:3],axis=0)
    return sample(iris),sample(last)


def gradient(spec, dimensions):
    w,h=dimensions
    if spec.get('background_bgr') is not None:
        return np.full((h,w,3),spec['background_bgr'],np.uint8)
    first=image(Path(spec['pupil_reference']))
    last=image(Path(spec['last_iris_reference']))
    a,b=palette(first,last)
    y,x=np.mgrid[:h,:w].astype(np.float32)
    t=np.clip(.65*y/max(1,h-1)+.35*x/max(1,w-1),0,1)[:,:,None]
    return np.rint(a*(1-t)+b*t).astype(np.uint8)


def crop_mask(mask):
    ys,xs=np.where(mask>64)
    if not len(xs):raise ValueError('Empty transition mask')
    return mask[ys.min():ys.max()+1,xs.min():xs.max()+1]


def pupil_mask(iris):
    h,w=iris.shape[:2];y,x=np.mgrid[:h,:w]
    candidate=((iris[:,:,:3].max(axis=2)<100)&(iris[:,:,3]>200)
               &(x>w*.40)&(x<w*.60)&(y>h*.13)&(y<h*.87)).astype(np.uint8)
    _,labels,stats,centers=cv2.connectedComponentsWithStats(candidate)
    options=[i for i in range(1,len(stats)) if stats[i,4]>3]
    if not options:raise ValueError('No vertical pupil in endpoint iris')
    i=min(options,key=lambda i:np.linalg.norm((centers[i]-[w/2,h/2])/[w,h]))
    mask=(labels==i).astype(np.uint8)*255
    return mask, np.median(iris[labels==i,:3],axis=0)


def layered_iris_textures(iris):
    """Three flat purple regions following Phoebe's actual iris teeth."""
    pupil,_=pupil_mask(iris)
    clean=iris.copy()
    clean[:,:,:3]=cv2.inpaint(iris[:,:,:3],cv2.dilate(pupil,np.ones((3,3),np.uint8)),4,cv2.INPAINT_TELEA)
    colors=np.float32([[136,51,109],[182,83,146],[238,149,196]])
    pixels=clean[:,:,:3].astype(np.float32)
    nearest=np.argmin(np.sum((pixels[:,:,None,:]-colors[None,None,:,:])**2,axis=3),axis=2)
    flat=clean.copy();flat[:,:,:3]=colors[nearest].astype(np.uint8)
    return flat,clean


@lru_cache(maxsize=4)
def original_tooth_layers(path):
    """Extract nested flat-color masks from the actual high-resolution iris."""
    a=image(Path(path));ys,xs=np.where(a[:,:,3]>128)
    cx=(xs.min()+xs.max())/2;cy=(ys.min()+ys.max())/2
    extent=max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)+4
    size=1024;sc=size/extent
    iris=warp_rgba_premultiplied(a,np.float32([[sc,0,size/2-cx*sc],[0,sc,size/2-cy*sc]]),(size,size))
    pupil,_=pupil_mask(iris)
    clean=cv2.inpaint(iris[:,:,:3],cv2.dilate(pupil,np.ones((5,5),np.uint8)),8,cv2.INPAINT_TELEA)
    colors=np.float32([[136,51,109],[182,83,146],[238,149,196]])
    nearest=np.argmin(np.sum((clean.astype(np.float32)[:,:,None,:]-colors[None,None,:,:])**2,axis=3),axis=2)
    masks=[]
    for selection in (nearest>=1,nearest==2):
        mask=(selection&(iris[:,:,3]>100)).astype(np.uint8)*255
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
        # Remove texture-induced threshold speckles without changing the much
        # larger tooth contours; rethreshold so the interior stays flat color.
        mask=(cv2.GaussianBlur(mask,(0,0),14)>=128).astype(np.uint8)*255
        # Continue the lower color field beyond the source disc: the teeth,
        # rather than the source's circular crop, define the opening background.
        for x in range(size):
            ys=np.flatnonzero(mask[:,x])
            if len(ys):mask[ys[-1]:,x]=255
        masks.append(cv2.GaussianBlur(mask,(0,0),1.2))
    return tuple(masks)


def pattern_frame(spec,dimensions,t,diameter):
    w,h=dimensions;initial=np.hypot(w,h)*1.8
    frame=np.full((h,w,3),(136,51,109),np.uint8)
    masks=original_tooth_layers(spec['pattern_reference'])
    for mask,color,ratio,dy,duration in zip(masks,[(182,83,146),(238,149,196)],[.50,.80],[-220,320],[.68,.86]):
        converge=float(smooth(t/duration))
        layer_diameter=diameter*(ratio+(1-ratio)*converge)
        sc=layer_diameter/mask.shape[0]
        m=np.float32([[sc,0,w/2-mask.shape[1]/2*sc],[0,sc,h/2-mask.shape[0]/2*sc+dy*(1-converge)]])
        # Extend the already-filled lower edge beyond the mask canvas.  A
        # transparent warp border otherwise cuts the middle field off while
        # its independently scaled tooth pattern is still above the frame.
        alpha=cv2.warpAffine(mask,m,(w,h),flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE).astype(np.float32)[:,:,None]/255
        frame=np.rint(frame*(1-alpha)+np.asarray(color)*alpha).astype(np.uint8)
    # The outline of the iris is outside the frame at the beginning and enters
    # as the whole pattern contracts.
    iris=image(Path(spec['pupil_reference']));size=int(spec.get('final_iris_size',iris.shape[0]))
    iris=cv2.resize(iris,(size,size),interpolation=cv2.INTER_AREA)
    sc=diameter/size;m=np.float32([[sc,0,w/2-size/2*sc],[0,sc,h/2-size/2*sc]])
    outer=cv2.warpAffine(iris[:,:,3],m,(w,h),flags=cv2.INTER_LINEAR).astype(np.float32)/255
    outer=np.where(outer>.94,1,outer)
    yy,xx=np.ogrid[:h,:w];rad=np.sqrt(((xx-w/2)/(diameter/2))**2+((yy-h/2)/(diameter/2))**2)
    rim=np.clip((rad-.93)/.025,0,1)*outer
    frame=np.rint(frame*(1-rim[:,:,None])+np.asarray([80,16,58])*rim[:,:,None]).astype(np.uint8)
    frame=np.rint(frame*outer[:,:,None]+250*(1-outer[:,:,None])).astype(np.uint8)
    return frame


def resonance_background(spec,dimensions,t=0):
    if not spec.get('layered_iris_background'):
        return gradient(spec,dimensions)
    return pattern_frame(spec,dimensions,t,np.hypot(*dimensions)*1.8)


def normalized(mask):
    crop=crop_mask(mask);h,w=crop.shape
    scale=min(900/h,460/w);nw,nh=round(w*scale),round(h*scale)
    result=np.zeros((1024,512),np.uint8)
    x,y=(512-nw)//2,(1024-nh)//2
    result[y:y+nh,x:x+nw]=cv2.resize(crop,(nw,nh),interpolation=cv2.INTER_CUBIC)
    return result


def signed(mask):
    binary=(mask>127).astype(np.uint8)
    return cv2.distanceTransform(binary,cv2.DIST_L2,5)-cv2.distanceTransform(1-binary,cv2.DIST_L2,5)


class Morph:
    def __init__(self,spec,iris):
        self.mark=normalized(image(Path(spec['mark_path']))[:,:,3])
        self.pupil,self.color=pupil_mask(iris)
        self.target=normalized(self.pupil)
        self.source_distance=signed(self.mark);self.target_distance=signed(self.target)

    def layer(self,amount,height,dimensions):
        w,h=dimensions
        if amount<=0:mask=self.mark
        elif amount>=1:mask=self.target
        else:
            distance=(1-amount)*self.source_distance+amount*self.target_distance
            mask=(np.clip(distance+.5,0,1)*255).astype(np.uint8)
        cel=np.zeros((1024,512,4),np.uint8)
        cel[:,:,:3]=self.color*amount;cel[:,:,3]=mask
        sc=height/900
        m=np.float32([[sc,0,w/2-255.5*sc],[0,sc,h/2-511.5*sc]])
        return warp_rgba_premultiplied(cel,m,(w,h))


def endpoint(spec, dimensions):
    frame=resonance_background(spec,dimensions)
    first=image(Path(spec['pupil_reference']))
    over_rgb(frame,Morph(spec,first).layer(0,float(spec.get('mark_height',1050)),dimensions),0,0)
    return frame


def make_opening(spec,iris,dimensions):
    w,h=dimensions;count=int(spec['frames'])
    bg=gradient(spec,dimensions);morph=Morph(spec,iris)
    pupil_height=crop_mask(morph.pupil).shape[0]
    final=float(spec.get('final_iris_size',iris.shape[0]));size=iris.shape[0]
    initial=float(np.hypot(w,h)*1.8)
    for f in range(count):
        if f==0:
            yield endpoint(spec,dimensions);continue
        t=f/max(1,count-1);ease=float(smooth(t))
        frame=np.full((h,w,3),250,np.uint8)
        # The whole iris starts outside all four edges; its colored field IS
        # the initial background. Its rim moves inward, never fades into view.
        diameter=final+(initial-final)*(1-ease)**2.2
        sc=diameter/size
        matrix=np.float32([[sc,0,w/2-size/2*sc],[0,sc,h/2-size/2*sc]])
        iris_layer=warp_rgba_premultiplied(iris,matrix,(w,h))
        iris_layer[:,:,3][iris_layer[:,:,3]>240]=255
        real_pupil=float(smooth((t-.84)/.06))
        if spec.get('layered_iris_background'):
            frame=pattern_frame(spec,dimensions,t,diameter)
            pupil_alpha=cv2.warpAffine(morph.pupil,matrix,(w,h),flags=cv2.INTER_LINEAR)
            pupil_layer=iris_layer.copy()
            pupil_layer[:,:,3]=np.minimum(pupil_layer[:,:,3],pupil_alpha)
            pupil_layer[:,:,3]=np.rint(pupil_layer[:,:,3]*real_pupil).astype(np.uint8)
            over_rgb(frame,pupil_layer,0,0)
            # Only the fine artwork/shading is blended at the end. The three
            # tooth masks themselves continuously move and resize into place.
            detail=float(smooth((t-.80)/.20))
            if detail:
                # The first character's glint belongs to the iris artwork and
                # shares its transform and opacity during the opening handoff.
                glow=np.zeros((h,w),np.uint8)
                cv2.ellipse(glow,(round(w/2-diameter*.045),round(h/2-diameter*.15)),
                            (max(2,round(diameter*.035)),max(2,round(diameter*.023))),
                            0,0,360,170,-1,cv2.LINE_AA)
                alpha=glow.astype(np.float32)/255
                iris_layer[:,:,:3]=(iris_layer[:,:,:3]*(1-alpha[:,:,None])+255*alpha[:,:,None]).astype(np.uint8)
            iris_layer[:,:,3]=np.rint(iris_layer[:,:,3]*detail).astype(np.uint8)
            over_rgb(frame,iris_layer,0,0)
        else:
            hole=cv2.warpAffine(morph.pupil,matrix,(w,h),flags=cv2.INTER_LINEAR).astype(np.float32)/255
            texture=float(smooth((t-.05)/.65))*(1-hole)+real_pupil*hole
            iris_layer[:,:,:3]=np.rint(bg*(1-texture[:,:,None])+iris_layer[:,:,:3]*texture[:,:,None]).astype(np.uint8)
            over_rgb(frame,iris_layer,0,0)
        size_lock=float(smooth(t/.50))
        morph_amount=float(smooth((t-.75)/.13))
        height=min(float(spec.get('mark_height',1050)),
                   (1-size_lock)*float(spec.get('mark_height',1050))+size_lock*pupil_height*diameter/size)
        layer=morph.layer(morph_amount,min(height,diameter*.82),dimensions)
        # The mark contracts in its original shape; only near its final size
        # does it turn into the pupil and hand off to the actual iris artwork.
        layer[:,:,3]=np.rint(layer[:,:,3]*(1-smooth((t-.84)/.06))).astype(np.uint8)
        over_rgb(frame,layer,0,0)
        yield frame


def make_ending(spec,shot,dimensions):
    w,h=dimensions;count=int(spec['frames']);loop=spec['loop']
    bg=resonance_background(loop,dimensions);eye=shot['iris'];morph=Morph(loop,eye)
    base_scale=float(shot['scale']);ex,ey=shot['eye_center_source']
    base=np.float64([[base_scale,0,w/2-ex*base_scale+shot['camera_dx'][-1]],
                     [0,base_scale,h/2-ey*base_scale+float(shot.get('camera_dy',0))],[0,0,1]])
    pupil_height=crop_mask(morph.pupil).shape[0]
    final_frame=endpoint(loop,dimensions)
    for f in range(count):
        if f==count-1:
            yield final_frame.copy();continue
        t=f/max(1,count-1)
        zoom=1+2*float(smooth(t/.75))
        z=np.float64([[zoom,0,w/2*(1-zoom)],[0,zoom,h/2*(1-zoom)],[0,0,1]])
        camera=(z@base)[:2]
        frame=bg.copy()
        # Render the actual final eye first, with both iris textures still bound.
        scene=np.full((h,w,3),250,np.uint8)
        from render import iris_group_transform
        group=warp_rgba_premultiplied(shot['iris_layer'],(z@np.vstack((iris_group_transform(shot,(w,h)),[0,0,1])))[:2],(w,h))
        window=None
        if shot.get('iris_clip') is not None:
            window=cv2.warpAffine(shot['iris_clip'][:,:,3],camera,(w,h),flags=cv2.INTER_LINEAR)
            group[:,:,3]=np.minimum(group[:,:,3],window)
        over_rgb(scene,group,0,0)
        occlusion=np.zeros((h,w),np.uint8)
        for key in ('face','hair','accessories'):
            if shot.get(key) is not None:
                layer=warp_rgba_premultiplied(shot[key],camera,(w,h))
                occlusion=np.maximum(occlusion,layer[:,:,3])
                over_rgb(scene,layer,0,0)
        fade=float(smooth((t-.10)/.56))
        frame=cv2.addWeighted(scene,1-fade,frame,fade,0)
        # Start from the real pupil at the scene's zoom, with the SAME eyelid
        # occlusion. Only after the face has gone may it unfold into a waveform.
        unfold=float(smooth((t-.62)/.38))
        amount=1-unfold
        height=(1-unfold)*pupil_height*zoom+unfold*float(loop.get('mark_height',1050))
        layer=morph.layer(amount,height,dimensions)
        layer[:,:,3]=np.rint(layer[:,:,3]*smooth(t/.18)).astype(np.uint8)
        if fade<1:
            aperture=255-occlusion
            if window is not None:aperture=np.minimum(aperture,window)
            layer[:,:,3]=np.minimum(layer[:,:,3],aperture)
        over_rgb(frame,layer,0,0)
        yield frame
