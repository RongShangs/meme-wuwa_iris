"""Overscanned full portraits, stationary iris, horizontal-only source timing."""
from pathlib import Path
import csv,json,subprocess,sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'deliverables'/'denia-v16';OUT.mkdir(parents=True,exist_ok=True)
W,H=540,788; MARGIN=100; SW=W+2*MARGIN

def read(p,mode):
    a=cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),mode)
    if a is None: raise FileNotFoundError(p)
    return a

def over(dst,src,x,y):
    h,w=src.shape[:2];l=max(0,x);t=max(0,y);r=min(W,x+w);b=min(H,y+h)
    if l>=r or t>=b:return
    s=src[t-y:b-y,l-x:r-x];a=s[:,:,3:4].astype(np.float32)/255
    dst[t:b,l:r]=(dst[t:b,l:r]*(1-a)+s[:,:,:3]*a).astype(np.uint8)

old=ROOT/'deliverables'/'denia-v8'
iris=read(ROOT/'assets'/'denia-v16'/'iris-ready.png',-1); ih,iw=iris.shape[:2]
with_highlight='--no-highlight' not in sys.argv
# Independently editable white highlight, rasterized at 4x then antialiased.
dot=np.zeros((64,64,4),np.uint8)
cv2.circle(dot,(32,32),23,(255,255,255,255),-1,cv2.LINE_AA)
dot=cv2.resize(dot,(16,16),interpolation=cv2.INTER_AREA)
cv2.imencode('.png',dot)[1].tofile(str(OUT/'highlight.png'))
faces=[];hairs=[]
for name,x in [('normal',458),('dark',740)]:
    sourcepath=(ROOT/'assets'/'denia-v11'/'dark-fierce-full.png') if name=='dark' else (ROOT/'assets'/'denia-v2'/'normal-full.png')
    source=read(sourcepath,1)
    mat=np.float32([[1.30,0,SW/2-x*1.30],[0,1.30,H/2-720*1.30]])
    overscan=cv2.warpAffine(source,mat,(SW,H),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
    rgba=np.dstack((overscan,np.full((H,SW),255,np.uint8)))
    # The existing cutout includes a repainted lower lid. Expand its opening
    # with connected near-white source pixels, not a rectangular eye cut.
    existing=read(old/f'{name}-face-cutout.png',-1)
    origin=overscan[:,MARGIN:MARGIN+W]
    if name=='dark':
        # Retain the new expression while using the prior transparency as a starting mask.
        # Preserve the repaired lower lid from v10; the new upper lid is restored below.
        retained=existing[435:464,220:321].copy()
        existing[:,:,:3]=origin
        existing[435:464,220:321]=retained
    b,g,r=cv2.split(origin.astype(np.int16))
    white=(b>=243)&(g>=240)&(r>=246)
    yy,xx=np.ogrid[:H,:W]
    bounds=(xx>=155)&(xx<=385)&(yy>=348)&(yy<=456)
    opening=(existing[:,:,3]<100)
    combined=(opening | (white & bounds)).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(combined,8)
    eye_label=labels[H//2,W//2]
    if eye_label==0: raise RuntimeError('eye opening is not connected')
    refined=(labels==eye_label)
    # Include only immediately adjacent antialiased white edge pixels.
    border=cv2.dilate(refined.astype(np.uint8),np.ones((3,3),np.uint8))>0
    softwhite=(b>=238)&(g>=235)&(r>=244)&bounds&border
    refined|=softwhite
    existing[:,:,3][refined]=0
    if name=='dark':
        # Follow the new upper lid column by column.  Restore skin and ink above
        # the lid, but do not resurrect dark pixels of the original iris below it.
        hsv_new=cv2.cvtColor(origin,cv2.COLOR_BGR2HSV)
        edges=[]
        for col in range(170,385):
            ys=np.flatnonzero(hsv_new[350:385,col,2]<145)
            edges.append((350+int(ys[-1])) if len(ys) else 372)
        # Suppress isolated iris-outline spikes in the per-column edge estimate.
        edges=cv2.medianBlur((np.asarray(edges,dtype=np.int16)-350).astype(np.uint8).reshape(1,-1),15).ravel().astype(np.int16)+350
        for col,edge in zip(range(170,385),edges):
            existing[320:int(edge)+1,col,3]=255
    # Remove isolated colored pixels below the second eye without touching its lashes.
    if name=='dark':
        patch=existing[460:470,307:325,:3]
        hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
        specks=((hsv[:,:,2]<200)&(hsv[:,:,1]>45)).astype(np.uint8)*255
        if np.any(specks):
            existing[460:470,307:325,:3]=cv2.inpaint(patch,specks,2,cv2.INPAINT_TELEA)
        # Fresh tracing for the narrowed eye: do not reuse the old eye aperture.
        # Points follow the underside of the newly generated lid and the lower
        # eye boundary in the 540x788 coordinate space.
        contour=[(170,400),(180,393),(190,389),(200,384),(210,381),(220,379),
                 (230,377),(240,375),(250,373),(260,372),(270,371),(280,370),
                 (290,370),(300,370),(310,370),(320,371),(330,372),(340,374),
                 (350,375),(360,377),(370,379),(374,381),(370,405),(365,420),
                 (350,432),(335,440),(320,448),(305,452),(280,455),(260,454),
                 (240,451),(225,448),(210,441),(195,432),(185,417),(177,405)]
        high=np.zeros((H*4,W*4),np.uint8)
        cv2.fillPoly(high,[(np.array(contour,np.int32)*4).reshape(-1,1,2)],255,lineType=cv2.LINE_AA)
        aperture=cv2.resize(high,(W,H),interpolation=cv2.INTER_AREA)
        existing[:,:,:3]=origin
        existing[:,:,3]=255-aperture
        # Remove old iris rim at the lower lid and restore a clean foreground stroke.
        existing[435:463,220:321,:3]=origin[468:469,220:321]
        line=np.zeros((H*4,W*4,4),np.uint8)
        arc=np.array([(218,447),(230,451),(244,454),(260,456),(276,456),(292,455),(308,452),(322,448)],np.int32)*4
        cv2.polylines(line,[arc.reshape(-1,1,2)],False,(74,30,121,255),8,cv2.LINE_AA)
        line=cv2.resize(line,(W,H),interpolation=cv2.INTER_AREA)
        la=line[:,:,3:4].astype(np.float32)/255
        existing[:,:,:3]=(existing[:,:,:3]*(1-la)+line[:,:,:3]*la).astype(np.uint8)
        existing[:,:,3]=np.maximum(existing[:,:,3],line[:,:,3])
    rgba[:,MARGIN:MARGIN+W]=existing
    # Trace the complete visible forehead fringe, including its outer forelocks.
    # Keep the eyelid and eye aperture outside the animated cutout.
    hairmask=np.zeros((H,SW),np.uint8)
    if name=='normal':
        poly=[(0,0),(540,0),(540,395),(510,401),(480,390),(450,375),
              (420,356),(395,330),(375,302),(357,266),(345,226),(335,178),
              (329,120),(329,60),(321,0),(290,0),(280,80),(262,133),
              (235,185),(210,230),(190,275),(172,323),(163,370),(170,416),
              (185,470),(218,530),(179,520),(140,499),(98,469),(60,432),(0,390)]
    else:
        poly=[(0,0),(540,0),(540,304),(510,325),(480,339),(450,346),
              (415,347),(385,340),(355,329),(325,312),(300,296),(275,277),
              (250,254),(225,229),(200,207),(175,180),(153,152),(133,116),
              (113,64),(106,0),(95,0),(98,80),(108,170),(124,250),
              (143,312),(162,356),(190,390),(158,401),(120,392),(80,373),
              (40,345),(0,310)]
    points=np.array([(px+MARGIN,py) for px,py in poly],np.int32)
    cv2.fillPoly(hairmask,[points],255,lineType=cv2.LINE_AA)
    hair=np.dstack((rgba[:,:,:3].copy(),hairmask))
    # Inpaint under the whole cutout, not merely its edge, so translated ink
    # lines cannot reveal a second copy of the original fringe underneath.
    repaired=cv2.inpaint(rgba[:,:,:3],hairmask,3,cv2.INPAINT_TELEA)
    rgba[:,:,:3][hairmask>0]=repaired[hairmask>0]
    cv2.imencode('.png',hair)[1].tofile(str(OUT/f'{name}-fringe.png'))
    cv2.imencode('.png',rgba)[1].tofile(str(OUT/f'{name}-overscan-cutout.png'))
    check=np.zeros((H,W,3),np.uint8);check[:]=(0,220,0)
    over(check,existing,0,0)
    review=cv2.resize(check[320:480,145:395],(1000,640),interpolation=cv2.INTER_NEAREST)
    cv2.imencode('.png',review)[1].tofile(str(OUT/f'{name}-mask-review.png'))
    faces.append(rgba)
    hairs.append(hair)

first=[55,31,23,19,16,15,14,14,11,7,0,-9,-21]
second=[47,20,16,11,9,7,6,4,2,0,-3,-9]
timeline=[(33+i,0,x-14) for i,x in enumerate(first)]+[(46+i,1,-(x-6)) for i,x in enumerate(second)]
with (OUT/'motion.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['output_frame','source_frame','form','horizontal_px','vertical_px'])
    for i,(src,form,dx) in enumerate(timeline):w.writerow([i,src,form+1,dx,0])

audio=json.loads((ROOT/'project.json').read_text(encoding='utf-8'))['source']
output=OUT/('denia-v16.mp4' if with_highlight else 'denia-v16-no-highlight.mp4')
cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','30','-i','-', '-ss',str(33/30),'-i',audio,'-map','0:v','-map','1:a:0?', '-t',str(25/30),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(output)]
p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
try:
    for index,(_,shot,dx) in enumerate(timeline):
        frame=np.full((H,W,3),250,np.uint8)
        # Composite the movable glint over the iris and clip it to its alpha.
        # The face is then composited above both, so the lid can hide it.
        iris_frame=iris.copy()
        if with_highlight:
            local_x=round(iw/2-8+dx*0.6); local_y=round(ih/2-37)
            x0=local_x-dot.shape[1]//2; y0=local_y-dot.shape[0]//2
            x1=x0+dot.shape[1];y1=y0+dot.shape[0]
            if 0<=x0 and 0<=y0 and x1<=iw and y1<=ih:
                dest=iris_frame[y0:y1,x0:x1];a=(dot[:,:,3:4].astype(np.float32)/255)*(dest[:,:,3:4].astype(np.float32)/255)
                dest[:,:,:3]=(dest[:,:,:3]*(1-a)+dot[:,:,:3]*a).astype(np.uint8)
        over(frame,iris_frame,W//2-iw//2,H//2-ih//2)
        # Sampling from the overscanned full portrait avoids stretching borders.
        face=faces[shot][:,MARGIN-dx:MARGIN-dx+W]
        if face.shape[1]!=W:raise ValueError('not enough overscan')
        over(frame,face,0,0)
        # A small delayed counter-motion makes the cutout lag behind the face.
        local=index if shot==0 else index-13
        wave=[-6,-3,-1,0,1.5,3,4,3,1,-2,-5,-8,-11]
        sway=wave[min(local,len(wave)-1)] * (1 if shot==0 else -1)
        mat_h=np.float32([[1,0,sway],[0,1,0]])
        hair_moved=cv2.warpAffine(hairs[shot],mat_h,(SW,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
        hair_crop=hair_moved[:,MARGIN-dx:MARGIN-dx+W]
        over(frame,hair_crop,0,0)
        if index in (6,19):cv2.imencode('.png',frame)[1].tofile(str(OUT/('normal-check.png' if shot==0 else 'dark-check.png')))
        p.stdin.write(frame.tobytes())
finally:
    p.stdin.close();code=p.wait()
if code:raise RuntimeError(f'ffmpeg exited {code}')
print(output)
