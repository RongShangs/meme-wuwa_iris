"""Overscanned full portraits, stationary iris, horizontal-only source timing."""
from pathlib import Path
import csv,json,subprocess
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'deliverables'/'denia-v13';OUT.mkdir(parents=True,exist_ok=True)
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
iris=read(ROOT/'assets'/'denia-v13'/'iris-ready.png',-1); ih,iw=iris.shape[:2]
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
    rgba[:,MARGIN:MARGIN+W]=existing
    # The foreground fringe is cut from the original portrait. The lasso is
    # deliberately limited to one lock per view; it does not include the eye.
    hairmask=np.zeros((H,SW),np.uint8)
    if name=='normal':
        poly=[(292,0),(540,0),(540,394),(510,398),(480,388),(450,373),(420,353),(395,330),(375,302),(356,265),(343,225),(333,175),(327,120),(327,55),(315,0)]
    else:
        poly=[(0,0),(105,0),(105,70),(113,130),(124,193),(143,251),(160,300),(181,350),(198,375),(168,397),(139,390),(110,375),(80,350),(50,315),(23,280),(0,250)]
    points=np.array([(px+MARGIN,py) for px,py in poly],np.int32)
    cv2.fillPoly(hairmask,[points],255,lineType=cv2.LINE_AA)
    hair=np.dstack((rgba[:,:,:3].copy(),hairmask))
    # Reconstruct only the narrow strip revealed by the lock's small movement.
    inner=cv2.erode(hairmask,np.ones((9,9),np.uint8))
    ring=cv2.subtract(hairmask,inner)
    ring=(ring>0).astype(np.uint8)*255
    repaired=cv2.inpaint(rgba[:,:,:3],ring,3,cv2.INPAINT_TELEA)
    rgba[:,:,:3][ring>0]=repaired[ring>0]
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
output=OUT/'denia-v13.mp4'
cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','30','-i','-', '-ss',str(33/30),'-i',audio,'-map','0:v','-map','1:a:0?', '-t',str(25/30),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(output)]
p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
try:
    for index,(_,shot,dx) in enumerate(timeline):
        frame=np.full((H,W,3),250,np.uint8)
        over(frame,iris,W//2-iw//2,H//2-ih//2)
        # Sampling from the overscanned full portrait avoids stretching borders.
        face=faces[shot][:,MARGIN-dx:MARGIN-dx+W]
        if face.shape[1]!=W:raise ValueError('not enough overscan')
        over(frame,face,0,0)
        # A small delayed counter-motion makes the cutout lag behind the face.
        local=index if shot==0 else index-13
        wave=[-3,-2,-1,0,1,1.5,1.5,1,0,-0.5,-1,-2,-2.5]
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
