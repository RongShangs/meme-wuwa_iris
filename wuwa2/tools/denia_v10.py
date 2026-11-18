"""Overscanned full portraits, stationary iris, horizontal-only source timing."""
from pathlib import Path
import csv,json,subprocess
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'deliverables'/'denia-v10';OUT.mkdir(parents=True,exist_ok=True)
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
iris=read(old/'iris-independent.png',-1); ih,iw=iris.shape[:2]
faces=[]
for name,x in [('normal',458),('dark',740)]:
    source=read(ROOT/'assets'/'denia-v2'/f'{name}-full.png',1)
    mat=np.float32([[1.30,0,SW/2-x*1.30],[0,1.30,H/2-720*1.30]])
    overscan=cv2.warpAffine(source,mat,(SW,H),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
    rgba=np.dstack((overscan,np.full((H,SW),255,np.uint8)))
    # The existing cutout includes a repainted lower lid. Expand its opening
    # with connected near-white source pixels, not a rectangular eye cut.
    existing=read(old/f'{name}-face-cutout.png',-1)
    origin=overscan[:,MARGIN:MARGIN+W]
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
    # Remove isolated colored pixels below the second eye without touching its lashes.
    if name=='dark':
        patch=existing[460:470,307:325,:3]
        hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
        specks=((hsv[:,:,2]<200)&(hsv[:,:,1]>45)).astype(np.uint8)*255
        if np.any(specks):
            existing[460:470,307:325,:3]=cv2.inpaint(patch,specks,2,cv2.INPAINT_TELEA)
    rgba[:,MARGIN:MARGIN+W]=existing
    cv2.imencode('.png',rgba)[1].tofile(str(OUT/f'{name}-overscan-cutout.png'))
    check=np.zeros((H,W,3),np.uint8);check[:]=(0,220,0)
    over(check,existing,0,0)
    review=cv2.resize(check[320:480,145:395],(1000,640),interpolation=cv2.INTER_NEAREST)
    cv2.imencode('.png',review)[1].tofile(str(OUT/f'{name}-mask-review.png'))
    faces.append(rgba)

first=[55,31,23,19,16,15,14,14,11,7,0,-9,-21]
second=[47,20,16,11,9,7,6,4,2,0,-3,-9]
timeline=[(33+i,0,x-14) for i,x in enumerate(first)]+[(46+i,1,-(x-6)) for i,x in enumerate(second)]
with (OUT/'motion.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['output_frame','source_frame','form','horizontal_px','vertical_px'])
    for i,(src,form,dx) in enumerate(timeline):w.writerow([i,src,form+1,dx,0])

audio=json.loads((ROOT/'project.json').read_text(encoding='utf-8'))['source']
output=OUT/'denia-v10.mp4'
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
        if index in (6,19):cv2.imencode('.png',frame)[1].tofile(str(OUT/('normal-check.png' if shot==0 else 'dark-check.png')))
        p.stdin.write(frame.tobytes())
finally:
    p.stdin.close();code=p.wait()
if code:raise RuntimeError(f'ffmpeg exited {code}')
print(output)
