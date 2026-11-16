"""Place a stationary iris behind face layers with transparent eye openings."""
from pathlib import Path
import json, subprocess
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'deliverables'/'denia-v4'; OUT.mkdir(parents=True,exist_ok=True)
W,H=540,788
iris_path=ROOT/'deliverables'/'denia-v2'/'iris-independent.png'
iris=cv2.imdecode(np.frombuffer(iris_path.read_bytes(),np.uint8),cv2.IMREAD_UNCHANGED)
iris=cv2.resize(iris,None,fx=1.28,fy=1.28,interpolation=cv2.INTER_CUBIC)
ih,iw=iris.shape[:2]
cv2.imencode('.png',iris)[1].tofile(str(OUT/'iris-independent.png'))

# Eye-opening contours in final 540x788 coordinates.  The opaque face, lashes,
# and eyelids are above the iris.  Only the white and original iris are removed.
configs=[
 ('normal',458,720,[(157,390),(174,377),(191,368),(210,359),(230,353),(250,351),(276,351),(299,354),(320,361),(339,371),(352,385),(347,401),(338,416),(324,430),(307,441),(285,452),(265,454),(241,453),(218,446),(197,436),(180,422),(167,407)]),
 ('dark',740,720,[(177,385),(192,371),(211,361),(232,354),(255,351),(278,351),(300,353),(320,359),(339,368),(356,377),(374,390),(365,407),(352,422),(335,436),(314,446),(291,453),(267,454),(245,452),(224,441),(208,430),(194,416),(185,401)])
]
faces=[]
for name,x,y,points in configs:
    srcpath=ROOT/'assets'/'denia-v2'/f'{name}-full.png'
    src=cv2.imdecode(np.frombuffer(srcpath.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
    scale=1.30
    mat=np.array([[scale,0,W/2-x*scale],[0,scale,H/2-y*scale]],np.float32)
    frame=cv2.warpAffine(src,mat,(W,H),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    # Remove any colored fringe from the original iris just outside the cutout.
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    yy,xx=np.ogrid[:H,:W]
    fringe=(xx>205)&(xx<335)&(yy>347)&(yy<457)&(hsv[:,:,1]>65)&(hsv[:,:,2]>155)
    frame[fringe]=(250,250,250)
    # Supersampled cutout yields a soft antialiased border.
    scale_mask=4
    mask=np.zeros((H*scale_mask,W*scale_mask),np.uint8)
    pts=(np.array(points,np.int32)*scale_mask).reshape((-1,1,2))
    cv2.fillPoly(mask,[pts],255,lineType=cv2.LINE_AA)
    mask=cv2.GaussianBlur(mask,(5,5),0)
    mask=cv2.resize(mask,(W,H),interpolation=cv2.INTER_AREA)
    rgba=np.dstack((frame,255-mask))
    cv2.imencode('.png',rgba)[1].tofile(str(OUT/f'{name}-face-cutout.png'))
    cv2.imencode('.png',mask)[1].tofile(str(OUT/f'{name}-opening-mask.png'))
    faces.append(rgba)

def over(dst, src, x,y):
    h,w=src.shape[:2]; l=max(0,x); t=max(0,y); r=min(W,x+w); b=min(H,y+h)
    if l>=r or t>=b:return
    s=src[t-y:b-y,l-x:r-x]; d=dst[t:b,l:r]
    a=s[:,:,3:4].astype(np.float32)/255
    d[:]=(s[:,:,:3]*a+d*(1-a)).astype(np.uint8)

audio=json.loads((ROOT/'project.json').read_text(encoding='utf-8'))['source']
output=OUT/'denia-v4.mp4'
cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','30','-i','-', '-i',audio,'-map','0:v','-map','1:a:0?', '-t','3','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(output)]
p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
try:
    for i in range(90):
        frame=np.full((H,W,3),250,np.uint8)
        over(frame,iris,W//2-iw//2,H//2-ih//2)
        if i<8:
            # Ease out quickly, then retain a slow drift.
            dx=round(W*(1-i/8)**3+3)
            moving=cv2.GaussianBlur(faces[0],(17 if i<5 else 5,1),0)
            over(frame,moving,dx,0)
        elif i<40:
            over(frame,faces[0],round(3-(i-8)*0.09),0)
        elif i<49:
            # Keep a full outgoing backing under both moving layers: no blank gap.
            u=(i-40)/9
            over(frame,faces[0],0,0)
            outgoing=round(W*(u**2))
            incoming=-round(W*((1-u)**3))
            over(frame,cv2.GaussianBlur(faces[0],(19,1),0),outgoing,0)
            over(frame,cv2.GaussianBlur(faces[1],(19 if u<.65 else 7,1),0),incoming,0)
        else:
            over(frame,faces[1],round(min((i-49)*0.08,3)),0)
        if i in (20,65): cv2.imencode('.png',frame)[1].tofile(str(OUT/('normal-check.png' if i==20 else 'dark-check.png')))
        p.stdin.write(frame.tobytes())
finally:
    p.stdin.close(); code=p.wait()
if code: raise RuntimeError(f'ffmpeg failed: {code}')
print(output)
