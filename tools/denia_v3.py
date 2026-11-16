"""Place a stationary iris behind face layers with transparent eye openings."""
from pathlib import Path
import json, subprocess
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'deliverables'/'denia-v3'; OUT.mkdir(parents=True,exist_ok=True)
W,H=540,788
iris_path=ROOT/'deliverables'/'denia-v2'/'iris-independent.png'
iris=cv2.imdecode(np.frombuffer(iris_path.read_bytes(),np.uint8),cv2.IMREAD_UNCHANGED)
ih,iw=iris.shape[:2]
cv2.imencode('.png',iris)[1].tofile(str(OUT/'iris-independent.png'))

# Eye-opening contours in final 540x788 coordinates.  The opaque face, lashes,
# and eyelids are above the iris.  Only the white and original iris are removed.
configs=[
 ('normal',458,720,[(156,390),(174,377),(191,368),(210,361),(230,356),(252,355),(275,355),(298,358),(319,364),(338,373),(352,385),(347,401),(338,416),(324,428),(308,438),(287,445),(265,448),(241,447),(218,442),(197,434),(180,422),(167,407)]),
 ('dark',740,720,[(177,385),(192,373),(211,364),(232,358),(255,355),(278,355),(300,356),(320,361),(339,368),(356,377),(374,390),(365,407),(352,422),(335,434),(314,442),(291,447),(267,448),(245,445),(224,438),(208,428),(194,416),(185,401)])
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
    fringe=(xx>205)&(xx<335)&(yy>437)&(yy<458)&(hsv[:,:,1]>85)&(hsv[:,:,2]>160)
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
output=OUT/'denia-v3.mp4'
cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','30','-i','-', '-i',audio,'-map','0:v','-map','1:a:0?', '-t','3','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(output)]
p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
try:
    for i in range(90):
        shot=0 if i<45 else 1; local=i%45
        dx=round((1-min(local/5,1))**2*W)*(1 if shot==0 else -1)
        frame=np.full((H,W,3),250,np.uint8)
        over(frame,iris,W//2-iw//2,H//2-ih//2)
        over(frame,faces[shot],dx,0)
        if i in (20,65): cv2.imencode('.png',frame)[1].tofile(str(OUT/('normal-check.png' if shot==0 else 'dark-check.png')))
        p.stdin.write(frame.tobytes())
finally:
    p.stdin.close(); code=p.wait()
if code: raise RuntimeError(f'ffmpeg failed: {code}')
print(output)
