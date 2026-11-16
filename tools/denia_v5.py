"""Place a stationary iris behind face layers with transparent eye openings."""
from pathlib import Path
import json, subprocess
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'deliverables'/'denia-v5'; OUT.mkdir(parents=True,exist_ok=True)
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
    # Remove colored fringe from the original iris around the new opening.
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    yy,xx=np.ogrid[:H,:W]
    fringe=(xx>205)&(xx<335)&(yy>354)&(yy<457)&(hsv[:,:,1]>65)&(hsv[:,:,2]>155)
    frame[fringe]=(250,250,250)
    # Clear the old iris outline at the lower lid before replacing that stroke.
    # Sample nearby skin for each column to avoid retaining purple/magenta fragments.
    frame[435:464,220:321]=frame[468:469,220:321]
    # Supersampled cutout yields a soft antialiased border.
    scale_mask=4
    mask=np.zeros((H*scale_mask,W*scale_mask),np.uint8)
    pts=(np.array(points,np.int32)*scale_mask).reshape((-1,1,2))
    cv2.fillPoly(mask,[pts],255,lineType=cv2.LINE_AA)
    mask=cv2.GaussianBlur(mask,(5,5),0)
    mask=cv2.resize(mask,(W,H),interpolation=cv2.INTER_AREA)
    rgba=np.dstack((frame,255-mask))
    # Reconstruct a smooth lower-eyelid stroke as a separate opaque foreground line.
    high=np.zeros((H*4,W*4,4),np.uint8)
    lower=np.array([(214,447),(229,452),(247,456),(269,458),(291,457),(310,454),(326,449)],np.int32)*4
    cv2.polylines(high,[lower.reshape(-1,1,2)],False,(74,30,121,255),7,cv2.LINE_AA)
    line=cv2.resize(high,(W,H),interpolation=cv2.INTER_AREA)
    la=line[:,:,3:4].astype(np.float32)/255
    rgba[:,:,:3]=(line[:,:,:3]*la+rgba[:,:,:3]*(1-la)).astype(np.uint8)
    rgba[:,:,3]=np.maximum(rgba[:,:,3],line[:,:,3])
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
output=OUT/'denia-v5.mp4'
cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','60','-i','-', '-i',audio,'-map','0:v','-map','1:a:0?', '-t','3','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(output)]
p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
def over_shifted(frame, face, dx, blur=False):
    # Fractional affine shifts interpolate between pixels; do not round the drift.
    mat=np.array([[1,0,float(dx)],[0,1,0]],np.float32)
    shifted=cv2.warpAffine(face,mat,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
    if blur: shifted=cv2.GaussianBlur(shifted,(15,1),0)
    over(frame,shifted,0,0)
try:
    for i in range(180):
        frame=np.full((H,W,3),250,np.uint8)
        over(frame,iris,W//2-iw//2,H//2-ih//2)
        if i<7:
            # Most deceleration is complete within ~0.12 seconds.
            u=i/7
            dx=W*(1-u)**4+1.2*(1-u)
            over_shifted(frame,faces[0],dx,blur=i<4)
        elif i<80:
            over_shifted(frame,faces[0],-0.012*(i-7))
        elif i<92:
            # Keep a full outgoing backing under both moving layers: no blank gap.
            u=(i-80)/12
            over(frame,faces[0],0,0)
            outgoing=W*(u**2)
            incoming=-W*((1-u)**4)
            over_shifted(frame,faces[0],outgoing,blur=True)
            over_shifted(frame,faces[1],incoming,blur=u<.55)
        else:
            over_shifted(frame,faces[1],0.012*(i-92))
        if i in (40,130): cv2.imencode('.png',frame)[1].tofile(str(OUT/('normal-check.png' if i==40 else 'dark-check.png')))
        p.stdin.write(frame.tobytes())
finally:
    p.stdin.close(); code=p.wait()
if code: raise RuntimeError(f'ffmpeg failed: {code}')
print(output)
