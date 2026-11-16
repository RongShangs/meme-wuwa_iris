"""Place a stationary iris behind face layers with transparent eye openings."""
from pathlib import Path
import json, subprocess
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'deliverables'/'denia-v7'; OUT.mkdir(parents=True,exist_ok=True)
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
    # Segment the original white and colored iris by color in a bounded eye ROI.
    # Merge them and fill the old iris outline, retaining eyelashes as foreground.
    b,g,r=cv2.split(frame.astype(np.int16))
    eye_white=(b>244)&(g>241)&(r>247)
    if name=='normal': original_iris=(b-g>30)&(b>150)&(r>135)
    else: original_iris=(r-g>65)&(r>150)&(b>95)
    roi=np.zeros((H,W),np.uint8)
    if name=='normal':
        contour=[(160,392),(175,375),(195,364),(220,354),(243,349),(271,349),(300,353),(326,363),(350,378),(354,398),(343,418),(327,433),(307,447),(286,454),(263,455),(240,451),(218,444),(197,434),(178,420),(165,404)]
    else:
        contour=[(177,388),(194,374),(215,363),(238,354),(260,350),(280,350),(306,354),(330,362),(350,373),(374,390),(363,407),(350,421),(334,434),(314,444),(292,451),(270,455),(247,453),(226,446),(207,434),(191,418),(183,402)]
    cv2.fillPoly(roi,[np.array(contour,np.int32)],255)
    seed=((eye_white|original_iris)&(roi>0)).astype(np.uint8)*255
    iris_cover=np.zeros((H,W),np.uint8)
    cv2.ellipse(iris_cover,(270,394),(72,68),0,0,360,255,-1)
    seed=cv2.bitwise_or(seed,cv2.bitwise_and(iris_cover,roi))
    seed=cv2.morphologyEx(seed,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    seed=cv2.dilate(seed,np.ones((3,3),np.uint8),iterations=1)
    contours,_=cv2.findContours(seed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    mask=np.zeros((H,W),np.uint8)
    if not contours: raise RuntimeError(f'No eye mask for {name}')
    cv2.drawContours(mask,[max(contours,key=cv2.contourArea)],-1,255,-1,lineType=cv2.LINE_AA)
    # Remove the last remnants of the source iris outline immediately above
    # the lower lid; retain the lower lid and eye-corner strokes below it.
    cv2.rectangle(mask,(225,443),(315,454),255,-1)
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
output=OUT/'denia-v7.mp4'
cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','60','-i','-', '-i',audio,'-map','0:v','-map','1:a:0?', '-t','3','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(output)]
p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
def over_shifted(frame, face, dx, blur=False, extend=False, protect_iris=False):
    # Fractional affine shifts interpolate between pixels; do not round the drift.
    mat=np.array([[1,0,float(dx)],[0,1,0]],np.float32)
    shifted=cv2.warpAffine(face,mat,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE if extend else cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
    if protect_iris:
        # During the crossing the eye apertures move away from the fixed iris.
        # Cut a screen-space window through BOTH face layers so it stays visible.
        left=W//2-iw//2; top=H//2-ih//2
        old=shifted[top:top+ih,left:left+iw,3].astype(np.uint16)
        shifted[top:top+ih,left:left+iw,3]=(old*(255-iris[:,:,3].astype(np.uint16))//255).astype(np.uint8)
    over(frame,shifted,0,0)
try:
    for i in range(180):
        frame=np.full((H,W,3),250,np.uint8)
        over(frame,iris,W//2-iw//2,H//2-ih//2)
        if i<7:
            # Most deceleration is complete within ~0.12 seconds.
            u=i/7
            dx=W*(1-u)**4+1.2*(1-u)
            over_shifted(frame,faces[0],dx,protect_iris=True)
        elif i<80:
            over_shifted(frame,faces[0],-0.012*(i-7))
        elif i<92:
            # Keep a full outgoing backing under both moving layers: no blank gap.
            u=(i-80)/12
            outgoing=W*(u**2)
            incoming=-W*((1-u)**4)
            over_shifted(frame,faces[0],outgoing,extend=True,protect_iris=True)
            over_shifted(frame,faces[1],incoming,protect_iris=True)
        else:
            over_shifted(frame,faces[1],0.012*(i-92))
        if i in (40,130): cv2.imencode('.png',frame)[1].tofile(str(OUT/('normal-check.png' if i==40 else 'dark-check.png')))
        p.stdin.write(frame.tobytes())
finally:
    p.stdin.close(); code=p.wait()
if code: raise RuntimeError(f'ffmpeg failed: {code}')
print(output)
