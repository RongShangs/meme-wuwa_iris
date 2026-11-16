"""Two fast face entrances with a fixed, separately drawn circular iris."""
from pathlib import Path
import json, subprocess
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'deliverables'/'denia-v2'; OUT.mkdir(parents=True,exist_ok=True)
W,H,FPS=540,788,30
R=57

def make_iris():
    # Independently drawn at 4x for smooth circular edges; not cut from a face.
    k=4; n=2*R*k+8*k; c=n//2
    a=np.zeros((n,n,4),np.uint8)
    def circle(radius,color): cv2.circle(a,(c,c),round(radius*k),color,-1,cv2.LINE_AA)
    circle(R,(125,62,152,255)); circle(R-3,(198,142,204,255))
    circle(R*0.51,(155,93,175,255)); circle(R*0.30,(116,51,130,255))
    cv2.circle(a,(c-round(0.23*R*k),c-round(0.30*R*k)),round(R*.13*k),(255,250,255,255),-1,cv2.LINE_AA)
    return cv2.resize(a,(n//k,n//k),interpolation=cv2.INTER_AREA)

iris=make_iris(); cv2.imencode('.png',iris)[1].tofile(str(OUT/'iris-independent.png'))
frames=[]; tops=[]
for name,x,y in [('normal',458,720),('dark',740,720)]:
    path=ROOT/'assets'/'denia-v2'/f'{name}-full.png'
    src=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
    scale=1.30
    mat=np.array([[scale,0,W/2-x*scale],[0,scale,H/2-y*scale]],np.float32)
    frame=cv2.warpAffine(src,mat,(W,H),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    # Remove the old colored iris only: keep dark linework and eyelids intact.
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    yy,xx=np.ogrid[:H,:W]
    region=((xx-W//2)**2/(59**2)+(yy-H//2)**2/(59**2))<1
    colored=(hsv[:,:,2]>145)&region
    # Replace iris colors with the local white of the eye; the new iris covers most of it.
    frame[colored]=(248,248,255)
    cv2.imencode('.png',frame)[1].tofile(str(OUT/f'{name}-face.png'))
    frames.append(frame)
    # Preserve the dark eyelash stroke above the fixed iris.
    dark=(hsv[:,:,2]<110)&(yy < H//2-26)&region
    tops.append((dark,cv2.warpAffine(src,mat,(W,H),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)))

audio=json.loads((ROOT/'project.json').read_text(encoding='utf-8'))['source']
cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-', '-i',audio,'-map','0:v','-map','1:a:0?', '-t','3','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(OUT/'denia-v2.mp4')]
p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
try:
    for i in range(90):
        shot=0 if i<45 else 1; local=i%45
        # Start off-screen and settle over five frames; iris never moves.
        dx=round((1-min(local/5,1))**2*W)*(1 if shot==0 else -1)
        frame=np.full((H,W,3),250,np.uint8)
        if dx>=0: frame[:,dx:]=frames[shot][:,:W-dx] if dx<W else frame[:,dx:]
        else: frame[:,:W+dx]=frames[shot][:,-dx:] if -dx<W else frame[:,:W+dx]
        ih,iw=iris.shape[:2]; x=W//2-iw//2; y=H//2-ih//2
        roi=frame[y:y+ih,x:x+iw]; a=iris[:,:,3:4].astype(np.float32)/255
        roi[:]=(roi*(1-a)+iris[:,:,:3]*a).astype(np.uint8)
        # Restore upper lash after settling so the iris sits behind the lid.
        if dx==0:
            mask,original=tops[shot]; frame[mask]=original[mask]
        p.stdin.write(frame.tobytes())
finally:
    p.stdin.close(); code=p.wait()
if code: raise RuntimeError(f'ffmpeg failed: {code}')
print(OUT/'denia-v2.mp4')
