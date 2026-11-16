"""Match source frames 33..57; second shot's horizontal movement is mirrored."""
from pathlib import Path
import csv, json, subprocess
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'deliverables'/'denia-v9'; OUT.mkdir(parents=True,exist_ok=True)
W,H=540,788

def read(path,mode):
    return cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),mode)

def over(dst,src,x,y):
    h,w=src.shape[:2]; l=max(0,x);t=max(0,y);r=min(W,x+w);b=min(H,y+h)
    if l>=r or t>=b:return
    s=src[t-y:b-y,l-x:r-x];a=s[:,:,3:4].astype(np.float32)/255
    dst[t:b,l:r]=(dst[t:b,l:r]*(1-a)+s[:,:,:3]*a).astype(np.uint8)

old=ROOT/'deliverables'/'denia-v8'
iris=read(old/'iris-independent.png',cv2.IMREAD_UNCHANGED)
ih,iw=iris.shape[:2]
shots=[]
for name in ('normal','dark'):
    face=read(old/f'{name}-face-cutout.png',cv2.IMREAD_UNCHANGED)
    if face is None: raise FileNotFoundError(old/f'{name}-face-cutout.png')
    frame=np.full((H,W,3),250,np.uint8)
    over(frame,iris,W//2-iw//2,H//2-ih//2)
    over(frame,face,0,0)
    shots.append(frame)

# Template matching (at 540x788) measures the source frame relative to a
# reference frame. Baseline is the slow middle of each shot; x for shot 2 is
# negated to produce the requested reversed entrance. y is left unchanged.
first_x=[55,31,23,19,16,15,14,14,11,7,0,-9,-21]
first_y=[-2,-2,-2,-1,-1,-1,0,0,0,0,0,0,0]
second_x=[47,20,16,11,9,7,6,4,2,0,-3,-9]
second_y=[2,2,-6,-4,-2,0,1,1,0,0,-1,-3]
timeline=[]
for n,(x,y) in enumerate(zip(first_x,first_y)):
    timeline.append((33+n,0,x-14,y))
for n,(x,y) in enumerate(zip(second_x,second_y)):
    timeline.append((46+n,1,-(x-6),y))
with (OUT/'motion.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f);writer.writerow(['output_frame','source_frame','shot','dx_px_540','dy_px_788'])
    for i,(source,shot,dx,dy) in enumerate(timeline):writer.writerow([i,source,shot+1,dx,dy])

audio=json.loads((ROOT/'project.json').read_text(encoding='utf-8'))['source']
output=OUT/'denia-v9.mp4'
cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','30','-i','-', '-ss',str(33/30),'-i',audio,'-map','0:v','-map','1:a:0?', '-t',str(25/30),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(output)]
p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
try:
    for _,shot,dx,dy in timeline:
        mat=np.array([[1,0,dx],[0,1,dy]],np.float32)
        frame=cv2.warpAffine(shots[shot],mat,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        p.stdin.write(frame.tobytes())
finally:
    p.stdin.close(); code=p.wait()
if code: raise RuntimeError(f'ffmpeg failed: {code}')
print(output)
