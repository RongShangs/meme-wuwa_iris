"""Composite full-canvas generated transparent cels, then frame with a camera."""
from pathlib import Path
import csv,json,subprocess
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
A=ROOT/'assets'/'denia-v19';OUT=ROOT/'deliverables'/'denia-v19'
OUT.mkdir(parents=True,exist_ok=True)
W,H=540,788

def read(path):
    a=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_UNCHANGED)
    if a is None or a.ndim!=3 or a.shape[2]!=4:raise ValueError(f'Expected transparent PNG: {path}')
    return a

def rgba_over(dst,src):
    da=dst[:,:,3:4].astype(np.float32)/255;sa=src[:,:,3:4].astype(np.float32)/255
    oa=sa+da*(1-sa)
    numerator=src[:,:,:3].astype(np.float32)*sa+dst[:,:,:3].astype(np.float32)*da*(1-sa)
    rgb=np.divide(numerator,oa,out=np.zeros_like(numerator),where=oa>0)
    return np.dstack((np.clip(rgb,0,255).astype(np.uint8),np.clip(oa[:,:,0]*255,0,255).astype(np.uint8)))

def over_rgb(dst,src,x,y):
    h,w=src.shape[:2];l=max(0,x);t=max(0,y);r=min(W,x+w);b=min(H,y+h)
    if l>=r or t>=b:return
    s=src[t-y:b-y,l-x:r-x];a=s[:,:,3:4].astype(np.float32)/255
    dst[t:b,l:r]=(dst[t:b,l:r]*(1-a)+s[:,:,:3]*a).astype(np.uint8)

faces=[read(A/'normal-face.png'),read(A/'dark-face.png')]
hair=read(A/'bangs.png');accessories=read(A/'accessories.png')
iris=read(ROOT/'assets'/'denia-v16'/'iris-ready.png');ih,iw=iris.shape[:2]
for im in faces:
    if im.shape!=hair.shape or im.shape!=accessories.shape:raise ValueError('Cel canvas sizes do not match')
    if im[720,458,3]>16 or im[720,740,3]>16:raise ValueError('Eye interiors are not transparent')
CH,CW=hair.shape[:2]

first=[55,31,23,19,16,15,14,14,11,7,0,-9,-21]
second=[47,20,16,11,9,7,6,4,2,0,-3,-9]
timeline=[(33+i,0,x-14) for i,x in enumerate(first)]+[(46+i,1,-(x-6)) for i,x in enumerate(second)]
audio=json.loads((ROOT/'project.json').read_text(encoding='utf-8'))['source']
output=OUT/'denia-v19.mp4'
cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','30','-i','-', '-ss',str(33/30),'-i',audio,'-map','0:v','-map','1:a:0?', '-t',str(25/30),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(output)]
p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
try:
    for i,(_,shot,dx) in enumerate(timeline):
        local=i if shot==0 else i-13
        swing=[-6,-3,-1,0,1.5,3,4,3,1,-2,-5,-8,-11]
        sway=swing[min(local,len(swing)-1)]*(1 if shot==0 else -1)
        # Rotate around the crown; each source cel remains a complete canvas.
        rotation=cv2.getRotationMatrix2D((600,300),sway*.10,1)
        rotation[0,2]+=sway/1.30
        moving=cv2.warpAffine(hair,rotation,(CW,CH),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
        scene=rgba_over(rgba_over(faces[shot],moving),accessories)
        # Camera crop occurs only after full-canvas assembly. Iris is fixed to screen.
        eye_x=458 if shot==0 else 740
        camera=np.float32([[1.30,0,W/2-eye_x*1.30+dx],[0,1.30,H/2-720*1.30]])
        framed=cv2.warpAffine(scene,camera,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
        frame=np.full((H,W,3),250,np.uint8)
        over_rgb(frame,iris,W//2-iw//2,H//2-ih//2)
        over_rgb(frame,framed,0,0)
        if i in (6,19):cv2.imencode('.png',frame)[1].tofile(str(OUT/('normal-check.png' if shot==0 else 'dark-check.png')))
        p.stdin.write(frame.tobytes())
finally:
    p.stdin.close();code=p.wait()
if code:raise RuntimeError(f'ffmpeg exited {code}')
print(output)
