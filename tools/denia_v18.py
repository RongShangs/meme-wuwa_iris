"""Animation from generated transparent face, iris, and separate fringe cels."""
from pathlib import Path
import csv,json,subprocess
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
ASSETS=ROOT/'assets'/'denia-v17'
OUT=ROOT/'deliverables'/'denia-v18';OUT.mkdir(parents=True,exist_ok=True)
W,H,MARGIN=540,788,100;SW=W+2*MARGIN

def read(path):
    image=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_UNCHANGED)
    if image is None:raise FileNotFoundError(path)
    return image

def over(dst,src,x,y):
    h,w=src.shape[:2];l=max(0,x);t=max(0,y);r=min(W,x+w);b=min(H,y+h)
    if l>=r or t>=b:return
    s=src[t-y:b-y,l-x:r-x];a=s[:,:,3:4].astype(np.float32)/255
    dst[t:b,l:r]=(dst[t:b,l:r]*(1-a)+s[:,:,:3]*a).astype(np.uint8)

iris=read(ROOT/'assets'/'denia-v16'/'iris-ready.png'); ih,iw=iris.shape[:2]
fringe=read(ASSETS/'bangs-generated.png')
# Keep only the main component; detached pixels cannot appear as hair fragments.
binary=(fringe[:,:,3]>32).astype(np.uint8)
n,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
idx=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);fringe[:,:,3][labels!=idx]=0
# Headband and side jewel remain attached to the face in front of the fringe.
ys=np.arange(fringe.shape[0])[:,None]
fringe[:,:,3]=(fringe[:,:,3]*np.clip((ys-380)/20,0,1)).astype(np.uint8)
cv2.imencode('.png',fringe)[1].tofile(str(OUT/'fringe-clean.png'))
faces=[];hairs=[];ornaments=[]
for name,x in [('normal',458),('dark',740)]:
    source=read(ASSETS/f'{name}-alpha.png')
    if source.shape[2]!=4:raise ValueError(f'{name} face has no alpha channel')
    if source[720,x,3]>16:raise ValueError(f'{name} eye was not made transparent')
    # Eliminate small detached alpha artifacts around the holes while preserving
    # the main connected character component and the original eyelash geometry.
    mask=(source[:,:,3]>32).astype(np.uint8)
    n,lab,stats,_=cv2.connectedComponentsWithStats(mask,8)
    for k in range(1,n):
        left,top,width,height,area=stats[k]
        if area<60 and 320<left<880 and 630<top<800:
            source[:,:,3][lab==k]=0
    mat=np.float32([[1.30,0,SW/2-x*1.30],[0,1.30,H/2-720*1.30]])
    face=cv2.warpAffine(source,mat,(SW,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
    hair=cv2.warpAffine(fringe,mat,(SW,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
    # Restore the stationary jewel from the original with-bangs portrait, above
    # the animated hair so moving strands cannot erase or carry it away.
    original_path=(ROOT/'assets'/'denia-v11'/'dark-fierce-full.png') if name=='dark' else (ROOT/'assets'/'denia-v2'/'normal-full.png')
    original=read(original_path)
    gem_alpha=np.zeros(original.shape[:2],np.uint8)
    gem_shape=np.array([(840,477),(855,484),(874,482),(892,472),(908,462),
                        (902,490),(906,508),(910,529),(907,548),(899,566),
                        (885,578),(870,582),(852,579),(838,570),(828,556),
                        (823,538),(823,520),(827,501)],np.int32)
    cv2.fillPoly(gem_alpha,[gem_shape],255,lineType=cv2.LINE_AA)
    gem_source=np.dstack((original[:,:,:3],gem_alpha))
    ornament=cv2.warpAffine(gem_source,mat,(SW,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
    cv2.imencode('.png',face)[1].tofile(str(OUT/f'{name}-face.png'))
    cv2.imencode('.png',hair)[1].tofile(str(OUT/f'{name}-fringe.png'))
    # Bright green inspection background makes leftover white eye pixels visible.
    check=np.zeros((H,W,3),np.uint8);check[:]=(0,220,0)
    over(check,face[:,MARGIN:MARGIN+W],0,0)
    crop=cv2.resize(check[320:480,145:395],(1000,640),interpolation=cv2.INTER_NEAREST)
    cv2.imencode('.png',crop)[1].tofile(str(OUT/f'{name}-mask-review.png'))
    faces.append(face);hairs.append(hair);ornaments.append(ornament)

first=[55,31,23,19,16,15,14,14,11,7,0,-9,-21]
second=[47,20,16,11,9,7,6,4,2,0,-3,-9]
timeline=[(33+i,0,x-14) for i,x in enumerate(first)]+[(46+i,1,-(x-6)) for i,x in enumerate(second)]
audio=json.loads((ROOT/'project.json').read_text(encoding='utf-8'))['source']
output=OUT/'denia-v18.mp4'
cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','30','-i','-', '-ss',str(33/30),'-i',audio,'-map','0:v','-map','1:a:0?', '-t',str(25/30),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(output)]
p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
try:
    for i,(_,shot,dx) in enumerate(timeline):
        frame=np.full((H,W,3),250,np.uint8)
        # The unhighlighted iris remains anchored in screen coordinates.
        over(frame,iris,W//2-iw//2,H//2-ih//2)
        over(frame,faces[shot][:,MARGIN-dx:MARGIN-dx+W],0,0)
        local=i if shot==0 else i-13
        swing=[-6,-3,-1,0,1.5,3,4,3,1,-2,-5,-8,-11]
        sway=swing[min(local,len(swing)-1)]*(1 if shot==0 else -1)
        # Small rotation and counter-motion produce separate springy hair motion.
        rotation=cv2.getRotationMatrix2D((SW/2,0),sway*0.12,1)
        rotation[0,2]+=sway
        moving=cv2.warpAffine(hairs[shot],rotation,(SW,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
        over(frame,moving[:,MARGIN-dx:MARGIN-dx+W],0,0)
        over(frame,ornaments[shot][:,MARGIN-dx:MARGIN-dx+W],0,0)
        if i in (6,19):cv2.imencode('.png',frame)[1].tofile(str(OUT/('normal-check.png' if shot==0 else 'dark-check.png')))
        p.stdin.write(frame.tobytes())
finally:
    p.stdin.close();code=p.wait()
if code:raise RuntimeError(f'ffmpeg exited {code}')
print(output)
