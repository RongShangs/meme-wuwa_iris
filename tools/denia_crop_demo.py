"""Draft alignment proof using existing fan-art reference, not final character art."""
from pathlib import Path
import json
import subprocess
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / '素材' / '参考原始文件' / '达尼亚历史参考.png'
OUT = ROOT / 'deliverables'
OUT.mkdir(exist_ok=True)
W, H, FPS = 540, 788, 30
# Manually estimated centres and iris radii in the original reference image.
SHOTS = [dict(name='normal', cx=145, cy=280, radius=11.5),
         dict(name='dark', cx=1526, cy=293, radius=11.5)]
TARGET_RADIUS = 48
img = cv2.imdecode(np.frombuffer(SOURCE.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(SOURCE)
frames = []
for s in SHOTS:
    scale = TARGET_RADIUS / s['radius']
    matrix = np.array([[scale, 0, W/2-s['cx']*scale], [0, scale, H/2-s['cy']*scale]], dtype=np.float32)
    frame = cv2.warpAffine(img, matrix, (W,H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imencode('.png',frame)[1].tofile(str(OUT / (s['name']+'-aligned.png')))
    # The circular alpha is a rough iris patch; lids/highlights inside it remain.
    r = int(round(s['radius']))
    cx, cy = s['cx'], s['cy']
    patch = img[cy-r:cy+r+1,cx-r:cx+r+1]
    yy, xx = np.ogrid[-r:r+1,-r:r+1]
    alpha = np.clip((r-np.sqrt(xx*xx+yy*yy))*255,0,255).astype(np.uint8)
    rgba = np.dstack([patch,alpha])
    cv2.imencode('.png',rgba)[1].tofile(str(OUT / (s['name']+'-iris-draft.png')))
    frames.append(frame)
source_video = str(ROOT / json.loads((ROOT/'project.json').read_text(encoding='utf-8'))['source'])
output = OUT/'denia-alignment-draft.mp4'
cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-', '-i',source_video,'-map','0:v','-map','1:a:0?','-t','3','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(output)]
p = subprocess.Popen(cmd,stdin=subprocess.PIPE)
try:
    for i in range(90):
        if i < 24:
            frame = np.full((H,W,3), 250, np.uint8)
            # An iris-sized sample grows into the common centre as an entrance.
            radius = max(3,round(TARGET_RADIUS*(0.35+0.65*i/23)))
            iris = cv2.resize(frames[0][H//2-TARGET_RADIUS:H//2+TARGET_RADIUS, W//2-TARGET_RADIUS:W//2+TARGET_RADIUS],(2*radius,2*radius))
            mask = np.zeros((2*radius,2*radius),np.uint8)
            cv2.circle(mask,(radius,radius),radius-1,255,-1,cv2.LINE_AA)
            roi=frame[H//2-radius:H//2+radius,W//2-radius:W//2+radius]
            a=(mask.astype(np.float32)/255)[:,:,None]
            roi[:]=(roi*(1-a)+iris*a).astype(np.uint8)
        else:
            frame=frames[0 if i<55 else 1]
        p.stdin.write(frame.tobytes())
finally:
    p.stdin.close()
    if p.wait(): raise RuntimeError('ffmpeg failed')
print(output)
