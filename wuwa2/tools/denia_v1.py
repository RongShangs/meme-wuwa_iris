"""Render the Denia two-form eye-alignment test from generated portraits."""
from pathlib import Path
import json
import subprocess
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
W, H, FPS, FRAMES = 540, 788, 30, 90
OUT = ROOT / 'deliverables' / 'denia-v1'
OUT.mkdir(parents=True, exist_ok=True)
# Coordinates and radii are measured in each source image; refine after visual review.
SHOTS = [('normal', 414, 737, 40), ('dark', 675, 737, 40)]
target_r = 52
aligned = []
for name, x, y, radius in SHOTS:
    path = ROOT / 'assets' / 'denia-v1' / f'{name}-full.png'
    source = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    scale = target_r / radius
    transform = np.array([[scale, 0, W/2-x*scale], [0, scale, H/2-y*scale]], dtype=np.float32)
    frame = cv2.warpAffine(source, transform, (W,H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imencode('.png',frame)[1].tofile(str(OUT / f'{name}-aligned.png'))
    patch = source[y-radius:y+radius, x-radius:x+radius]
    yy, xx = np.ogrid[:radius*2,:radius*2]
    dist = np.sqrt((xx-radius)**2 + (yy-radius)**2)
    mask = np.clip((radius-dist)*255, 0, 255).astype(np.uint8)
    cv2.imencode('.png',np.dstack((patch, mask)))[1].tofile(str(OUT / f'{name}-iris-rough.png'))
    aligned.append(frame)

audio = json.loads((ROOT/'project.json').read_text(encoding='utf-8'))['source']
output = OUT/'denia-v1.mp4'
cmd = ['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-', '-i', audio, '-map','0:v','-map','1:a:0?', '-t',str(FRAMES/FPS), '-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(output)]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
try:
    for i in range(FRAMES):
        if i < 24:
            frame = np.full((H,W,3), 250, np.uint8)
            r = max(4,round(target_r*(0.3 + 0.7*i/23)))
            crop = aligned[0][H//2-target_r:H//2+target_r, W//2-target_r:W//2+target_r]
            iris = cv2.resize(crop, (2*r,2*r), interpolation=cv2.INTER_AREA)
            mask = np.zeros((2*r,2*r),np.uint8)
            cv2.circle(mask,(r,r),r-1,255,-1,cv2.LINE_AA)
            a = (mask.astype(np.float32)/255)[:,:,None]
            roi = frame[H//2-r:H//2+r, W//2-r:W//2+r]
            roi[:] = (roi*(1-a)+iris*a).astype(np.uint8)
        else:
            frame = aligned[0 if i < 55 else 1]
        proc.stdin.write(frame.tobytes())
finally:
    proc.stdin.close()
    code = proc.wait()
if code:
    raise RuntimeError(f'ffmpeg failed: {code}')
print(output)
