"""Open the four requested side frames and joined bottoms, preserving upper lashes."""
from pathlib import Path
import cv2
import numpy as np
from render import image
from new_build import redrawn_eye_centers, save

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'角色/椿/素材/q2-v6/第二形态-脸部重生原图.png'
OUTPUT=ROOT/'角色/椿/素材/q2-v6/第二形态-眼眶清线.png'


def main():
    a=image(SOURCE);centers,stats=redrawn_eye_centers(a)
    _,labels=cv2.connectedComponents((a[:,:,3]<128).astype(np.uint8))
    result=a.copy()
    for (cx,cy),(x,y,w,h,_) in zip(centers,stats):
        x0=max(0,x-12);x1=min(a.shape[1],x+w+12)
        y0=max(0,y-2);y1=min(a.shape[0],y+h+12)
        hole=(labels[y0:y1,x0:x1]==labels[cy,cx]).astype(np.uint8)*255
        expanded=cv2.dilate(hole,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))
        # The upper lash is an intentional part of the character: only open
        # side/bottom strokes, gradually starting beneath the first 8 pixels.
        gate=np.clip((np.arange(y0,y1)-(y+8))/4,0,1)[:,None]
        expanded=np.maximum(hole,expanded*gate).astype(np.uint8)
        expanded=cv2.GaussianBlur(expanded,(3,3),.45)
        result[y0:y1,x0:x1,3]=np.minimum(result[y0:y1,x0:x1,3],255-expanded)
    assert np.array_equal(result[:,:,:3],a[:,:,:3])
    save(OUTPUT,result)
    print('Saved requested eye-frame alpha cleanup:',OUTPUT)


if __name__=='__main__':main()
