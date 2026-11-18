"""Clear the remaining neutral-white sclera connected to Chun's eye holes."""
from pathlib import Path
import cv2
import numpy as np
from render import image
from new_build import redrawn_eye_centers,save

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'角色/椿/素材/q2-v6/第二形态-眼眶清线.png'
OUTPUT=ROOT/'角色/椿/素材/q2-v7/第二形态-眼白清理.png'


def main():
    a=image(SOURCE);result=a.copy();centers,stats=redrawn_eye_centers(a)
    for (cx,cy),(x,y,w,h,_) in zip(centers,stats):
        x0=max(0,x-45);x1=min(a.shape[1],x+w+45)
        y0=max(0,y-3);y1=min(a.shape[0],y+h+18)
        roi=a[y0:y1,x0:x1];rgb=roi[:,:,:3].astype(np.int16)
        neutral=(rgb.min(axis=2)>205)&((rgb.max(axis=2)-rgb.min(axis=2))<18)
        possible=((roi[:,:,3]<200)|neutral).astype(np.uint8)
        _,labels=cv2.connectedComponents(possible)
        component=labels[cy-y0,cx-x0]
        if not component:raise ValueError('Eye center is not transparent')
        mask=(labels==component).astype(np.uint8)*255
        # Do not remove the upper eyelash even where it adjoins the white.
        mask[:max(0,y+4-y0)]=0
        mask=cv2.GaussianBlur(mask,(3,3),.45)
        result[y0:y1,x0:x1,3]=np.minimum(result[y0:y1,x0:x1,3],255-mask)
    assert np.array_equal(a[:,:,:3],result[:,:,:3])
    save(OUTPUT,result)
    print('Cleared remaining connected eye whites:',OUTPUT)


if __name__=='__main__':main()
