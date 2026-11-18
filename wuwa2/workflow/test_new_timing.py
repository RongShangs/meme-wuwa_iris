"""Timing and iris regressions for the music-aligned montage."""
import json
from pathlib import Path
import unittest
import wave
import cv2
import numpy as np
from new_timing import plan
from render import image

ROOT=Path(__file__).resolve().parents[1]


class TimingTest(unittest.TestCase):
    def test_every_cut_falls_within_half_a_video_frame_of_a_beat(self):
        manifest=json.loads((ROOT/'workflow/new-角色清单.json').read_text(encoding='utf-8'))
        timing=plan(manifest)
        beat_grid=timing['opening_frames']/30+np.arange(17)*60/timing['bpm']
        self.assertLessEqual(np.max(abs(np.array(timing['starts'])/30-beat_grid)),1/60+1e-8)
        self.assertEqual(timing['opening_frames']+sum(timing['lengths'])+timing['ending_frames'],timing['total_frames'])
        self.assertGreater(len(set(timing['lengths'])),1)
        with wave.open(str(ROOT/manifest['music_timing']['prepared_audio']),'rb') as f:
            self.assertEqual(f.getnframes(),round(timing['total_frames']/30*f.getframerate()))

    def test_chun_complete_iris_is_full_width_and_includes_its_outer_rim(self):
        manifest=json.loads((ROOT/'workflow/new-角色清单.json').read_text(encoding='utf-8'))
        spec=next(d for d in manifest['characters'] if d['name']=='椿')['second']
        iris=image(ROOT/spec['iris_generated'])
        _,labels,stats,_=cv2.connectedComponentsWithStats((iris[:,:,3]>128).astype(np.uint8))
        i=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);x,y,w,h,_=stats[i]
        self.assertGreater(h,.9*w)
        self.assertLess(h,1.15*w)
        solid=(labels==i).astype(np.uint8)
        band=(solid>0)&(cv2.erode(solid,np.ones((11,11),np.uint8))==0)&(iris[:,:,3]>200)
        self.assertLess(np.median(iris[band,:3].max(axis=1)),110)

    def test_chun_character_and_camera_match_the_adopted_previous_version(self):
        import hashlib
        manifest=json.loads((ROOT/'workflow/new-角色清单.json').read_text(encoding='utf-8'))
        spec=next(d for d in manifest['characters'] if d['name']=='椿')['second']
        locked=spec.get('character_lock')
        source=ROOT/spec['generated_cel']
        if locked:
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),locked['rgba_file_sha256'])
        config=ROOT/'角色/椿/new-项目.json'
        shot=json.loads(config.read_text(encoding='utf-8'))['shots'][1]
        np.testing.assert_array_equal(image(source),image(config.parent/shot['face']))
        if locked:
            for key in ('scale','camera_dy','camera_dx','eye_center_source','iris_centers_source'):
                self.assertEqual(shot[key],locked[key],key+' unexpectedly changed on the character')
        else:
            face=image(source)
            for x,y in shot['iris_centers_source']:
                self.assertLessEqual(face[y,x,3],16)

    def test_chun_iris_preserves_original_low_division_and_pupil_proportions(self):
        manifest=json.loads((ROOT/'workflow/new-角色清单.json').read_text(encoding='utf-8'))
        spec=next(d for d in manifest['characters'] if d['name']=='椿')['second']
        iris=image(ROOT/spec['iris_generated'])
        opaque=iris[:,:,3]>128
        yy,xx=np.where(opaque);x0,y0,x1,y1=xx.min(),yy.min(),xx.max(),yy.max()
        hsv=cv2.cvtColor(iris[:,:,:3],cv2.COLOR_BGR2HSV)
        red=opaque&((hsv[:,:,0]<15)|(hsv[:,:,0]>170))&(hsv[:,:,1]>65)&(hsv[:,:,2]>35)
        _,labels,stats,centers=cv2.connectedComponentsWithStats(red.astype(np.uint8))
        candidates=[i for i in range(1,len(stats)) if stats[i,4]>100 and abs(centers[i,0]-(x0+x1)/2)<(x1-x0)*.15]
        self.assertTrue(candidates,'Missing original reddish-brown pupil ring')
        i=max(candidates,key=lambda i:stats[i,4]);cx,cy=centers[i];_,_,rw,rh,_=stats[i]
        self.assertGreater((cy-y0)/(y1-y0),.58)
        self.assertLess((cy-y0)/(y1-y0),.71)
        self.assertGreater(rh,.90*rw)
        self.assertLess(rh,1.15*rw)
        b,g,r=iris[round(cy),round(cx),:3].astype(float)
        self.assertGreater(min(b,g,r),190)
        self.assertGreaterEqual(b,r-5)
        x,y,w,h,_=stats[i]
        patch=iris[y:y+h,x:x+w,:3]
        pale=((patch.min(axis=2)>190)&(patch.max(axis=2)-patch.min(axis=2)<45)).astype(np.uint8)
        _,core_labels,core_stats,_=cv2.connectedComponentsWithStats(pale)
        core_id=core_labels[round(cy-y),round(cx-x)]
        self.assertGreater(core_id,0)
        core_w,core_h=core_stats[core_id,2:4]
        self.assertGreater(core_h,1.15*core_w)
        self.assertLess(core_h,1.60*core_w)
        row=opaque[round(y0+.985*(y1-y0)),x0:x1+1]
        self.assertGreater(np.count_nonzero(row),.40*(x1-x0),'Bottom became a semicircle instead of the original flatter contour')


if __name__=='__main__':unittest.main()
