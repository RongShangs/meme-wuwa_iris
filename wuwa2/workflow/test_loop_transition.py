"""Behavior checks for the nonblack waveform loop and both vertical pupils."""
from pathlib import Path
import copy
import unittest
from itertools import islice
import cv2
import numpy as np
from render import load_project, warp_rgba_premultiplied, over_rgb
from loop_transition import endpoint, make_opening, make_ending, pupil_mask, resonance_background, smooth

ROOT=Path(__file__).resolve().parents[1]


class LoopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        manifest=json.loads((ROOT/'workflow/new-角色清单.json').read_text(encoding='utf-8'))
        path=ROOT/'workflow'/(manifest['config_name']+'.json')
        cls.cfg,cls.shots,*_=load_project(path)
        cls.spec=copy.deepcopy(cls.cfg['opening'])
        for key in ('mark_path','pupil_reference','last_iris_reference','pattern_reference'):
            if cls.spec.get(key):cls.spec[key]=str((path.parent/cls.spec[key]).resolve())

    def test_endpoints_are_same_waveform_without_black_screen(self):
        spec=dict(self.spec,frames=2)
        start=next(make_opening(spec,self.shots[0]['iris'],(1080,1576)))
        ending=dict(frames=2,loop=spec)
        end=list(make_ending(ending,self.shots[-1],(1080,1576)))[-1]
        np.testing.assert_array_equal(start,end)
        self.assertGreater(start.std(),15)
        self.assertGreater(start.mean(),30)
        y,x=np.where(start.max(axis=2)<15)
        self.assertLess(abs((x.min()+x.max())/2-540),2)
        self.assertLess(abs((y.min()+y.max())/2-788),2)
        # Thresholding the antialiased, very fine tips shortens the dark core.
        self.assertLess(abs(np.ptp(y)-self.spec['mark_height']),16)

    def test_actual_endpoint_pupils_are_vertical(self):
        for shot in (self.shots[0],self.shots[-1]):
            mask,_=pupil_mask(shot['iris'])
            y,x=np.where(mask>0)
            self.assertGreater(np.ptp(y),3*np.ptp(x))

    def test_three_flat_purple_layers_are_present_at_loop_start(self):
        if not self.spec.get('layered_iris_background'):
            self.skipTest('This version does not use layered teeth')
        frame=resonance_background(self.spec,(1080,1576))
        for color in ([136,51,109],[182,83,146],[238,149,196]):
            self.assertGreater(np.count_nonzero(np.all(frame==color,axis=2)),10000)

    def test_whole_iris_contracts_from_outside_canvas(self):
        frames=list(make_opening(dict(self.spec,frames=5),self.shots[0]['iris'],(1080,1576)))
        self.assertFalse(np.all(frames[0][0,0]==250))
        self.assertTrue(np.all(frames[2][0,0]==250))
        widths=[]
        for frame in (frames[2],frames[4]):
            x=np.where(np.any(frame[788]!=250,axis=1))[0]
            widths.append(np.ptp(x))
        self.assertGreater(widths[0],widths[1]*3)

    def test_pupil_and_rim_have_same_scale_after_early_handoff(self):
        spec=dict(self.spec,frames=6)
        frames=list(make_opening(spec,self.shots[0]['iris'],(1080,1576)))
        ratios=[]
        initial=np.hypot(1080,1576)*1.8;final=spec['final_iris_size']
        for i in (3,5):
            t=i/5;diameter=final+(initial-final)*(1-smooth(t))**2.2
            x0,x1=round(540-.09*diameter),round(540+.09*diameter)
            y0,y1=round(788-.37*diameter),round(788+.37*diameter)
            mask=frames[i][y0:y1,x0:x1].max(axis=2)<100
            y,x=np.where(mask)
            ratios.append(np.ptp(y)/diameter)
        self.assertLess(abs(ratios[0]-ratios[1]),.025)

    def test_outro_overlay_does_not_cross_still_visible_lower_lid(self):
        shot=self.shots[-1];w,h=1080,1576;index=14;t=index/28
        frame=next(islice(make_ending(dict(frames=29,loop=self.spec),shot,(w,h)),index,None))
        zoom=1+2*float(smooth(t/.75));scale=shot['scale'];ex,ey=shot['eye_center_source']
        base=np.float64([[scale,0,w/2-ex*scale+shot['camera_dx'][-1]],
                         [0,scale,h/2-ey*scale+shot.get('camera_dy',0)],[0,0,1]])
        z=np.float64([[zoom,0,w/2*(1-zoom)],[0,zoom,h/2*(1-zoom)],[0,0,1]])
        camera=(z@base)[:2]
        window=cv2.warpAffine(shot['iris_clip'][:,:,3],camera,(w,h))
        x=w//2;y=int(np.where(window[:,x]>127)[0].max())+12
        self.assertLess(y,h)
        scene=np.full((h,w,3),250,np.uint8)
        for key in ('face','hair','accessories'):
            if shot.get(key) is not None:
                over_rgb(scene,warp_rgba_premultiplied(shot[key],camera,(w,h)),0,0)
        fade=float(smooth((t-.10)/.56))
        expected=cv2.addWeighted(scene,1-fade,resonance_background(self.spec,(w,h)),fade,0)
        np.testing.assert_allclose(frame[y-2:y+3,x-2:x+3],expected[y-2:y+3,x-2:x+3],atol=3)


if __name__=='__main__':unittest.main()
