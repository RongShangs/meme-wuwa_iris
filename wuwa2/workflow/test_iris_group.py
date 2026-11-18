"""Regression checks for rigid eye movement and existing single-eye recipes."""
import unittest

import cv2
import numpy as np

from render import over_irises, over_rgb, over_shifted_iris, warp_rgba_premultiplied


class IrisGroupTests(unittest.TestCase):
    def test_eye_backing_moves_with_window_and_iris_stays_screen_anchored(self):
        group=np.zeros((80,160,4),np.uint8)
        group[40,60]=(20,40,200,255)
        window=np.zeros_like(group);window[30:51,45:66,3]=255
        shot={'iris_layer':group,'iris_clip':window,'eye_center_source':[80,40],
              'scale':1,'eye_background_bgr':[226,231,253]}
        for dx in (0,10):
            frame=np.full((80,160,3),250,np.uint8)
            over_irises(frame,shot,0,np.float32([[1,0,dx],[0,1,0]]))
            np.testing.assert_array_equal(frame[40,46+dx],[226,231,253])
            np.testing.assert_array_equal(frame[40,60],[20,40,200])
            np.testing.assert_array_equal(frame[40,90],[250,250,250])
    def test_eye_windows_move_but_plate_pattern_stays_screen_anchored(self):
        group=np.full((80,160,4),(0,20,200,255),np.uint8)
        group[:,:,0]=np.arange(160,dtype=np.uint8)[None,:]
        window=np.zeros_like(group)
        for center in ((45,40),(115,40)):
            cv2.circle(window,center,7,(255,255,255,255),-1)
        shot={'iris_layer':group,'iris_clip':window,'eye_center_source':[80,40],'scale':1}
        for dx in (0,10):
            frame=np.full((80,160,3),250,np.uint8)
            camera=np.float32([[1,0,dx],[0,1,0]])
            over_irises(frame,shot,0,camera)
            for cx in (45+dx,115+dx):
                self.assertEqual(int(frame[40,cx,0]),cx)
            np.testing.assert_array_equal(frame[40,80],[250,250,250])

    def test_pair_stays_fixed_while_face_moves_and_eyelids_still_occlude(self):
        face = np.full((80, 160, 4), (40, 60, 80, 255), dtype=np.uint8)
        group = np.zeros_like(face)
        for center in ((45, 40), (115, 40)):
            cv2.circle(face, center, 7, (0, 0, 0, 0), -1)
            cv2.circle(group, center, 12, (200, 20, 200, 255), -1)
        shot = {'iris_layer': group, 'eye_center_source': [80, 40], 'scale': 1}
        reference = None
        for dx in (-9, 0, 11):
            with self.subTest(dx=dx):
                frame = np.full((80, 160, 3), 250, dtype=np.uint8)
                camera = np.float32([[1, 0, dx], [0, 1, 0]])
                over_irises(frame, shot, 0)
                if reference is None:
                    reference = frame.copy()
                np.testing.assert_array_equal(frame, reference)
                moving_face = warp_rgba_premultiplied(face, camera, (160, 80))
                over_rgb(frame, moving_face, 0, 0)
                visible = ((frame[:, :, 0] > 150) & (frame[:, :, 1] < 40)
                           & (frame[:, :, 2] > 150)).astype(np.uint8)
                count, _, _, centers = cv2.connectedComponentsWithStats(visible)
                self.assertEqual(count, 3)
                self.assertTrue(np.all(moving_face[:, :, 3][visible > 0] == 0))
                np.testing.assert_allclose(centers[2] - centers[1], [70, 0])

    def test_legacy_single_iris_keeps_its_screen_anchoring(self):
        iris = np.full((13, 13, 4), (180, 40, 170, 220), dtype=np.uint8)
        expected = np.full((80, 160, 3), 250, dtype=np.uint8)
        actual = expected.copy()
        over_shifted_iris(expected, iris, 3.25)
        over_irises(actual, {'iris': iris}, 3.25)
        np.testing.assert_array_equal(actual, expected)

    def test_pair_registration_preserves_spacing_under_shared_screen_shift(self):
        group = np.zeros((30, 120, 4), dtype=np.uint8)
        for center in ((20, 14), (100, 14)):
            cv2.circle(group, center, 5, (200, 20, 200, 255), -1)
        shot = {'iris_layer': group,
                'scale': 2, 'eye_center_source': [45, 25],
                'iris_layer_registration': [0.5, 0, 10, 0, 0.5, 20]}
        for dx in (0, 10):
            frame = np.full((100, 180, 3), 250, dtype=np.uint8)
            over_irises(frame, shot, dx)
            mask = ((frame[:, :, 0] > 150) & (frame[:, :, 1] < 40)
                    & (frame[:, :, 2] > 150)).astype(np.uint8)
            count, _, _, centers = cv2.connectedComponentsWithStats(mask)
            self.assertEqual(count, 3)
            np.testing.assert_allclose(centers[1:], [[40+dx, 54], [120+dx, 54]])


if __name__ == '__main__':
    unittest.main()
