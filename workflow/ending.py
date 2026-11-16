"""Outro: zoom the last face and iris together, enlarge its pupil faster."""
import cv2
import numpy as np
from render import over_rgb, warp_rgba_premultiplied


def homogeneous(m):
    return np.vstack((m, [0, 0, 1]))


def make_ending_frames(spec, shot, dimensions):
    w, h = dimensions
    eye = shot['iris']
    if eye is None:
        raise ValueError('ending requires a final shot with an iris')
    ex, ey = shot['eye_center_source']
    scale = float(shot['scale'])
    dx, sway = shot['camera_dx'][-1], shot['hair_sway'][-1]
    base = np.float32([[scale, 0, w/2-ex*scale+dx], [0, scale, h/2-ey*scale+float(shot.get('camera_dy', 0))]])
    rot = cv2.getRotationMatrix2D(tuple(shot['hair_pivot']), float(sway)*float(shot.get('rotation_degrees_per_sway', .1)), 1)
    rot[0, 2] += float(sway)/scale
    registration = np.asarray(shot.get('hair_registration', [1,0,0,0,1,0]), np.float64).reshape(2,3)
    ih, iw = eye.shape[:2]
    count = int(spec['frames'])
    for f in range(count):
        # Reach the black screen faster but retain the full audio duration.
        t = min(1.0, f/20)
        zoom = 1 + 17*t*t
        pupil_radius = 700 * (w/540.0) * t**2.8
        iris = eye
        zoom_m = np.float64([[zoom,0,w/2*(1-zoom)],[0,zoom,h/2*(1-zoom)],[0,0,1]])
        iris_m = np.float64([[zoom,0,w/2-(iw-1)/2*zoom],[0,zoom,h/2-(ih-1)/2*zoom]])
        camera = (zoom_m @ homogeneous(base))[:2]
        hair = (zoom_m @ homogeneous(base) @ homogeneous(rot) @ homogeneous(registration))[:2]
        frame = np.full((h,w,3), 250, np.uint8)
        over_rgb(frame, warp_rgba_premultiplied(iris,iris_m,(w,h)),0,0)
        over_rgb(frame, warp_rgba_premultiplied(shot['face'],camera,(w,h)),0,0)
        over_rgb(frame, warp_rgba_premultiplied(shot['hair'],hair,(w,h)),0,0)
        if shot['accessories'] is not None:
            acc = hair if shot.get('accessories_follow_hair',True) else camera
            over_rgb(frame, warp_rgba_premultiplied(shot['accessories'],acc,(w,h)),0,0)
        # The pupil is deliberately above every face, hair and accessory layer.
        if pupil_radius > 0:
            cv2.circle(frame, (w//2, h//2), max(1, round(pupil_radius)), (0,0,0), -1, cv2.LINE_AA)
        yield frame
