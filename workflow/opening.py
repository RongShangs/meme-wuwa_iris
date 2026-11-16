"""Opening sequence: shrinking black mark and red-gold iris."""
import cv2
import numpy as np
from pathlib import Path
from render import image, over_rgb, warp_rgba_premultiplied


def smooth(t):
    t = min(1.0, max(0.0, t))
    return t*t*(3-2*t)


def interp(frame, keys):
    for (a, x), (b, y) in zip(keys, keys[1:]):
        if frame <= b:
            return x + (y-x)*smooth((frame-a)/(b-a))
    return keys[-1][1]


def iris_texture(actual):
    side = 512
    y, x = np.mgrid[:side, :side].astype(np.float32)
    nx, ny = (x-(side-1)/2)/(side/2), (y-(side-1)/2)/(side/2)
    rad = np.sqrt(nx*nx+ny*ny)
    angle = np.arctan2(ny, nx)
    gold = np.clip((ny+0.52)*0.95, 0, 1)
    dark = np.clip(1.05 - .34*rad + .045*np.sin(angle*34+rad*12), .45, 1)
    # OpenCV BGR, burgundy upper half and amber lower half.
    top = np.array([35., 19., 136.], np.float32)
    bottom = np.array([25., 150., 250.], np.float32)
    rgb = (top[None,None,:]*(1-gold[:,:,None])+bottom[None,None,:]*gold[:,:,None])*dark[:,:,None]
    alpha = np.clip((1-rad)*side/3, 0, 1)*255
    art = np.dstack((rgb, alpha)).astype(np.uint8)
    iris = cv2.resize(actual, (side, side), interpolation=cv2.INTER_CUBIC)
    return art, iris


def make_opening_frames(spec, iris, dimensions):
    width, height = dimensions
    resolution_scale = width / 540.0
    mark = image(Path(spec['mark_path']))
    artwork, target = iris_texture(iris)
    alpha = mark[:,:,3]
    ys, xs = np.where(alpha > 64)
    cx, cy = (xs.min()+xs.max())/2, (ys.min()+ys.max())/2
    # Work from the original high-resolution iris so enlarged slit edges stay smooth.
    if spec.get('pupil_reference'):
        original = image(Path(spec['pupil_reference']))
        oy, ox = np.where(original[:,:,3] > 48)
        crop = original[oy.min():oy.max()+1, ox.min():ox.max()+1]
        side = max(crop.shape[:2])
        square = np.zeros((side, side, 4), np.uint8)
        y0, x0 = (side-crop.shape[0])//2, (side-crop.shape[1])//2
        square[y0:y0+crop.shape[0], x0:x0+crop.shape[1]] = crop
        pupil_source = cv2.resize(square, (1024,1024), interpolation=cv2.INTER_CUBIC)
    else:
        pupil_source = cv2.resize(target, (1024,1024), interpolation=cv2.INTER_CUBIC)
    b, g, r, a = cv2.split(pupil_source)
    candidate = ((r < 75) & (b < 38) & (g < 18) & (a > 200)).astype(np.uint8)
    candidate[:180] = 0
    candidate[850:] = 0
    candidate[:, :440] = 0
    candidate[:, 584:] = 0
    _, labels = cv2.connectedComponents(candidate)
    component = labels[512,512]
    if component == 0:
        raise ValueError('Could not identify first character iris pupil')
    pupil_mask = (labels == component).astype(np.uint8)
    def signed_distance(mask):
        return cv2.distanceTransform(mask, cv2.DIST_L2, 5) - cv2.distanceTransform(1-mask, cv2.DIST_L2, 5)
    count = int(spec.get('frames', 30))
    for f in range(count):
        if f == 0:
            yield np.zeros((height,width,3), np.uint8)
            continue
        # Both start shrinking together; a short gentle passage retains motion.
        diameter = (2200 - (2200-125) * (f / max(1, count-1))**2.8) * resolution_scale
        if f <= 13:
            mark_scale = interp(f, [(0,6.0),(9,.49),(13,.45)])
        elif f <= 21:
            # Keep the waveform intact while it accelerates out of the pause.
            q = (f-13)/8
            mark_scale = .45 - .13*q**2.2
        else:
            # Settle to its final scale during the subsequent shape morph.
            q = (f-21)/max(1, count-1-21)
            mark_scale = .32 - (.32-.075)*(1-(1-q)**2)
        mark_scale *= resolution_scale
        blend = smooth((f-24)/5)
        # Rotate only the temporary colored background; the reference iris and
        # pupil remain registered while this background fades out.
        rotation_t = min(1.0, f/28)
        degrees = 40 * (1-(1-rotation_t)**3)
        bg_rotation = cv2.getRotationMatrix2D((255.5,255.5), degrees, 1.0)
        rotating_art = warp_rgba_premultiplied(artwork, bg_rotation, (512,512))
        cel = cv2.addWeighted(rotating_art, 1-blend, target, blend, 0)
        sc = diameter/512
        matrix = np.float32([[sc,0,width/2-255.5*sc],[0,sc,height/2-255.5*sc]])
        frame = np.full((height,width,3), 255, dtype=np.uint8)
        over_rgb(frame, warp_rgba_premultiplied(cel,matrix,(width,height)),0,0)
        # Morph in screen coordinates: the destination is the actual pupil,
        # transformed by exactly the same matrix as the shrinking iris.
        morph = smooth((f-21)/6)
        wm = np.float32([[mark_scale,0,width/2-cx*mark_scale],[0,mark_scale,height/2-cy*mark_scale]])
        mark_layer = warp_rgba_premultiplied(mark,wm,(width,height))
        if morph > 0:
            pupil_matrix = np.float32([[sc/2,0,width/2-511.5*sc/2],[0,sc/2,height/2-511.5*sc/2]])
            if morph < 1:
                source_mask = (mark_layer[:,:,3] > 127).astype(np.uint8)
                target_mask = cv2.warpAffine(pupil_mask, pupil_matrix, (width,height), flags=cv2.INTER_NEAREST)
                signed = (1-morph)*signed_distance(source_mask) + morph*signed_distance(target_mask)
                mark_layer[:,:,3] = (np.clip(signed + .5, 0, 1)*255).astype(np.uint8)
            else:
                target_alpha = cv2.warpAffine(pupil_mask*255, pupil_matrix, (width,height), flags=cv2.INTER_LINEAR)
                mark_layer[:,:,3] = target_alpha
            mark_layer[:,:,3] = cv2.GaussianBlur(mark_layer[:,:,3], (0,0), .7)
        # Quickly match the central dark-red pupil so the emblem merges into it.
        pupil_bgr = iris[iris.shape[0]//2, iris.shape[1]//2, :3].astype(np.float32)
        tint = smooth((f-1)/8)
        mark_layer[:, :, :3] = (mark_layer[:, :, :3].astype(np.float32)*(1-tint) + pupil_bgr[None,None,:]*tint).astype(np.uint8)
        opacity = 1-smooth((f-26)/3)
        mark_layer[:,:,3] = (mark_layer[:,:,3].astype(np.float32)*opacity).astype(np.uint8)
        over_rgb(frame,mark_layer,0,0)
        yield frame
