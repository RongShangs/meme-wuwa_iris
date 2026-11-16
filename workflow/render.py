"""Reusable transparent-cel eye montage renderer; paths are relative to config."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def image(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    result = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if result is None or result.ndim != 3 or result.shape[2] != 4:
        raise ValueError(f'Expected a 4-channel transparent PNG: {path}')
    return result


def rgba_over(dst, src):
    da = dst[:, :, 3:4].astype(np.float32) / 255
    sa = src[:, :, 3:4].astype(np.float32) / 255
    oa = sa + da * (1 - sa)
    numerator = src[:, :, :3].astype(np.float32) * sa + dst[:, :, :3].astype(np.float32) * da * (1 - sa)
    rgb = np.divide(numerator, oa, out=np.zeros_like(numerator), where=oa > 0)
    return np.dstack((np.clip(rgb, 0, 255).astype(np.uint8), np.clip(oa[:, :, 0] * 255, 0, 255).astype(np.uint8)))


def over_rgb(dst, src, x, y):
    h, w = src.shape[:2]
    height, width = dst.shape[:2]
    left, top, right, bottom = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
    if left >= right or top >= bottom:
        return
    s = src[top-y:bottom-y, left-x:right-x]
    a = s[:, :, 3:4].astype(np.float32) / 255
    dst[top:bottom, left:right] = (dst[top:bottom, left:right] * (1-a) + s[:, :, :3] * a).astype(np.uint8)


def warp_rgba_premultiplied(src, matrix, size):
    """Resample color and alpha together to avoid dark transparent-edge halos."""
    a = src[:, :, 3:4].astype(np.float32) / 255
    premult = np.dstack((src[:, :, :3].astype(np.float32) * a, a[:, :, 0]))
    warped = cv2.warpAffine(premult, matrix, size, flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    wa = np.clip(warped[:, :, 3:4], 0, 1)
    rgb = np.divide(warped[:, :, :3], wa, out=np.zeros_like(warped[:, :, :3]), where=wa > 1e-5)
    return np.dstack((np.clip(rgb, 0, 255).astype(np.uint8), (wa[:, :, 0] * 255).astype(np.uint8)))


def load_project(config_path):
    config_path = config_path.resolve()
    cfg = json.loads(config_path.read_text(encoding='utf-8-sig'))
    root = config_path.parent
    def resolve(value):
        p = Path(value)
        return p if p.is_absolute() else (root / p).resolve()
    canvas = cfg['canvas']
    w, h, fps = int(canvas['width']), int(canvas['height']), int(canvas['fps'])
    if w <= 0 or h <= 0 or w % 2 or h % 2 or fps <= 0:
        raise ValueError('width and height must be positive even numbers; fps must be positive')
    background = canvas.get('background_bgr', [250, 250, 250])
    if len(background) != 3 or any(not 0 <= int(x) <= 255 for x in background):
        raise ValueError('background_bgr must contain three values from 0 to 255')
    shots = []
    if not cfg.get('shots'):
        raise ValueError('At least one shot is required')
    for spec in cfg['shots']:
        shot = dict(spec)
        if spec.get('type') == 'still':
            shot['face'] = image(resolve(spec['face']))
            sh, sw = shot['face'].shape[:2]
            x, y = map(float, spec['anchor_source'])
            if not (0 <= x < sw and 0 <= y < sh) or float(spec['scale']) <= 0:
                raise ValueError(f"{spec['name']}: invalid still anchor or scale")
            if not spec.get('camera_dx'):
                raise ValueError(f"{spec['name']}: empty camera_dx")
            shots.append(shot)
            continue
        for key in ('face', 'hair'):
            shot[key] = image(resolve(spec[key]))
        shot['iris'] = image(resolve(spec['iris'])) if spec.get('type') != 'masked' else None
        if shot['iris'] is not None and spec.get('iris_size') is not None:
            size = int(spec['iris_size'])
            if size <= 0:
                raise ValueError(f"{spec['name']}: iris_size must be positive")
            shot['iris'] = cv2.resize(shot['iris'], (size, size), interpolation=cv2.INTER_AREA)
        if spec.get('face_cover'):
            cover = image(resolve(spec['face_cover']))
            if cover.shape != shot['face'].shape:
                raise ValueError(f"{spec['name']}: face_cover must share the face canvas")
            shot['face'] = rgba_over(shot['face'], cover)
        shot['accessories'] = image(resolve(spec['accessories'])) if spec.get('accessories') else None
        if shot['face'].shape != shot['hair'].shape or (shot['accessories'] is not None and shot['face'].shape != shot['accessories'].shape):
            raise ValueError(f"{spec['name']}: face, hair and accessories must share the same full canvas")
        if len(spec['camera_dx']) == 0 or len(spec['camera_dx']) != len(spec['hair_sway']):
            raise ValueError(f"{spec['name']}: camera_dx and hair_sway must have the same nonzero length")
        x, y = map(int, spec['eye_center_source'])
        sh, sw = shot['face'].shape[:2]
        if not (0 <= x < sw and 0 <= y < sh):
            raise ValueError(f"{spec['name']}: eye center lies outside face canvas")
        if spec.get('type') != 'masked' and shot['face'][y, x, 3] > 16:
            raise ValueError(f"{spec['name']}: face is not transparent at the specified eye center")
        if float(spec['scale']) <= 0:
            raise ValueError(f"{spec['name']}: scale must be positive")
        if 'hair_registration' in spec and (len(spec['hair_registration']) != 6 or not np.isfinite(spec['hair_registration']).all()):
            raise ValueError(f"{spec['name']}: hair_registration must contain six finite affine values")
        shots.append(shot)
    audio = cfg.get('audio')
    if audio is not None:
        audio = dict(audio)
        audio['path'] = resolve(audio['path'])
        if not audio['path'].is_file():
            raise FileNotFoundError(f"Audio source not found: {audio['path']}")
        if float(audio.get('start_seconds', 0)) < 0:
            raise ValueError('audio.start_seconds must be non-negative')
    return cfg, shots, audio, resolve(cfg['output']), (w, h, fps, background)


def camera_swing(frame, shot, index, amplitude):
    if not amplitude:
        return frame
    # Direction follows entry, then eases back with a barely perceptible rebound.
    profile = [0.25, 1.0, 0.72, 0.45, 0.22, 0.07, -0.05, -0.04, -0.02, 0.0]
    weight = profile[index] if index < len(profile) else 0.0
    if not weight:
        return frame
    direction = 1 if shot['camera_dx'][0] > 0 else -1
    h, w = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((w/2, h/2), direction*float(amplitude)*weight, 1.0)
    return cv2.warpAffine(frame, matrix, (w,h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(250,250,250))


def camera_shift_offset(shot, index, amplitude):
    if not amplitude:
        return 0.0
    count = len(shot['camera_dx'])
    if index == 0 or count < 3:
        return 0.0
    t = min(1.0, (index - 1) / (count - 2))
    weight = 1.0 - t*t*(3.0 - 2.0*t)
    direction = -1 if shot['camera_dx'][0] > 0 else 1
    return direction*float(amplitude)*weight


def over_shifted_iris(frame, iris, dx):
    h, w = frame.shape[:2]
    ih, iw = iris.shape[:2]
    x, y = w//2-iw//2, h//2-ih//2
    if not dx:
        over_rgb(frame, iris, x, y)
    else:
        transform = np.float32([[1,0,x+dx],[0,1,y]])
        over_rgb(frame, warp_rgba_premultiplied(iris, transform, (w,h)), 0, 0)


def render(shots, audio, output, dimensions, opening=None, ending=None, swing=0, shift=0):
    w, h, fps, background = dimensions
    intro_frames = int(opening.get('frames', 0)) if opening else 0
    total = sum(len(s['camera_dx']) for s in shots) + intro_frames + (int(ending.get('frames', 0)) if ending else 0)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-']
    if audio:
        cmd += ['-ss', str(audio.get('start_seconds', 0)), '-i', str(audio['path']), '-map', '0:v', '-map', '1:a:0?']
    else:
        cmd += ['-map', '0:v']
    cmd += ['-t', str(total/fps), '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p']
    if audio:
        cmd += ['-c:a', 'aac']
    cmd += ['-movflags', '+faststart', str(output)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        if opening:
            from opening import make_opening_frames
            for frame in make_opening_frames(opening, shots[0]['iris'], (w, h)):
                proc.stdin.write(frame.tobytes())
        for shot_index, shot in enumerate(shots):
            if shot.get('type') == 'still':
                ex, ey = shot['anchor_source']
                scale = float(shot['scale'])
                for frame_index, dx in enumerate(shot['camera_dx']):
                    shift_dx = camera_shift_offset(shot, frame_index, shift)
                    camera = np.float32([[scale, 0, w/2-ex*scale+float(dx)+shift_dx], [0, scale, h/2-ey*scale+float(shot.get('camera_dy', 0))]])
                    frame = np.full((h, w, 3), background, dtype=np.uint8)
                    over_rgb(frame, warp_rgba_premultiplied(shot['face'], camera, (w, h)), 0, 0)
                    frame = camera_swing(frame, shot, frame_index, swing)
                    proc.stdin.write(frame.tobytes())
                continue
            ch, cw = shot['face'].shape[:2]
            ih, iw = shot['iris'].shape[:2] if shot['iris'] is not None else (0, 0)
            ex, ey = shot['eye_center_source']
            scale = float(shot['scale'])
            for frame_index, (dx, sway) in enumerate(zip(shot['camera_dx'], shot['hair_sway'])):
                shift_dx = camera_shift_offset(shot, frame_index, shift)
                rotation = cv2.getRotationMatrix2D(tuple(shot['hair_pivot']), float(sway)*float(shot.get('rotation_degrees_per_sway', 0.1)), 1)
                rotation[0, 2] += float(sway)/scale
                camera = np.float32([[scale, 0, w/2-ex*scale+float(dx)+shift_dx], [0, scale, h/2-ey*scale+float(shot.get('camera_dy', 0))]])
                if 'hair_registration' in shot:
                    registration = np.asarray(shot['hair_registration'], np.float64).reshape(2, 3)
                    homogeneous = lambda m: np.vstack((m, [0, 0, 1]))
                    hair_transform = (homogeneous(camera) @ homogeneous(rotation) @ homogeneous(registration))[:2]
                    frame = np.full((h, w, 3), background, dtype=np.uint8)
                    if shot['iris'] is not None:
                        over_shifted_iris(frame, shot['iris'], shift_dx)
                    face_layer = warp_rgba_premultiplied(shot['face'], camera, (w, h))
                    hair_layer = warp_rgba_premultiplied(shot['hair'], hair_transform, (w, h))
                    if opening and shot_index == 0:
                        fade_frames = int(opening.get('face_fade_frames', 5))
                        fade = min(1.0, (frame_index + 1) / max(1, fade_frames))
                        face_layer[:, :, 3] = (face_layer[:, :, 3].astype(np.float32) * fade).astype(np.uint8)
                        hair_layer[:, :, 3] = (hair_layer[:, :, 3].astype(np.float32) * fade).astype(np.uint8)
                    over_rgb(frame, face_layer, 0, 0)
                    over_rgb(frame, hair_layer, 0, 0)
                    if shot['accessories'] is not None:
                        acc_transform = (homogeneous(camera) @ homogeneous(rotation))[:2] if shot.get('accessories_follow_hair', True) else camera
                        over_rgb(frame, warp_rgba_premultiplied(shot['accessories'], acc_transform, (w, h)), 0, 0)
                    frame = camera_swing(frame, shot, frame_index, swing)
                    proc.stdin.write(frame.tobytes())
                    continue
                moving = cv2.warpAffine(shot['hair'], rotation, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
                # Head accessories follow the exact same pivot and movement as hair.
                scene = rgba_over(shot['face'], moving)
                if shot['accessories'] is not None:
                    attached = cv2.warpAffine(shot['accessories'], rotation, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)) if shot.get('accessories_follow_hair', True) else shot['accessories']
                    scene = rgba_over(scene, attached)
                framed = cv2.warpAffine(scene, camera, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
                frame = np.full((h, w, 3), background, dtype=np.uint8)
                if shot['iris'] is not None:
                    over_shifted_iris(frame, shot['iris'], shift_dx)
                over_rgb(frame, framed, 0, 0)
                frame = camera_swing(frame, shot, frame_index, swing)
                proc.stdin.write(frame.tobytes())
        if ending:
            from ending import make_ending_frames
            for frame in make_ending_frames(ending, shots[-1], (w, h)):
                proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        status = proc.wait()
    if status:
        raise RuntimeError(f'ffmpeg exited {status}')
    print(f'Wrote {output} ({total} frames, {fps} fps)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--output', type=Path, help='Override output path')
    parser.add_argument('--validate', action='store_true', help='Check configuration and inputs without rendering')
    args = parser.parse_args()
    if not shutil.which('ffmpeg') and not args.validate:
        parser.error('ffmpeg must be on PATH')
    cfg, shots, audio, output, dimensions = load_project(args.config)
    print(f"Validated {len(shots)} shots / {sum(len(s['camera_dx']) for s in shots)} frames")
    if not args.validate:
        opening = cfg.get('opening')
        if opening:
            opening = dict(opening)
            opening['mark_path'] = str((args.config.resolve().parent / opening['mark_path']).resolve())
            if opening.get('pupil_reference'):
                opening['pupil_reference'] = str((args.config.resolve().parent / opening['pupil_reference']).resolve())
        render(shots, audio, args.output.resolve() if args.output else output, dimensions, opening, cfg.get('ending'), cfg.get('camera_swing_degrees', 0), cfg.get('camera_shift_pixels', 0))


if __name__ == '__main__':
    main()
