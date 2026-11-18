"""Normalize generated cels and arrange both irises in one registered group.

Original imagegen assets are retained. This only frames/resamples supplied cels
and composites layers, using the project's existing premultiplied-alpha tools.
"""
from pathlib import Path
import json
import os
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'workflow'))
from render import image, rgba_over, warp_rgba_premultiplied, over_rgb

ASSETS = HERE / '素材' / '分层-v1'


def save(name, pixels):
    cv2.imencode('.png', pixels)[1].tofile(str(ASSETS / name))


def normalize_iris(form, size):
    source = image(ASSETS / f'{form}-虹膜原图.png')
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (source[:, :, 3] > 128).astype(np.uint8))
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = stats[index]
    # Frame the main iris disc instead of sizing the tool's outer transparent margin.
    side = max(w, h) + 4
    cx, cy = x + w / 2, y + h / 2
    factor = size / side
    transform = np.float32([[factor, 0, size/2-cx*factor],
                            [0, factor, size/2-cy*factor]])
    iris = warp_rgba_premultiplied(source, transform, (size, size))
    save(f'{form}-虹膜.png', iris)
    return iris


def prepare_second_original():
    """Use the supplied Q image verbatim; only open its original eye interiors."""
    source = cv2.imdecode(np.frombuffer((ROOT / 'new/1菲比/2.jpg').read_bytes(), np.uint8),
                          cv2.IMREAD_COLOR)
    face = np.dstack((source, np.full(source.shape[:2], 255, dtype=np.uint8)))
    openings = np.zeros(source.shape[:2], dtype=np.uint8)
    for x0, y0, x1, y1 in [(440, 620, 655, 825), (770, 625, 985, 825)]:
        roi = source[y0:y1, x0:x1]
        dark = ((roi[:, :, 0] < 100) & (roi[:, :, 1] < 85)
                & (roi[:, :, 2] < 100)).astype(np.uint8) * 255
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 10000:
            raise ValueError('Could not identify the original Q eye boundary')
        mask = np.zeros(dark.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        inside = mask.copy()
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)))
        colors = roi.astype(np.int16)
        purple = ((colors[:, :, 0] > colors[:, :, 2] + 8)
                  & (colors[:, :, 0] > colors[:, :, 1] + 15))
        mask[(inside > 0) & purple] = 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        mask = cv2.GaussianBlur(mask, (5, 5), 0.75)
        mask[inside == 0] = 0
        openings[y0:y1, x0:x1] = mask
    face[:, :, 3] = 255 - openings
    save('第二形态-原图眼眶.png', face)
    save('第二形态-空前发.png', np.zeros_like(face))
    assert np.array_equal(face[:, :, :3], source), 'Original Q pixels changed'
    assert np.all(face[:, :, 3][openings == 0] == 255)


def prepare_first_hat():
    hat = image(ASSETS / '第一形态-帽子-生成原图.png')
    face = image(ASSETS / '第一形态-面部-v2.png')
    record = json.loads((ASSETS / '第一形态-帽子配准.json').read_text(encoding='utf-8'))
    matrix = np.asarray(record['matrix'], np.float64)
    save('第一形态-帽子-v2.png', warp_rgba_premultiplied(
        hat, matrix, (face.shape[1], face.shape[0])))


def make_iris_group(form, shot, iris):
    face = image(HERE / shot['face'])
    h, w = face.shape[:2]
    scale = float(shot['scale'])
    diameter = float(shot['iris_size'])
    centers = np.asarray(shot['iris_centers_source'], dtype=np.float64) * scale
    origin = np.floor(centers.min(axis=0) - diameter/2 - 3)
    end = np.ceil(centers.max(axis=0) + diameter/2 + 3)
    gw, gh = (end - origin).astype(int)
    group = np.zeros((gh, gw, 4), dtype=np.uint8)
    factor = diameter / iris.shape[1]
    for (sx, sy), (cx, cy) in zip(shot['iris_centers_source'], centers - origin):
        if face[int(sy), int(sx), 3] > 16:
            raise ValueError(f'{form}: iris center must remain transparent in the face')
        placement = np.float32([[factor, 0, cx-diameter/2],
                                [0, factor, cy-diameter/2]])
        group = rgba_over(group, warp_rgba_premultiplied(iris, placement, (gw, gh)))
    save(f'{form}-双虹膜.png', group)
    # Preserve output-resolution iris detail. One parent registration adapts eye
    # spacing to the face, with no per-eye motion or repeated down/up-sampling.
    registration = [1/scale, 0, origin[0]/scale, 0, 1/scale, origin[1]/scale]
    shot['iris_layer_registration'] = registration
    source_group = warp_rgba_premultiplied(group, np.float32(registration).reshape(2, 3), (w, h))
    preview = rgba_over(source_group, face)
    preview = rgba_over(preview, image(HERE / shot['hair']))
    if shot.get('accessories'):
        preview = rgba_over(preview, image(HERE / shot['accessories']))
    save(f'{form}-双虹膜合成检查.png', preview)
    green = np.zeros((h, w, 3), np.uint8)
    green[:] = (85, 230, 95)
    over_rgb(green, face, 0, 0)
    save(f'{form}-双眼透明检查.png', green)


def main():
    config = json.loads((HERE / '项目.json').read_text(encoding='utf-8'))
    prepare_second_original()
    prepare_first_hat()
    normal = normalize_iris('第一形态', 250)
    chibi = normalize_iris('第二形态', 680)
    make_iris_group('第一形态', config['shots'][0], normal)
    make_iris_group('第二形态', config['shots'][1], chibi)
    (HERE / '项目.json').write_text(json.dumps(config, ensure_ascii=False, indent=2)
                                    + '\n', encoding='utf-8')
    # Keep a workflow-folder entry alongside the standalone character recipe.
    workflow = ROOT / 'workflow'
    for shot in config['shots']:
        for key in ('face', 'hair', 'iris', 'iris_layer', 'accessories'):
            if shot.get(key):
                shot[key] = Path(os.path.relpath(HERE / shot[key], workflow)).as_posix()
    config['audio']['path'] = Path(os.path.relpath(
        HERE / config['audio']['path'], workflow)).as_posix()
    config['output'] = '../deliverables/菲比-双虹膜固定版/菲比-双虹膜固定版.mp4'
    (workflow / '菲比-双虹膜固定版.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Prepared registered eye cels; original generated PNGs preserved.')


if __name__ == '__main__':
    main()
