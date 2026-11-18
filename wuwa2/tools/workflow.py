"""Reference-frame animatic pipeline. Requires OpenCV, NumPy and FFmpeg."""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / 'project.json').read_text(encoding='utf-8'))
FPS = CONFIG['fps']


def shots():
    with (ROOT / 'shots.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    expected = 0
    for row in rows:
        start, end = int(row['start_frame']), int(row['end_frame'])
        if start != expected or end <= start:
            raise ValueError(f"Invalid or discontinuous shot: {row['id']}")
        expected = end
    if expected != CONFIG['total_frames']:
        raise ValueError('Timeline does not end at total_frames')
    return rows


def extract(source):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(source)
    for row in shots():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row['sample_frame']))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {row['sample_frame']}")
        path = ROOT / row['image']
        path.parent.mkdir(parents=True, exist_ok=True)
        # imencode handles Unicode paths on Windows.
        path.write_bytes(cv2.imencode('.png', frame)[1].tobytes())
    cap.release()


def render(source, output, width):
    height = round(width * CONFIG['height'] / CONFIG['width'] / 2) * 2
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
           '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(FPS), '-i', '-',
           '-i', str(source), '-map', '0:v:0', '-map', '1:a:0?', '-c:v', 'libx264',
           '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
           '-b:a', '192k', '-t', str(CONFIG['total_frames'] / FPS), '-movflags', '+faststart', str(output)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for row in shots():
            path = ROOT / row['image']
            if not path.is_file():
                raise FileNotFoundError(f'{path}; run extract or supply replacement artwork')
            image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f'Cannot decode {path}')
            # Centre crop to the target aspect ratio, preserving proportions.
            h, w = image.shape[:2]
            aspect = width / height
            if w / h > aspect:
                nw = round(h * aspect); x = (w - nw) // 2; image = image[:, x:x+nw]
            else:
                nh = round(w / aspect); y = (h - nh) // 2; image = image[y:y+nh, :]
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            for _ in range(int(row['end_frame']) - int(row['start_frame'])):
                proc.stdin.write(image.tobytes())
    finally:
        proc.stdin.close()
        code = proc.wait()
    if code:
        raise RuntimeError(f'ffmpeg exited {code}')
    print(f'Wrote {output}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['extract', 'render'])
    parser.add_argument('--source', default=str(ROOT / CONFIG['source']))
    parser.add_argument('--output', default=str(ROOT / 'deliverables' / 'animatic.mp4'))
    parser.add_argument('--width', type=int, default=540)
    args = parser.parse_args()
    if args.width < 2 or args.width % 2:
        parser.error('--width must be a positive even number')
    if args.action == 'extract':
        extract(Path(args.source))
    else:
        render(Path(args.source), Path(args.output), args.width)


if __name__ == '__main__':
    main()
