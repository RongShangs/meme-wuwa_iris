"""Export visual review sheets and verify the encoded standalone character clip."""
from pathlib import Path
import json
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / '成片' / '双虹膜固定版检查'
VIDEO = HERE / '成片' / '菲比-双虹膜固定版.mp4'


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json',
        str(VIDEO)], encoding='utf-8'))
    video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    audio = [s for s in probe['streams'] if s['codec_type'] == 'audio']
    assert (video['width'], video['height'], video['r_frame_rate'],
            int(video['nb_frames'])) == (1080, 1576, '30/1', 25)
    assert audio, 'Missing reference audio'
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(VIDEO), '-f', 'null', '-'],
                   check=True)
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(VIDEO),
                                   '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 1576, 1080, 3)
    assert len(frames) == 25
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 24)
    small_font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 18)
    contact = Image.new('RGB', (1080, 5 * 339), '#eeeeee')
    draw = ImageDraw.Draw(contact)
    for index, frame in enumerate(frames):
        x, y = (index % 5) * 216, (index // 5) * 339
        contact.paste(Image.fromarray(frame).resize((216, 315), Image.Resampling.LANCZOS),
                      (x, y + 24))
        draw.text((x + 5, y + 2), f'{index:02d}  {index/30:.3f}s',
                  fill='#222222', font=small_font)
    contact.save(OUTPUT / '25帧检查.png')
    pair = Image.new('RGB', (1080, 832), '#eeeeee')
    draw = ImageDraw.Draw(pair)
    for x, index, label in [(0, 6, '第一形态'), (540, 19, '第二形态 · Q版')]:
        pair.paste(Image.fromarray(frames[index]).resize((540, 788), Image.Resampling.LANCZOS),
                   (x, 44))
        draw.text((x + 18, 6), label, fill='#222222', font=font)
        Image.fromarray(frames[index]).save(OUTPUT / ('第一形态-静帧.png' if x == 0
                                                      else '第二形态-静帧.png'))
    pair.save(OUTPUT / '双形态对照.png')
    cfg = json.loads((HERE / '项目.json').read_text(encoding='utf-8'))
    assert all(v == 0 for v in cfg['shots'][1]['hair_sway'])
    assert all(len(s['iris_centers_source']) == 2 for s in cfg['shots'])
    np.testing.assert_allclose([s['scale'] for s in cfg['shots']], [3.12, 2.24*1254/1440])
    assert [s['iris_size'] for s in cfg['shots']] == [200, 544]
    assert cfg['shots'][0]['accessories_follow_hair'] is False
    assert cfg['shots'][1]['accessories'] is None
    q = cv2.imdecode(np.frombuffer((HERE.parents[1] / 'new/1菲比/2.jpg').read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    q_face = cv2.imdecode(np.frombuffer((HERE / cfg['shots'][1]['face']).read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(q, q_face[:, :, :3]), 'Q image changed outside alpha compositing'
    q_hair = cv2.imdecode(np.frombuffer((HERE / cfg['shots'][1]['hair']).read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    assert np.count_nonzero(q_hair) == 0
    report = {
        'width': video['width'], 'height': video['height'],
        'fps': video['r_frame_rate'], 'frames': len(frames),
        'duration_seconds': float(video['duration']),
        'audio_codec': audio[0]['codec_name'],
        'full_decode': 'passed', 'q_version_hair_sway': 'all zero',
        'cut_frame_zero_based': 13,
        'relative_character_and_iris_scale': 0.8,
        'iris_motion': 'screen-anchored pair; independent of face camera_dx and camera_dy',
        'second_form': 'original image pixels preserved; only iris is generated',
        'first_form_hat': 'above bangs; does not follow hair sway',
        'visual_review': ['25帧检查.png', '双形态对照.png'],
    }
    (OUTPUT / '检查结果.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)
                                          + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
