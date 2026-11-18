"""Frame timing shared by role builds, the montage, review sheets and checks."""
import numpy as np


def plan(manifest):
    spec=manifest.get('music_timing')
    if not spec:
        starts=np.array([30+i*25+j for i in range(8) for j in (0,13)]+[230])
        return dict(opening_frames=30,ending_frames=29,total_frames=259,
                    starts=starts.tolist(),lengths=np.diff(starts).tolist())
    fps=30;period=60/float(spec['bpm']);intro=int(spec.get('opening_frames',30))
    starts=np.rint((intro/fps+np.arange(17)*period)*fps).astype(int)
    total=round(int(spec.get('loop_beats',20))*period*fps)
    if total-starts[-1]<15:raise ValueError('Not enough frames for the loop outro')
    return dict(opening_frames=int(starts[0]),ending_frames=int(total-starts[-1]),
                total_frames=total,starts=starts.tolist(),lengths=np.diff(starts).tolist(),
                beat_seconds=period,bpm=float(spec['bpm']))


def resample(values,count):
    return np.interp(np.linspace(0,1,count),np.linspace(0,1,len(values)),values).round(5).tolist()


def samples(shots):
    first,second=(len(s['camera_dx']) for s in shots)
    return [first//2,first+second//2]
