"""Prepare continuous original-song audio and document the measured beat grid."""
from pathlib import Path
import json
import subprocess
import wave
import numpy as np
from new_timing import plan

ROOT=Path(__file__).resolve().parents[1]


def decode(path,start,duration,rate=44100,channels=2):
    raw=subprocess.check_output(['ffmpeg','-v','error','-ss',str(start),'-i',str(path),
        '-t',str(duration),'-vn','-ac',str(channels),'-ar',str(rate),'-f','f32le','-'])
    return np.frombuffer(raw,np.float32).reshape(-1,channels)


def prepare(manifest,out):
    spec=manifest['music_timing'];timing=plan(manifest);rate=44100
    start=float(spec['source_first_face_beat_seconds'])-timing['opening_frames']/30
    duration=timing['total_frames']/30;n=round(duration*rate)
    blend_n=round(spec.get('loop_crossfade_ms',35)*rate/1000)
    source=ROOT/spec['source']
    raw=decode(source,start-blend_n/rate,duration+blend_n/rate+.03)
    audio=raw[blend_n:blend_n+n].copy()
    assert len(audio)==n
    # Crossfade into the pre-roll BEFORE the first sample, preserving timeline
    # length and avoiding a silent gap or a repeated first 35 ms at the seam.
    weight=(.5-.5*np.cos(np.linspace(0,np.pi,blend_n)))[:,None]
    audio[-blend_n:]=audio[-blend_n:]*(1-weight)+raw[:blend_n]*weight
    audio*=float(spec.get('output_gain',1.0))
    target=ROOT/spec['prepared_audio'];target.parent.mkdir(parents=True,exist_ok=True)
    with wave.open(str(target),'wb') as f:
        f.setnchannels(2);f.setsampwidth(2);f.setframerate(rate)
        f.writeframes(np.rint(np.clip(audio,-1,1)*32767).astype('<i2').tobytes())
    # Measure the onset envelope from the actual original-song passage.
    sr=22050;hop=64;window=1024
    mono=decode(source,start,duration,sr,1)[:,0]
    padded=np.pad(mono,(window//2,window//2))
    frames=np.lib.stride_tricks.sliding_window_view(padded,window)[::hop]
    spectrum=np.abs(np.fft.rfft(frames*np.hanning(window),axis=1))
    freq=np.fft.rfftfreq(window,1/sr)
    log=np.log1p(spectrum*10)
    diff=np.maximum(np.diff(log,axis=0,prepend=log[:1]),0)
    bands=[np.mean(diff[:,(freq>lo)&(freq<hi)],axis=1) for lo,hi in [(50,250),(250,2000),(2000,9000)]]
    envelope=sum(b/(np.percentile(b,95)+1e-9) for b in bands)
    times=np.arange(len(envelope))*hop/sr
    exact=timing['opening_frames']/30+np.arange(17)*timing['beat_seconds']
    errors=np.asarray(timing['starts'])/30-exact
    # Fit the pulse independently of the chosen 30 fps quantization.
    best=(-1,None,None)
    for bpm in np.linspace(128,132,401):
        for phase in np.linspace(.98,1.02,81):
            grid=phase+np.arange(17)*60/bpm
            score=float(np.mean(np.interp(grid,times,envelope)))
            if score>best[0]:best=(score,float(bpm),float(phase))
    out.mkdir(parents=True,exist_ok=True)
    report=dict(source=spec['source'],source_start_seconds=start,prepared_audio=spec['prepared_audio'],
        sample_rate=rate,samples=n,duration_seconds=duration,selected_bpm=spec['bpm'],
        fitted_bpm=best[1],fitted_first_face_beat_seconds=best[2],
        cut_frames=timing['starts'],cut_error_ms=(errors*1000).round(3).tolist(),
        maximum_grid_error_ms=float(np.max(abs(errors))*1000),
        loop_beats=spec['loop_beats'],loop_crossfade_ms=spec['loop_crossfade_ms'],
        audio_seam_sample_step=float(np.max(abs(audio[-1]-audio[0]))),
        previous_problem='fixed 13/12 frame shots progressively preceded the actual song pulse')
    (out/'音乐卡点核对.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei']
    fig,axes=plt.subplots(2,1,figsize=(16,6),sharex=True,layout='constrained')
    for ax,title,cuts,color in [(axes[0],'上一版：固定 13/12 帧，后半段逐渐抢拍',np.array([30+i*25+j for i in range(8) for j in (0,13)])/30,'#c94b51'),
            (axes[1],'修正版：按约 130 BPM 分配切点，取最近的视频帧',np.array(timing['starts'])/30,'#168273')]:
        ax.plot(times,envelope,color='#65748a',lw=.7)
        for x in exact:ax.axvline(x,color='#222222',lw=.6,alpha=.3)
        for x in cuts:ax.axvline(x,color=color,lw=1.2)
        ax.set_title(title,loc='left');ax.set_ylabel('声音起音强度');ax.grid(alpha=.15)
    axes[-1].set_xlabel('成片时间 / 秒');axes[-1].set_xlim(0,duration)
    fig.savefig(out/'音乐卡点对照.png',dpi=150);plt.close(fig)
    print('Prepared beat audio:',timing['total_frames'],'frames; max grid error',round(report['maximum_grid_error_ms'],2),'ms',flush=True)
    return target
