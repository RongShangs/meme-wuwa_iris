"""Verify the exported montage and export opening/outro contact sheets."""
from pathlib import Path
import csv
import json
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from new_build import ROOT, WF, OUT, MANIFEST, CONFIG_NAME, TIMING, read_any, write_json
from new_timing import samples
from render import load_project


def main():
    path=OUT/(CONFIG_NAME+'.mp4')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)],encoding='utf-8'))
    video=next(s for s in probe['streams'] if s['codec_type']=='video')
    audio=next(s for s in probe['streams'] if s['codec_type']=='audio')
    total=TIMING['total_frames'];intro=TIMING['opening_frames'];outro=TIMING['ending_frames'];outro_start=TIMING['starts'][-1]
    assert (video['width'],video['height'],video['r_frame_rate'],int(video['nb_frames']))==(1080,1576,'30/1',total)
    assert abs(float(video['duration'])-total/30)<.001
    source_audio_path=ROOT/MANIFEST['music_timing']['prepared_audio'] if MANIFEST.get('music_timing') else ROOT/'素材/参考原始文件/原参考视频.mp4'
    source_probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',
        str(source_audio_path)],encoding='utf-8'))
    source_audio=next(s for s in source_probe['streams'] if s['codec_type']=='audio')
    expected_audio=min(total/30,float(source_audio['duration'])-(0 if MANIFEST.get('music_timing') else .1))
    assert abs(float(audio['duration'])-expected_audio)<.01
    subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True)
    original_checks={}
    generated_normal_checks={}
    character_lock_checks={}
    role_configs=[]
    for data in MANIFEST['characters']:
        who=data['name'];p=ROOT/'角色'/who/'new-项目.json'
        cfg=json.loads(p.read_text(encoding='utf-8'));load_project(p)
        role_configs.append(cfg)
        source=ROOT/(data.get('second',{}).get('generated_cel') or data.get('second',{}).get('source','new/1菲比/2.jpg'))
        a=read_any(source);b=read_any(p.parent/cfg['shots'][1]['face'])
        assert np.array_equal(a[:,:,:3],b[:,:,:3]),who+' did not adopt the selected Q cel'
        if data.get('second',{}).get('generated_cel'):
            assert np.array_equal(a,b),who+' generated Q alpha was changed'
        locked=data.get('second',{}).get('character_lock')
        if locked:
            import hashlib
            assert hashlib.sha256(source.read_bytes()).hexdigest()==locked['rgba_file_sha256']
            for key in ('scale','camera_dy','camera_dx','eye_center_source','iris_centers_source'):
                assert cfg['shots'][1][key]==locked[key],who+' character camera changed: '+key
            character_lock_checks[who]={'RGBA':'previous adopted character unchanged','camera':'previous adopted values unchanged','sha256':locked['rgba_file_sha256']}
        if data.get('first_face_cel'):
            generated=read_any(ROOT/data['first_face_cel'])
            adopted=read_any(p.parent/cfg['shots'][0]['face'])
            assert np.array_equal(generated,adopted),who+' generated normal-form alpha was changed'
            generated_normal_checks[who]='generated RGBA adopted unchanged'
        original_checks[who]='newly generated complete Q character cel adopted' if data.get('second',{}).get('generated_cel') else 'approved Phoebe cel retained'
    selected=list(range(intro))+list(range(outro_start,total))
    raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-vf',
                                 f"select='lt(n,{intro})+gte(n,{outro_start})',scale=270:394",'-fps_mode','passthrough',
                                 '-f','rawvideo','-pix_fmt','rgb24','-'])
    frames=np.frombuffer(raw,np.uint8).reshape(-1,394,270,3)
    assert len(frames)==intro+outro
    # Check the exported movie, not only current config/source paths: catches a
    # stale full montage after regenerating a role's cel or changing framing.
    indices=[TIMING['starts'][2*i]+j for i,cfg in enumerate(role_configs) for j in samples(cfg['shots'])]
    expression='+'.join(f'eq(n,{i})' for i in indices)
    raw_shots=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-vf',
        f"select='{expression}',scale=540:788",'-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','-'])
    actual=np.frombuffer(raw_shots,np.uint8).reshape(16,788,540,3)
    frame_checks={}
    for i,data in enumerate(MANIFEST['characters']):
        qa=ROOT/'角色'/data['name']/'成片'/MANIFEST['review_folder']
        errors=[]
        eye_errors=[]
        for j,index in enumerate(samples(role_configs[i]['shots'])):
            expected=np.asarray(Image.open(qa/f'frame-{index}.png').resize((540,788),Image.Resampling.BILINEAR))
            error=float(np.abs(actual[i*2+j].astype(float)-expected.astype(float)).mean())
            assert error<3,f'{data["name"]}: exported montage contains stale form {j+1}: {error:.3f}'
            if j==1 and data.get('second',{}).get('iris_generated'):
                eye_error=float(np.abs(actual[i*2+j,345:485,190:355].astype(float)-expected[345:485,190:355].astype(float)).mean())
                assert eye_error<4,f'{data["name"]}: exported iris is stale: {eye_error:.3f}'
                eye_errors.append(eye_error)
            errors.append(error)
        frame_checks[data['name']]={'forms':errors,'iris_roi':eye_errors} if eye_errors else errors
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',18)
    for label,start,count in [(f'开头{intro}帧检查',0,intro),(f'结尾{outro}帧检查',intro,outro)]:
        sheet=Image.new('RGB',(6*270,5*420),'#eeeeee');draw=ImageDraw.Draw(sheet)
        for j in range(count):
            i=start+j;x=j%6*270;y=j//6*420
            sheet.paste(Image.fromarray(frames[i]),(x,y+26))
            draw.text((x+4,y+2),f'{selected[i]:03d} {selected[i]/30:.3f}s',font=font,fill='#222222')
        sheet.save(OUT/(label+'.png'))
    seam_error=float(np.abs(frames[-1].astype(float)-frames[0].astype(float)).mean())
    assert seam_error<1.5,f'Loop endpoint mismatch: {seam_error}'
    assert frames[0].std()>15,'Missing waveform / iris gradient at loop endpoint'
    with (OUT/'镜头时间表.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['角色','形态','起始帧(从0)','结束帧(含)','起始秒','结束秒'])
        writer.writerow(['开头','声痕',0,intro-1,0,intro/30])
        for i,data in enumerate(MANIFEST['characters']):
            base,mid,end=TIMING['starts'][2*i:2*i+3]
            for form,a,b in [('第一形态',base,mid-1),('第二形态',mid,end-1)]:
                writer.writerow([data['name'],form,a,b,round(a/30,6),round((b+1)/30,6)])
        writer.writerow(['结尾','卡卡罗竖瞳放大变回声痕',outro_start,total-1,round(outro_start/30,6),round(total/30,6)])
    alignment=[]
    audio_loop={}
    if MANIFEST.get('music_timing'):
        from new_music import decode
        for t in (.5,3.5,6.5):
            reference=decode(source_audio_path,t,1,22050,1)[:,0]
            exported=decode(path,t,1,22050,1)[:,0]
            n=1<<(len(reference)+len(exported)-1).bit_length()
            c=np.fft.irfft(np.fft.rfft(exported,n)*np.fft.rfft(reference[::-1],n),n)
            zero=len(reference)-1;lag=int(np.argmax(c[zero-220:zero+221]))-220
            correlation=float(np.corrcoef(reference[:min(len(reference),len(exported))],exported[:min(len(reference),len(exported))])[0,1])
            assert abs(lag)<=10 and correlation>.98,(t,lag,correlation)
            alignment.append(dict(time=t,lag_samples=lag,correlation=correlation))
        # Audit the actual encoded track at the video loop boundary, including
        # trimming encoder padding to the container's exact playback duration.
        rate=44100;n=round(total/30*rate)
        decoded=decode(path,0,total/30,rate,2)[:n]
        assert len(decoded)==n,'Exported audio ends before the video'
        step=float(np.max(abs(decoded[-1]-decoded[0])))
        ordinary=np.max(abs(np.diff(decoded,axis=0)),axis=1)
        first_rms=float(np.sqrt(np.mean(decoded[:1544]**2)))
        last_rms=float(np.sqrt(np.mean(decoded[-1544:]**2)))
        assert first_rms>.01 and last_rms>.01,'Silent loop boundary'
        assert step<2*np.percentile(ordinary,99),'Large discontinuity at audio loop boundary'
        audio_loop=dict(samples=n,sample_rate=rate,video_audio_gap_seconds=abs(float(video['duration'])-float(audio['duration'])),
            boundary_sample_step=step,ordinary_step_95th=float(np.percentile(ordinary,95)),
            first_35ms_rms=first_rms,last_35ms_rms=last_rms,
            loop_beats=MANIFEST['music_timing']['loop_beats'],
            beat_phase_error_ms=(total/30-MANIFEST['music_timing']['loop_beats']*60/MANIFEST['music_timing']['bpm'])*1000,
            transition_ms=MANIFEST['music_timing']['loop_crossfade_ms'])
        import wave
        with wave.open(str(OUT/'首尾循环试听-两遍.wav'),'wb') as f:
            f.setnchannels(2);f.setsampwidth(2);f.setframerate(rate)
            pcm=np.rint(np.clip(decoded,-1,1)*32767).astype('<i2').tobytes()
            f.writeframes(pcm+pcm)
        write_json(OUT/'音乐循环核对.json',audio_loop)
    report={'size':[1080,1576],'fps':30,'frames':total,'duration_seconds':total/30,
            'audio_codec':audio['codec_name'],'full_decode':'passed','individual_clips':8,
            'audio_duration_seconds':float(audio['duration']),'audio_source_end_aligned':True,
            'individual_frames_reviewed':sum(TIMING['lengths']),'opening_frames':intro,'ending_frames':outro,
            'exported_audio_alignment':alignment,'beat_timing':TIMING,
            'exported_audio_loop':audio_loop,
            'loop_endpoint_mean_error':seam_error,
            'loop':'centered waveform -> Phoebe vertical pupil; Calcharo vertical pupil -> same waveform',
            'q_scale_factor':MANIFEST['rules']['q_scale_factor'],
            'q_individual_framing_factors':{d['name']:d.get('second',{}).get('framing_factor',1) for d in MANIFEST['characters']},
            'waveform_height':MANIFEST['rules']['mark_height'],
            'opening_iris':'three flat purple tooth-pattern layers transition into the actual Phoebe iris',
            'outro_pupil':'eye aperture and face alpha occlude the morph until the face has disappeared',
            'first_face_offset_x':MANIFEST['rules']['first_face_offset_x'],
            'second_form_source_checks':original_checks,
            'regenerated_normal_form_checks':generated_normal_checks,
            'character_lock_checks':character_lock_checks,
            'exported_shot_frame_mean_errors':frame_checks,
            'newly_generated_q_characters':sum(bool(d.get('second',{}).get('generated_cel')) for d in MANIFEST['characters']),
            'iris_type':'color plates; Chun uses a complete reference-matched iris fitted under the cleaned eye sockets',
            'iris_binding':'pair anchored to screen; independent of face camera_dx/camera_dy',
            'fringe':'short front fringe only; rear hair fixed',
            'exceptions':{'达尼亚':'reference closed-eye expression redrawn; no iris added','赞妮':'one visible eye; no hidden eye invented'}}
    write_json(OUT/'检查结果.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
