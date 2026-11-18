"""Build the montage from normal-form cels and newly redrawn complete Q cels."""
from pathlib import Path
import argparse
import copy
import json
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render import image, rgba_over, warp_rgba_premultiplied, load_project
from new_timing import plan, resample, samples

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / 'workflow'
MANIFEST = json.loads((WF / 'new-角色清单.json').read_text(encoding='utf-8'))
OUT = ROOT / 'deliverables' / MANIFEST.get('output_name','new-八角色')
QA_NAME = MANIFEST.get('review_folder','new-检查')
CONFIG_NAME = MANIFEST.get('config_name','new-八角色')
TIMING=plan(MANIFEST)
OLD = json.loads((WF / '1080p-心前发画风修正.json').read_text(encoding='utf-8'))
DX1 = [59.7,18.9,5.3,-1.5,-6.6,-8.3,-10,-10,-15.1,-21.9,-33.8,-49.1,-69.5]
DX2 = [-69.7,-23.8,-17,-8.5,-5.1,-1.7,0,3.4,6.8,10.2,15.3,25.5]
SWAY = [0,10.2,8.5,5.1,1.7,-1.7,-5.1,-6.8,-5.1,-1.7,3.4,8.5,13.6]
PATH_KEYS = ('face','hair','iris','accessories','face_cover','iris_layer','iris_clip')


def read_any(path):
    a = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise ValueError(path)
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGRA)
    if a.shape[2] == 3:
        a = np.dstack((a, np.full(a.shape[:2],255,np.uint8)))
    return a


def save(path, a):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode('.png', a)[1].tofile(str(path))


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def rel(path, folder):
    return Path(os.path.relpath(Path(path).resolve(),Path(folder).resolve())).as_posix()


def rebase(shot, source_folder, target_folder):
    result = copy.deepcopy(shot)
    for key in PATH_KEYS:
        if result.get(key):
            result[key] = rel(source_folder / result[key], target_folder)
    return result


def iris_asset(raw, out, side, plate=False):
    a = read_any(raw)
    if plate:
        a = cv2.resize(a,(side,side),interpolation=cv2.INTER_AREA)
        a[:,:,3] = 255
    else:
        _, _, stats, _ = cv2.connectedComponentsWithStats((a[:,:,3]>128).astype(np.uint8))
        x,y,w,h,_ = stats[1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))]
        extent = max(w,h)+4
        sc = side/extent
        m = np.float32([[sc,0,side/2-(x+w/2)*sc],[0,sc,side/2-(y+h/2)*sc]])
        a = warp_rgba_premultiplied(a,m,(side,side))
    save(out,a)
    return a


def find_eyes(face):
    h,w = face.shape[:2]
    _,labels,stats,centers = cv2.connectedComponentsWithStats((face[:,:,3]<16).astype(np.uint8))
    candidates = [(i,c) for i,c in enumerate(centers[1:],1)
                  if .3*w<c[0]<.7*w and .25*h<c[1]<.65*h
                  and 25<stats[i,2]<180 and 8<stats[i,3]<100 and stats[i,4]>350]
    pairs = [(a,b) for a in candidates for b in candidates
             if b[1][0]-a[1][0]>.1*w and abs(a[1][1]-b[1][1])<15]
    if not pairs:
        raise ValueError('Cannot identify two transparent eyes')
    a,b = max(pairs,key=lambda p:stats[p[0][0],4]+stats[p[1][0],4])
    return [np.rint(a[1]).astype(int).tolist(),np.rint(b[1]).astype(int).tolist()]


def clean_eye_whites(face, centers):
    """Matte cleanup only: preserve RGB and lids, clear white interiors adjoining holes."""
    result = face.copy()
    for cx,cy in centers:
        x0,x1=max(0,cx-95),min(face.shape[1],cx+95)
        y0,y1=max(0,cy-38),min(face.shape[0],cy+38)
        patch=face[y0:y1,x0:x1]
        rgb=patch[:,:,:3].astype(np.int16)
        white=(rgb.min(axis=2)>215)&(rgb.max(axis=2)-rgb.min(axis=2)<18)
        possible=((patch[:,:,3]<180)|white).astype(np.uint8)
        _,lab=cv2.connectedComponents(possible)
        component=lab[cy-y0,cx-x0]
        if component:
            result[y0:y1,x0:x1,3][lab==component]=0
    return result


def eye_mask(source, eye, who):
    total=np.zeros(source.shape[:2],np.uint8)
    if 'polygon' in eye:
        cv2.fillPoly(total,[np.asarray(eye['polygon'],np.int32)],255,cv2.LINE_AA)
        return cv2.GaussianBlur(total,(3,3),.45)
    if 'ellipse' in eye:
        cx,cy,rx,ry=eye['ellipse']
        cv2.ellipse(total,(cx,cy),(rx,ry),0,0,360,255,-1,cv2.LINE_AA)
        return total
    x0,y0,x1,y1=eye['roi'];roi=source[y0:y1,x0:x1,:3]
    if who=='爱弥斯':
        hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        binary=((hsv[:,:,0]>=8)&(hsv[:,:,0]<=40)&(hsv[:,:,1]>75)&(hsv[:,:,2]>60)).astype(np.uint8)*255
    elif who=='椿':
        colors=roi.astype(np.int16)
        binary=((colors[:,:,0]>colors[:,:,2]+8)&(colors[:,:,0]>colors[:,:,1]+3)).astype(np.uint8)*255
    else:
        binary=(cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)<eye['threshold']).astype(np.uint8)*255
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contour=max(contours,key=cv2.contourArea)
    mask=np.zeros(binary.shape,np.uint8)
    if who in ('爱弥斯','椿'):
        contour=cv2.convexHull(contour)
    cv2.drawContours(mask,[contour],-1,255,-1)
    erode=eye.get('erode',2)
    if who in ('爱弥斯','椿'):
        erode=0
    if erode:
        mask=cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*erode+1,2*erode+1)))
    mask=cv2.GaussianBlur(mask,(3,3),.55)
    total[y0:y1,x0:x1]=mask
    return total


def original_second(data, who, assets):
    spec=data['second'];raw=read_any(ROOT/spec['source'])
    if spec.get('mode')=='closed':
        save(assets/'第二形态-原图.png',raw)
        return {'name':who+'第二形态-原图闭眼','type':'still','face':str(assets/'第二形态-原图.png'),
                'anchor_source':spec['anchor'],'scale':spec['scale'],'camera_dx':DX2}
    openings=np.zeros(raw.shape[:2],np.uint8)
    for eye in spec['eyes']:
        openings=np.maximum(openings,eye_mask(raw,eye,who))
    face=raw.copy();face[:,:,3]=255-openings
    assert np.array_equal(face[:,:,:3],raw[:,:,:3])
    for x,y in [e['center'] for e in spec['eyes']]:
        if face[y,x,3]>16:
            raise ValueError(f'{who}: source iris center ({x},{y}) was not opened')
    save(assets/'第二形态-原图眼眶.png',face)
    save(assets/'第二形态-空前发.png',np.zeros_like(face))
    plate=assets/'第二形态-虹膜.png'
    size=spec.get('iris_size',544)
    iris_asset(assets/'第二形态-色片原图.png',plate,size,True)
    return {'name':who+'第二形态-原图','face':str(assets/'第二形态-原图眼眶.png'),
            'hair':str(assets/'第二形态-空前发.png'),'accessories':None,'iris':str(plate),
            'eye_center_source':spec['eyes'][0]['center'],'iris_centers_source':[e['center'] for e in spec['eyes']],
            'scale':spec['scale'],'iris_size':size,'camera_dx':DX2,'hair_sway':[0]*12,
            'hair_pivot':[0,0],'rotation_degrees_per_sway':0,'hair_registration':[1,0,0,0,1,0]}


def redrawn_eye_centers(face, count=2):
    h,w=face.shape[:2]
    _,labels,stats,centers=cv2.connectedComponentsWithStats((face[:,:,3]<16).astype(np.uint8))
    choices=[(i,c) for i,c in enumerate(centers[1:],1)
             if .12*w<c[0]<.88*w and .30*h<c[1]<.80*h and stats[i,4]>500
             and 15<stats[i,2]<.40*w and 15<stats[i,3]<.35*h]
    if count==1:
        if not choices:raise ValueError('Missing generated visible eye opening')
        i,c=max(choices,key=lambda p:stats[p[0],4])
        return [np.rint(c).astype(int).tolist()], [stats[i].tolist()]
    pairs=[(a,b) for a in choices for b in choices
           if b[1][0]-a[1][0]>.08*w and abs(a[1][1]-b[1][1])<.12*h]
    if not pairs:raise ValueError('Missing generated paired eye openings')
    a,b=max(pairs,key=lambda p:stats[p[0][0],4]+stats[p[1][0],4])
    ordered=[b,a]
    return [np.rint(p[1]).astype(int).tolist() for p in ordered], [stats[p[0]].tolist() for p in ordered]


def generated_second(data, who, assets):
    spec=data['second'];raw=read_any(ROOT/spec['generated_cel'])
    assets.mkdir(parents=True,exist_ok=True)
    face_path=assets/'第二形态-重绘整图.png';save(face_path,raw)
    if spec.get('mode')=='closed':
        eyes=spec['generated_closed_eyes']
        scale=spec.get('target_eye_distance',530)/np.linalg.norm(np.asarray(eyes[0])-eyes[1])
        return {'name':who+'第二形态-重绘闭眼','type':'still','face':str(face_path),
                'anchor_source':eyes[0],'scale':float(scale),'camera_dx':DX2}
    geometry=image(ROOT/spec['iris_geometry_source']) if spec.get('iris_geometry_source') else raw
    if geometry.shape!=raw.shape:raise ValueError(who+': iris geometry source must share the face canvas')
    centers,stats=redrawn_eye_centers(geometry,len(spec['eyes']))
    # The rectangular color plate may extend beyond a transparent character's
    # silhouette. Limit visibility to the eye windows, without moving its color.
    _,labels=cv2.connectedComponents((raw[:,:,3]<128).astype(np.uint8))
    ids=[int(labels[y,x]) for x,y in centers]
    if 0 in ids:raise ValueError(who+': eye window center is not transparent')
    window=(np.isin(labels,ids).astype(np.uint8)*255)
    window=cv2.dilate(window,np.ones((5,5),np.uint8))
    clip=np.full_like(raw,255);clip[:,:,3]=window
    clip_path=assets/'第二形态-眼眶可见区.png';save(clip_path,clip)
    if len(centers)==2:
        old=np.asarray([e['center'] for e in spec['eyes']],np.float64)
        target=spec['scale']*np.linalg.norm(old[0]-old[1])
        scale=target/np.linalg.norm(np.asarray(centers[0])-centers[1])
    else:
        target=spec.get('target_eye_height',178.2)
        scale=target/stats[0][3]
    blank=assets/'第二形态-空前发.png';save(blank,np.zeros_like(raw))
    plate=assets/'第二形态-虹膜.png';size=spec.get('iris_size',544)
    old_plate=ROOT/spec['iris_generated'] if spec.get('iris_generated') else ROOT/'角色'/who/'素材/new-v1/第二形态-色片原图.png'
    iris_asset(old_plate,plate,size,not bool(spec.get('iris_generated')))
    write_json(assets/'眼位配准.json',{'generated_source':spec['generated_cel'],'eye_centers':centers,
               'eye_regions':stats,'scale':float(scale),'reference':spec['source'],
               'head_hair_ornaments':'one fixed generated full-character cel'})
    shot={'name':who+'第二形态-Q版重绘','face':str(face_path),'hair':str(blank),
            'accessories':None,'iris':str(plate),'iris_clip':str(clip_path),'eye_center_source':centers[0],
            'iris_centers_source':centers,'scale':float(scale),'iris_size':size,
            'camera_dx':DX2,'hair_sway':[0]*12,'hair_pivot':[0,0],
            'rotation_degrees_per_sway':0,'hair_registration':[1,0,0,0,1,0]}
    if spec.get('iris_generated'):
        shot['iris_source_eye_height']=int(stats[0][3])
        shot['iris_source_eye_region']=stats[0][:4]
    if spec.get('eye_background_bgr') is not None:
        shot['eye_background_bgr']=spec['eye_background_bgr']
    return shot


def make_group(shot, folder, assets, prefix):
    if shot.get('type')=='still':return
    face=image(folder/shot['face']);iris=image(folder/shot['iris'])
    scale=float(shot['scale']);diam=float(shot['iris_size'])
    centers=np.asarray(shot['iris_centers_source'],np.float64)*scale
    origin=np.floor(centers.min(axis=0)-diam/2-3)
    end=np.ceil(centers.max(axis=0)+diam/2+3)
    gw,gh=(end-origin).astype(int);group=np.zeros((gh,gw,4),np.uint8)
    factor=diam/iris.shape[1]
    for cx,cy in centers-origin:
        m=np.float32([[factor,0,cx-diam/2],[0,factor,cy-diam/2]])
        group=rgba_over(group,warp_rgba_premultiplied(iris,m,(gw,gh)))
    path=assets/(prefix+'-虹膜组.png');save(path,group)
    shot['iris_layer']=rel(path,folder)
    shot['iris_layer_registration']=[1/scale,0,origin[0]/scale,0,1/scale,origin[1]/scale]


def build_character(data):
    who=data['name'];folder=ROOT/'角色'/who;assets=folder/'素材/new-v1'
    assets.mkdir(parents=True,exist_ok=True)
    if data.get('approved_config'):
        old_path=ROOT/data['approved_config'];cfg=json.loads(old_path.read_text(encoding='utf-8'))
        cfg['shots']=[rebase(s,old_path.parent,folder) for s in cfg['shots']]
        q=cfg['shots'][1]
        iris_asset(assets/'第二形态-色片原图.png',assets/'第二形态-虹膜.png',q['iris_size'],True)
        q['iris']=rel(assets/'第二形态-虹膜.png',folder)
        if data.get('second',{}).get('generated_cel'):
            second=generated_second(data,who,folder/'素材/q2-v2')
            for key in PATH_KEYS:
                if second.get(key):second[key]=rel(second[key],folder)
            cfg['shots'][1]=second
    else:
        if data.get('reuse'):
            first=next(s for s in OLD['shots'] if s['name']==who+'第一形态')
            first=rebase(first,WF,folder)
            first.pop('face_cover',None)
            first['scale']*=.8
            first['iris_size']=200
            first['iris_centers_source']=data['first_eyes']
            first.setdefault('hair_registration',[1,0,0,0,1,0])
        else:
            raw=image(ROOT/data['first_face_cel']) if data.get('first_face_cel') else image(assets/'第一形态-面部.png')
            eyes=find_eyes(raw)
            face=raw.copy() if data.get('first_face_cel') else clean_eye_whites(raw,eyes)
            face_path=ROOT/data['first_face_cel'] if data.get('first_face_cel') else assets/'第一形态-面部-眼眶清理.png'
            if not data.get('first_face_cel'):
                save(face_path,face)
            hair=image(assets/'第一形态-前发-原图.png')
            if hair.shape!=face.shape:
                raise ValueError(f'{who}: fringe must share exact original canvas')
            save(assets/'第一形态-前发.png',hair)
            iris_asset(assets/'第一形态-虹膜原图.png',assets/'第一形态-虹膜.png',200)
            first={'name':who+'第一形态','face':rel(face_path,folder),
                   'hair':rel(assets/'第一形态-前发.png',folder),'accessories':None,
                   'iris':rel(assets/'第一形态-虹膜.png',folder),'iris_size':200,
                   'eye_center_source':eyes[0],'iris_centers_source':eyes,
                   'scale':data['first_scale'],'camera_dx':DX1,'hair_sway':SWAY,
                   'hair_pivot':[int((eyes[0][0]+eyes[1][0])/2),int(face.shape[0]*.12)],
                   'rotation_degrees_per_sway':.05,'hair_registration':[1,0,0,0,1,0]}
        if data['second'].get('generated_cel'):
            second=generated_second(data,who,folder/'素材/q2-v2')
        else:
            second=original_second(data,who,assets)
        for key in PATH_KEYS:
            if second.get(key):second[key]=rel(second[key],folder)
        cfg={'canvas':{'width':1080,'height':1576,'fps':30,'background_bgr':[250,250,250]},
             'camera_shift_pixels':10,'shots':[first,second]}
    # Apply once to freshly loaded sources: successive rebuilds cannot accumulate.
    first,q=cfg['shots']
    first['iris_highlight']=False if who=='卡卡罗' else {'vertical':-.13 if who in ('爱弥斯','今汐') else -.15}
    if who=='爱弥斯':
        q['iris_highlight']={'direction':-1,'vertical':-.13,'radius_x':.05,'radius_y':.033}
    first['camera_dx']=[float(dx)+MANIFEST['rules'].get('first_face_offset_x',0) for dx in first['camera_dx']]
    q_factor=MANIFEST['rules'].get('q_scale_factor',1)*data.get('second',{}).get('framing_factor',1)
    q['scale']*=q_factor
    if q.get('iris_size'):
        q['iris_size']=round(q['iris_size']*q_factor)
    if data.get('second',{}).get('iris_generated'):
        spec=data['second']
        aperture_height=q['iris_source_eye_height']*q['scale']
        q['iris_size']=round(aperture_height*data['second'].get('iris_height_ratio',1.28))
        q['camera_dy']=(q['iris_size']-aperture_height)/2+8
        if spec.get('character_lock'):
            # Freeze the adopted character and its framing; fit ONLY the new iris.
            import hashlib
            locked=spec['character_lock']
            assert hashlib.sha256((ROOT/spec['generated_cel']).read_bytes()).hexdigest()==locked['rgba_file_sha256']
            for key in ('scale','eye_center_source','iris_centers_source','camera_dx','camera_dy'):
                q[key]=copy.deepcopy(locked[key])
        if spec.get('iris_fit')=='eye_width':
            x,y,rw,rh=q['iris_source_eye_region']
            q['iris_size']=round(rw*q['scale']*spec.get('iris_width_ratio',1))
            eye_bottom=(y+rh-.5-q['eye_center_source'][1])*q['scale']+q.get('camera_dy',0)
            q['iris_screen_offset']=[0,round(eye_bottom-q['iris_size']/2-2,3)]
    q['hair_sway']=[0]*len(q['camera_dx'])
    index=2*(data['order']-1)
    for shot,length in zip(cfg['shots'],TIMING['lengths'][index:index+2]):
        for key in ('camera_dx','hair_sway'):
            if key in shot:shot[key]=resample(shot[key],length)
    if MANIFEST.get('music_timing'):
        cfg['audio']={'path':rel(ROOT/MANIFEST['music_timing']['prepared_audio'],folder),
                      'start_seconds':TIMING['starts'][index]/30}
    else:
        cfg['audio']={'path':rel(ROOT/'素材/参考原始文件/原参考视频.mp4',folder),
                      'start_seconds':1.1+(data['order']-1)*25/30}
    cfg['timing']={'global_start_frame':TIMING['starts'][index],
                   'sample_frames':samples(cfg['shots'])}
    cfg['output']='成片/'+MANIFEST.get('clip_name','new-双形态.mp4')
    cfg['provenance']={'order':data['order'],'first_form':'reused existing adopted cels' if data.get('reuse') or data.get('approved_config') else 'new normal-form cels',
                       'second_form':'new generated full Q character cel plus separate eye color plate' if data.get('second',{}).get('generated_cel') else 'approved Phoebe Q image plus eye color plate',
                       'iris_motion':'screen-anchored group, independent from face camera movement'}
    for index,shot in enumerate(cfg['shots'],1):
        group_assets=folder/'素材/q2-v2' if index==2 and data.get('second',{}).get('generated_cel') else assets
        make_group(shot,folder,group_assets,'第一形态' if index==1 else '第二形态')
    path=folder/'new-项目.json';write_json(path,cfg)
    load_project(path)
    (folder/'new-渲染.py').write_text(
        "from pathlib import Path\nimport subprocess,sys\np=Path(__file__).resolve().parent\n"
        "subprocess.run([sys.executable,str(p.parents[1]/'workflow'/'new_build.py'),'--characters',p.name,'--render'],check=True)\n",encoding='utf-8')
    print(f'Prepared {who}',flush=True)
    return path


def check_clip(config_path):
    cfg=json.loads(config_path.read_text(encoding='utf-8'));folder=config_path.parent
    video=(folder/cfg['output']).resolve()
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(video)],encoding='utf-8'))
    stream=next(s for s in probe['streams'] if s['codec_type']=='video')
    count=sum(len(s['camera_dx']) for s in cfg['shots']);sample_frames=samples(cfg['shots'])
    assert (stream['width'],stream['height'],stream['r_frame_rate'],int(stream['nb_frames']))==(1080,1576,'30/1',count)
    assert any(s['codec_type']=='audio' for s in probe['streams'])
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
    raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(video),'-f','rawvideo','-pix_fmt','rgb24','-'])
    frames=np.frombuffer(raw,np.uint8).reshape(count,1576,1080,3)
    qa=folder/'成片'/QA_NAME;qa.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    small=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',18)
    pair=Image.new('RGB',(1080,832),'#eeeeee');draw=ImageDraw.Draw(pair)
    q_label='第二形态 · 重绘' if cfg['provenance']['second_form'].startswith('new generated') else '第二形态'
    for x,index,label in [(0,sample_frames[0],'第一形态'),(540,sample_frames[1],q_label)]:
        pair.paste(Image.fromarray(frames[index]).resize((540,788),Image.Resampling.LANCZOS),(x,44))
        draw.text((x+15,7),folder.name+' · '+label,font=font,fill='#222222')
    pair.save(qa/'双形态对照.png')
    sheet=Image.new('RGB',(1080,((count+4)//5)*339),'#eeeeee');draw=ImageDraw.Draw(sheet)
    for i,frame in enumerate(frames):
        x,y=i%5*216,i//5*339
        sheet.paste(Image.fromarray(frame).resize((216,315),Image.Resampling.LANCZOS),(x,y+24))
        draw.text((x+4,y+1),f'{i:02d} {i/30:.3f}s',font=small,fill='#222222')
    sheet.save(qa/f'{count}帧检查.png')
    # Verify all original Q image RGB data are retained, not just a center sample.
    data=next(d for d in MANIFEST['characters'] if d['name']==folder.name)
    if data.get('second'):
        original=read_any(ROOT/data['second'].get('generated_cel',data['second']['source']))
        q=read_any(folder/cfg['shots'][1]['face'])
        assert np.array_equal(original[:,:,:3],q[:,:,:3]),folder.name+' selected generated Q cel changed'
        if data['second'].get('generated_cel'):
            assert np.array_equal(original,q),folder.name+' generated transparency was changed'
        if cfg['shots'][1].get('type')!='still':
            assert not np.any(read_any(folder/cfg['shots'][1]['hair']))
    for index in sample_frames:
        Image.fromarray(frames[index]).save(qa/f'frame-{index}.png')
    report={'frames':count,'sample_frames':sample_frames,'fps':30,'size':[1080,1576],'full_decode':'passed',
            'second_form_rgb':'generated cel adopted' if data.get('second',{}).get('generated_cel') else 'approved Phoebe cel',
            'second_form_sway':'none',
            'iris_binding':'screen-anchored group','first_form_reused':bool(data.get('reuse') or data.get('approved_config'))}
    write_json(qa/'检查结果.json',report)
    import shutil
    (OUT/'角色分段').mkdir(parents=True,exist_ok=True)
    shutil.copy2(video,OUT/'角色分段'/(folder.name+'.mp4'))
    return frames


def assemble():
    shots=[]
    for data in MANIFEST['characters']:
        p=ROOT/'角色'/data['name']/'new-项目.json'
        c=json.loads(p.read_text(encoding='utf-8'))
        shots.extend(rebase(s,p.parent,WF) for s in c['shots'])
    opening=copy.deepcopy(OLD['opening'])
    opening['pupil_reference']=shots[0]['iris']
    opening['pupil_max_rgb']=100
    opening['final_iris_size']=200
    opening['mode']='resonance_loop'
    opening['last_iris_reference']=shots[-1]['iris']
    opening['mark_height']=MANIFEST['rules'].get('mark_height',1050)
    if MANIFEST['rules'].get('resonance_background_bgr') is not None:
        opening['background_bgr']=MANIFEST['rules']['resonance_background_bgr']
    opening['layered_iris_background']=MANIFEST['rules'].get('layered_iris_background',False)
    if opening['layered_iris_background']:
        opening['pattern_reference']=rel(ROOT/'角色/菲比/素材/分层-v1/第一形态-虹膜原图.png',WF)
    opening['iris_contracts_from_outside']=True
    opening['frames']=TIMING['opening_frames']
    audio={'path':rel(ROOT/MANIFEST['music_timing']['prepared_audio'],WF),'start_seconds':0} if MANIFEST.get('music_timing') else {'path':'../素材/参考原始文件/原参考视频.mp4','start_seconds':.1}
    c={'canvas':copy.deepcopy(OLD['canvas']),'audio':audio,
       'output':rel(OUT/(CONFIG_NAME+'.mp4'),WF),'shots':shots,
       'opening':opening,'ending':{'frames':TIMING['ending_frames'],'mode':'resonance_loop'},'camera_shift_pixels':10,
       'timing':TIMING}
    path=WF/(CONFIG_NAME+'.json');write_json(path,c)
    load_project(path)
    subprocess.run([sys.executable,str(WF/'render.py'),str(path)],check=True)
    return path


def overview():
    canvas=Image.new('RGB',(1440,4*560),'#eeeeee');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    for i,data in enumerate(MANIFEST['characters']):
        qa=ROOT/'角色'/data['name']/'成片'/QA_NAME
        cfg=json.loads((ROOT/'角色'/data['name']/'new-项目.json').read_text(encoding='utf-8'))
        for j,index in enumerate(samples(cfg['shots'])):
            n=i*2+j;x=n%4*360;y=n//4*560
            im=Image.open(qa/f'frame-{index}.png').resize((360,525),Image.Resampling.LANCZOS)
            canvas.paste(im,(x,y+35));draw.text((x+10,y+3),data['name']+(' · 正常' if j==0 else ' · Q版'),font=font,fill='#222222')
    OUT.mkdir(parents=True,exist_ok=True);canvas.save(OUT/'八角色双形态总览.png')
    q_grid=Image.new('RGB',(1350,1500),'#eeeeee');q_draw=ImageDraw.Draw(q_grid)
    q_index=0
    for data in MANIFEST['characters']:
        if not data.get('second',{}).get('generated_cel'):continue
        folder=ROOT/'角色'/data['name'];cfg=json.loads((folder/'new-项目.json').read_text(encoding='utf-8'))
        shot=cfg['shots'][1];face=image(folder/shot['face'])
        preview=face
        if shot.get('type')!='still':
            group=image(folder/shot['iris_layer'])
            matrix=np.asarray(shot['iris_layer_registration'],np.float64).reshape(2,3)
            offset=np.asarray(shot.get('iris_screen_offset',[0,0]),float)
            offset[1]-=shot.get('camera_dy',0)
            matrix[:,2]+=offset/shot['scale']
            under=warp_rgba_premultiplied(group,matrix,(face.shape[1],face.shape[0]))
            window=image(folder/shot['iris_clip'])
            under[:,:,3]=np.minimum(under[:,:,3],window[:,:,3])
            if shot.get('eye_background_bgr') is not None:
                backing=np.empty_like(face);backing[:,:,:3]=shot['eye_background_bgr'];backing[:,:,3]=window[:,:,3]
                under=rgba_over(backing,under)
            preview=rgba_over(under,face)
        save(folder/'素材/q2-v2/角色完整预览.png',preview)
        rgba=Image.fromarray(cv2.cvtColor(preview,cv2.COLOR_BGRA2RGBA))
        rgba.thumbnail((430,450),Image.Resampling.LANCZOS)
        x=q_index%3*450;y=q_index//3*500
        q_grid.paste(rgba,(x+(450-rgba.width)//2,y+42+(450-rgba.height)//2),rgba)
        q_draw.text((x+15,y+7),data['name']+' · Q版重绘',font=font,fill='#222222')
        q_index+=1
    if q_index:q_grid.save(OUT/'Q版整图总览.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--characters',nargs='+')
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--assemble',action='store_true')
    parser.add_argument('--overview',action='store_true')
    args=parser.parse_args()
    data=[d for d in MANIFEST['characters'] if not args.characters or d['name'] in args.characters]
    if not args.overview:
        if MANIFEST.get('music_timing'):
            from new_music import prepare
            prepare(MANIFEST,OUT)
        for d in data:
            p=build_character(d)
            if args.render:
                subprocess.run([sys.executable,str(WF/'render.py'),str(p)],check=True)
                check_clip(p)
                print('Checked '+d['name'],flush=True)
    if args.assemble:assemble()
    if args.overview:overview()


if __name__=='__main__':main()
