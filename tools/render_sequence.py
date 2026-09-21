"""Render a completed frame sequence with shared cameras and a one-second hold per frame."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orhsurf import atomicio, cpubudget
cpubudget.apply(cpubudget.resolve())
import cv2
import numpy as np
from orhsurf.render import _load, _frame_cameras, _look_at, _render, MODES


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--frames',default='0,1,2,3,4')
    p.add_argument('--dest',type=Path,required=True)
    args=p.parse_args()
    try:
        frames=[int(x) for x in args.frames.split(',')]
    except ValueError:
        p.error('--frames must be comma-separated integer indices')
    if not frames or any(f < 0 for f in frames) or len(set(frames)) != len(frames):
        p.error('--frames must contain unique nonnegative indices')
    for f in frames:
        if not atomicio.is_done(args.out/f'{f:05d}'):
            p.error(f'frame {f:05d} is not complete; run orhsurf verify first')
    args.dest.mkdir(parents=True,exist_ok=True)
    w,h=384,288
    xyz,nrm,rgb=_load(args.out/f'{frames[0]:05d}',max_points=1_500_000)
    K,c,r=_frame_cameras(xyz,w,h)
    cameras=[]
    for az in (35.,85.):
        a=np.radians(az)
        eye=c+np.array([np.cos(a),np.sin(a),.35])*(r*2.2)
        cameras.append(_look_at(eye,c))
    (args.dest/'cameras.json').write_text(json.dumps(dict(K=K.tolist(),world_center=c.tolist(),radius=r,
                 world_to_camera=[t.tolist() for t in cameras],frame_indices=frames,hold_seconds=1),indent=2))
    sheet=np.zeros((h*6,w*len(frames),3),np.uint8)
    for j,f in enumerate(frames):
        if j:
            xyz,nrm,rgb=_load(args.out/f'{f:05d}',max_points=1_500_000)
        views=[]
        for vi,T in enumerate(cameras):
            modes=[]
            for mi,mode in enumerate(MODES):
                img=_render(xyz,nrm,rgb,T,K,w,h,mode,splat=2)
                cv2.putText(img,f'{f:05d}  view {vi+1}  {mode}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
                sheet[(vi*3+mi)*h:(vi*3+mi+1)*h,j*w:(j+1)*w]=img
                modes.append(img)
            views.append(np.hstack(modes))
        assert cv2.imwrite(str(args.dest/f'time_{j:05d}.png'),np.vstack(views))
        print(f'rendered {f:05d}',flush=True)
    assert cv2.imwrite(str(args.dest/'contact_sheet.png'),sheet)
    threads=str(cpubudget.resolve())
    subprocess.run(['ffmpeg','-y','-v','error','-framerate','1','-i',str(args.dest/'time_%05d.png'),
                    '-frames:v',str(24*len(frames)),'-threads',threads,'-c:v','libx264','-pix_fmt','yuv420p','-r','24',
                    '-crf','20',str(args.dest/'sequence.mp4')],check=True)
    subprocess.run(['ffmpeg','-v','error','-threads',threads,'-i',str(args.dest/'sequence.mp4'),
                    '-f','null','-'],check=True)
    print('MP4 full decode passed',flush=True)

if __name__=='__main__':
    main()
