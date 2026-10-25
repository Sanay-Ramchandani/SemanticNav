"""Create a replayable animation and a reproducible evaluation figure."""
from __future__ import annotations
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from .agent import run_episode


def make_gif(seed:int=1007,output='results/navigation_demo.gif',dynamic=True,noise=10):
    record, frames=run_episode(seed,'modular',capture=True,dynamic=dynamic,noise=noise)
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
    rendered=[]
    for idx,frame in enumerate(frames):
        rgb=Image.fromarray(frame)
        canvas=Image.new('RGB',(rgb.width,rgb.height+38),'#101827')
        canvas.paste(rgb,(0,38))
        draw=ImageDraw.Draw(canvas)
        draw.text((9,11),f'SemanticNav | seed {seed} | step {idx:03}',fill='#f8fafc')
        rendered.append(canvas)
    rendered[0].save(dest,save_all=True,append_images=rendered[1:],duration=95,loop=0,optimize=True)
    return {'file':str(dest),'seed':seed,'success':record.success,'steps':record.steps,'collisions':record.collisions}


def make_chart(source='results/benchmark.json',destination='results/benchmark.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=json.loads(Path(source).read_text())
    labels=['Reactive baseline','Modular agent']
    keys=['reactive','modular']
    numbers=[100*r['policies'][k]['success_rate'] for k in keys]
    ranges=[r['policies'][k]['95pct_wilson_CI'] for k in keys]
    fig,ax=plt.subplots(figsize=(7.5,4.8))
    bars=ax.bar(labels,numbers,width=.52)
    for i,b in enumerate(bars):
        low,high=ranges[i]; y=numbers[i]
        ax.errorbar(i,y,yerr=[[y-low*100],[high*100-y]],fmt='none',capsize=6,color='black')
        ax.text(i,y+2,f'{y:.1f}%',ha='center',fontsize=12)
    ax.set_ylim(0,112)
    ax.set_ylabel('Object-goal task success (%)')
    ax.set_title(f'Matched-seed navigation comparison (n = {r["trials_per_policy"]} per policy)')
    ax.text(.5,-.17,'Synthetic colour-keyed grid worlds. Not comparable to published real-robot results.',
            ha='center',transform=ax.transAxes,fontsize=9)
    fig.tight_layout()
    out=Path(destination);out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=160,bbox_inches='tight')
    plt.close(fig)
    return out

if __name__=='__main__':
    print(make_gif());print(make_chart())
