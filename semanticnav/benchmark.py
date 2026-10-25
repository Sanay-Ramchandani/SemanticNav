"""Matched-seed reproducible benchmark and Wilson binomial uncertainty intervals."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from .agent import run_episode


def wilson_interval(k:int,n:int,z:float=1.9599639845):
    p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    margin=(z/den)*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return [round(center-margin,4),round(center+margin,4)]


def run_benchmark(seeds:range | list[int], dynamic:bool=True, noise:float=10.0, max_steps:int=230, output:str|Path|None=None):
    seeds=list(seeds)
    if not seeds:
        raise ValueError('Pass at least one seed')
    episodes=[run_episode(seed,strategy,max_steps=max_steps,dynamic=dynamic,noise=noise)
              for seed in seeds for strategy in ('reactive','modular')]
    summary={'description':'Original partially observed 2D colour-keyed simulator, not real robot / AI2-THOR',
             'seed_start':seeds[0],'seed_end':seeds[-1],'trials_per_policy':len(seeds),
             'step_budget':max_steps,'dynamic':dynamic,'sensor_rgb_noise_std':noise,'policies':{}}
    for strategy in ('reactive','modular'):
        rows=[r for r in episodes if r.strategy==strategy]
        success=sum(r.success for r in rows)
        summary['policies'][strategy]={
            'successful_episodes':success,
            'success_rate':round(success/len(rows),4),
            '95pct_wilson_CI':wilson_interval(success,len(rows)),
            'mean_steps':round(mean(r.steps for r in rows),2),
            'mean_collisions':round(mean(r.collisions for r in rows),3),
            'mean_path_efficiency':round(mean(r.efficiency for r in rows),4),
        }
    summary['absolute_success_gain_pp']=round(100*(summary['policies']['modular']['success_rate']-summary['policies']['reactive']['success_rate']),2)
    if output:
        path=Path(output);path.mkdir(parents=True,exist_ok=True)
        (path/'benchmark.json').write_text(json.dumps(summary,indent=2))
        with (path/'episodes.csv').open('w',newline='') as f:
            fields=['seed','strategy','success','steps','collisions','shortest_path','efficiency','reason']
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for r in episodes:
                writer.writerow(r.summary())
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--episodes',type=int,default=200)
    parser.add_argument('--seed-start',type=int,default=1000)
    parser.add_argument('--output',default='results')
    parser.add_argument('--noise',type=float,default=10)
    parser.add_argument('--max-steps',type=int,default=230)
    parser.add_argument('--static',action='store_true')
    args=parser.parse_args()
    results=run_benchmark(range(args.seed_start,args.seed_start+args.episodes),dynamic=not args.static,noise=args.noise,max_steps=args.max_steps,output=args.output)
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    main()
