import argparse
import json
from .agent import run_episode
from .benchmark import run_benchmark
from .visualize import make_gif,make_chart

def main():
    parser=argparse.ArgumentParser(description='SemanticNav robot navigation simulator')
    sub=parser.add_subparsers(dest='command',required=True)
    b=sub.add_parser('bench');b.add_argument('--episodes',type=int,default=200);b.add_argument('--seed-start',type=int,default=1000);b.add_argument('--max-steps',type=int,default=230);b.add_argument('--noise',type=float,default=10.0);b.add_argument('--static',action='store_true');b.add_argument('--output',default='results')
    d=sub.add_parser('demo');d.add_argument('--seed',type=int,default=1007);d.add_argument('--output',default='results/navigation_demo.gif')
    args=parser.parse_args()
    if args.command=='bench':
        result=run_benchmark(range(args.seed_start,args.seed_start+args.episodes),output=args.output,max_steps=args.max_steps,noise=args.noise,dynamic=not args.static)
        print(json.dumps(result,indent=2));print('Chart:',make_chart(args.output+'/benchmark.json',args.output+'/benchmark.png'))
    else:
        print(json.dumps(make_gif(seed=args.seed,output=args.output),indent=2))

if __name__=='__main__':main()
