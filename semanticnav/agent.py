"""Modular explorer and memoryless reactive comparison, both using local vision."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .world import GridWorld, NEIGHBORS, FLOOR, WALL, TARGET
from .vision import SemanticMap, segment_rgb
from .planning import astar, find_frontier, manhattan


@dataclass
class Episode:
    seed:int
    strategy:str
    success:bool
    steps:int
    collisions:int
    shortest_path:int
    efficiency:float
    visited:list[tuple[int,int]]
    reason:str

    def summary(self):
        return {k:v for k,v in vars(self).items() if k!='visited'}


class ModularAgent:
    name='modular'

    def __init__(self,size:int,radius:int):
        self.map=SemanticMap(size)
        self.radius=radius
        self.replans=0

    def choose(self,world:GridWorld):
        self.map.update(world.observe(),world.pos)
        if self.map.target is not None:
            path=astar(self.map.grid,world.pos,self.map.target)
            if path and len(path)>1:
                self.replans+=1
                return path[1], 'target path'
        path=find_frontier(self.map.grid,world.pos,self.radius)
        if path and len(path)>1:
            self.replans+=1
            return path[1], 'frontier exploration'
        return None, 'no reachable unexplored frontier'

    def feedback(self,next_cell,step):
        if step.collided:
            self.map.collision(next_cell)


class ReactiveBaseline:
    """Memoryless image-only random walk, NOT the published learned baseline."""
    name='reactive'

    def __init__(self,size:int,radius:int,seed:int):
        self.radius=radius
        self.rng=np.random.default_rng(seed+710202)

    def choose(self,world:GridWorld):
        image=segment_rgb(world.observe())
        radius=self.radius
        moves=[]; observed_target=[]
        target_candidates=np.argwhere(image==TARGET)
        target=tuple(target_candidates[0]-radius) if len(target_candidates) else None
        for dr,dc in NEIGHBORS:
            pixel=(radius+dr,radius+dc)
            if image[pixel] in (FLOOR,TARGET):
                dest=(world.pos[0]+dr,world.pos[1]+dc)
                moves.append(dest)
                if target:
                    observed_target.append((manhattan((dr,dc),target),dest))
        if observed_target:
            observed_target.sort(key=lambda pair:pair[0])
            return observed_target[0][1],'reactive visible-goal pursuit'
        if moves:
            return moves[int(self.rng.integers(len(moves)))], 'memoryless exploration'
        return None,'no visible valid moves'

    def feedback(self,next_cell,step):
        return


def run_episode(seed:int, strategy:str='modular',size:int=19,radius:int=5,max_steps:int=230,dynamic:bool=False,noise:float=0, capture:bool=False):
    world=GridWorld(seed,size=size,radius=radius,dynamic=dynamic,noise=noise)
    from .world import accessible
    # Ground truth shortest path is for evaluation only. The agent does not receive it.
    from collections import deque
    queue=deque([(world.start,0)]); seen={world.start}; shortest=-1
    while queue:
        pos,d=queue.popleft()
        if pos==world.target:
            shortest=d;break
        for dr,dc in NEIGHBORS:
            nxt=(pos[0]+dr,pos[1]+dc)
            if 0<=nxt[0]<size and 0<=nxt[1]<size and world.grid[nxt]!=WALL and nxt not in seen:
                seen.add(nxt);queue.append((nxt,d+1))
    agent=ModularAgent(size,radius) if strategy=='modular' else ReactiveBaseline(size,radius,seed) if strategy=='reactive' else None
    if agent is None:
        raise ValueError('strategy must be modular or reactive')
    trail=[world.pos]; frames=[world.truth_image()] if capture else []
    reason='step budget exhausted'; success=False
    for _ in range(max_steps):
        next_cell,message=agent.choose(world)
        if next_cell is None:
            reason=message
            break
        feedback=world.step(next_cell)
        agent.feedback(next_cell,feedback)
        trail.append(world.pos)
        if capture:
            frames.append(world.truth_image())
        if feedback.succeeded:
            reason='target reached'; success=True
            break
    # SPL-like metric: shortest path / actual steps, zero on failure.
    efficiency=(shortest / max(shortest,world.steps)) if success else 0.0
    episode=Episode(seed,strategy,success,world.steps,world.collisions,shortest,round(efficiency,4),trail,reason)
    return (episode, frames) if capture else episode
