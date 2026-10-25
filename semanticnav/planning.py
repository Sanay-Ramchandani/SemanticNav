"""Original A* implementation plus information-gain-weighted frontier selection."""
from __future__ import annotations
import heapq
import math
import numpy as np
from .world import UNKNOWN, WALL, FLOOR, TARGET, NEIGHBORS


def manhattan(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])


def astar(grid:np.ndarray, start:tuple[int,int], goal:tuple[int,int]) -> list[tuple[int,int]] | None:
    if start==goal:
        return [start]
    if grid[goal] not in (FLOOR,TARGET):
        return None
    queue=[(manhattan(start,goal),0,start)]
    cost={start:0}; parent={}
    while queue:
        _,distance,node=heapq.heappop(queue)
        if distance!=cost[node]:
            continue
        if node==goal:
            rev=[goal]
            while rev[-1]!=start:
                rev.append(parent[rev[-1]])
            return rev[::-1]
        for dr,dc in NEIGHBORS:
            nxt=(node[0]+dr,node[1]+dc)
            if not (0<=nxt[0]<grid.shape[0] and 0<=nxt[1]<grid.shape[1]):
                continue
            if grid[nxt] not in (FLOOR,TARGET):
                continue
            new=distance+1
            if new<cost.get(nxt,math.inf):
                cost[nxt]=new
                parent[nxt]=node
                heapq.heappush(queue,(new+manhattan(nxt,goal),new,nxt))
    return None


def distance_map(grid:np.ndarray,start:tuple[int,int]):
    # BFS is significantly faster than running A* for each frontier.
    from collections import deque
    q=deque([start]); dist={start:0}; parent={}
    while q:
        node=q.popleft()
        for dr,dc in NEIGHBORS:
            nxt=(node[0]+dr,node[1]+dc)
            if not(0<=nxt[0]<grid.shape[0] and 0<=nxt[1]<grid.shape[1]):
                continue
            if nxt in dist or grid[nxt] not in (FLOOR,TARGET):
                continue
            dist[nxt]=dist[node]+1; parent[nxt]=node; q.append(nxt)
    return dist,parent


def find_frontier(grid:np.ndarray, position:tuple[int,int], radius:int=5):
    """Choose reachable observation viewpoints maximizing unseen area per travel cost."""
    distances,parents=distance_map(grid,position)
    candidates=[]
    for cell,travel in distances.items():
        r,c=cell
        adjacent_unknown=any(0<=r+dr<grid.shape[0] and 0<=c+dc<grid.shape[1] and grid[r+dr,c+dc]==UNKNOWN for dr,dc in NEIGHBORS)
        if not adjacent_unknown:
            continue
        # Approximate the number of unknown cells potentially made visible.
        gain=0
        for dr in range(-radius,radius+1):
            for dc in range(-radius,radius+1):
                rr=r+dr; cc=c+dc
                if dr*dr+dc*dc <=radius*radius and 0<=rr<grid.shape[0] and 0<=cc<grid.shape[1] and grid[rr,cc]==UNKNOWN:
                    gain+=1
        # Distance penalised, reward the expected new observations.
        score=travel-0.13*gain
        candidates.append((score, travel, r, c, cell))
    if not candidates:
        return None
    candidates.sort()
    dest=candidates[0][-1]
    path=[dest]
    while path[-1]!=position:
        path.append(parents[path[-1]])
    return path[::-1]
