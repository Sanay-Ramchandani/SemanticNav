"""A reproducible, partially observed, RGB-rendered indoor navigation simulator.

The agent is never given the ground-truth occupancy map. It receives a local
RGB semantic raster, simulated odometry, and whether its movement collided.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import numpy as np

UNKNOWN = -1
FLOOR = 0
WALL = 1
TARGET = 2
PALETTE = {
    UNKNOWN: (104, 109, 117),
    FLOOR: (234, 228, 212),
    WALL: (39, 55, 74),
    TARGET: (232, 64, 64),
}
NEIGHBORS = ((-1,0), (0,1), (1,0), (0,-1))


def accessible(grid: np.ndarray, start: tuple[int,int], target: tuple[int,int]) -> bool:
    queue = deque([start]); seen = {start}
    while queue:
        pos = queue.popleft()
        if pos == target:
            return True
        for dr,dc in NEIGHBORS:
            nxt = (pos[0]+dr, pos[1]+dc)
            if 0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1] and grid[nxt] != WALL and nxt not in seen:
                seen.add(nxt); queue.append(nxt)
    return False


def line_of_sight(a: tuple[int,int], b: tuple[int,int], world: np.ndarray) -> bool:
    """Integer Bresenham traversal; destination wall itself remains visible."""
    r0,c0=a; r1,c1=b
    dc=abs(c1-c0); dr=-abs(r1-r0)
    sc=1 if c0<c1 else -1; sr=1 if r0<r1 else -1
    err=dc+dr; r,c=r0,c0
    while (r,c)!=(r1,c1):
        # Origin is always free. Intermediate walls occlude, final wall is visible.
        if (r,c)!=(r0,c0) and world[r,c]==WALL:
            return False
        twice=2*err
        if twice >= dr:
            err += dr; c += sc
        if twice <= dc:
            err += dc; r += sr
    return True


@dataclass
class StepFeedback:
    moved: bool
    collided: bool
    succeeded: bool


class GridWorld:
    def __init__(self, seed: int, size: int=19, radius: int=5, dynamic: bool=False, noise: float=0.0):
        if size < 11 or radius < 2 or radius >= size // 2:
            raise ValueError('Require size>=11, radius>=2, radius<size//2')
        self.rng=np.random.default_rng(seed)
        self.seed=seed
        self.size=size
        self.radius=radius
        self.dynamic=dynamic
        self.noise=noise
        self.grid=np.zeros((size,size),dtype=np.int8)
        self.grid[[0,-1],:]=WALL; self.grid[:,[0,-1]]=WALL
        # Rectangular furniture and narrow passageways. Regenerate until reachable.
        for _ in range(100):
            self.grid[1:-1,1:-1] = FLOOR
            for _ in range(size//2 + 5):
                r=int(self.rng.integers(2,size-3)); c=int(self.rng.integers(2,size-3))
                h=int(self.rng.integers(1,4)); w=int(self.rng.integers(1,4))
                self.grid[r:min(size-1,r+h), c:min(size-1,c+w)]=WALL
            empty=np.argwhere(self.grid==FLOOR)
            if len(empty)<size*3:
                continue
            picks=self.rng.choice(len(empty),size=2,replace=False)
            start=tuple(map(int,empty[picks[0]])); target=tuple(map(int,empty[picks[1]]))
            if abs(start[0]-target[0])+abs(start[1]-target[1])>=size//2 and accessible(self.grid,start,target):
                self.start=start; self.target=target; self.pos=start; break
        else:
            raise RuntimeError('Could not generate reachable map')
        self.steps=0
        self.collisions=0
        self.dynamic_placed=False

    def _maybe_change(self, attempted: tuple[int,int]):
        if not self.dynamic or self.dynamic_placed or self.steps != 9:
            return
        # Introduce a previously unobserved moving obstruction exactly on the
        # planned next step, but only when an alternate route remains possible.
        # This makes collision sensing and reactive replanning testable.
        if attempted == self.target or self.grid[attempted] != FLOOR:
            return
        self.grid[attempted] = WALL
        if accessible(self.grid,self.pos,self.target):
            self.dynamic_placed = True
        else:
            self.grid[attempted] = FLOOR

    def observe(self) -> np.ndarray:
        """Local RGB semantic image. Occluded/out-of-bounds cells are UNKNOWN."""
        width=2*self.radius+1
        image=np.empty((width,width,3),dtype=np.uint8)
        image[:]=PALETTE[UNKNOWN]
        for dr in range(-self.radius,self.radius+1):
            for dc in range(-self.radius,self.radius+1):
                r=self.pos[0]+dr; c=self.pos[1]+dc
                if 0<=r<self.size and 0<=c<self.size and dr*dr+dc*dc<=self.radius*self.radius:
                    if line_of_sight(self.pos,(r,c),self.grid):
                        key=TARGET if (r,c)==self.target else int(self.grid[r,c])
                        image[dr+self.radius,dc+self.radius]=PALETTE[key]
        if self.noise:
            offsets=self.rng.normal(0,self.noise,image.shape)
            image=np.clip(image.astype(float)+offsets,0,255).astype(np.uint8)
        return image

    def step(self, new_position: tuple[int,int]) -> StepFeedback:
        if sum(abs(a-b) for a,b in zip(new_position,self.pos))!=1:
            raise ValueError('Only four-connected unit movements are permitted')
        self.steps+=1
        self._maybe_change(new_position)
        if self.grid[new_position] == WALL:
            self.collisions+=1
            return StepFeedback(False,True,False)
        self.pos=new_position
        return StepFeedback(True,False,self.pos==self.target)

    def truth_image(self, scale:int=16) -> np.ndarray:
        img=np.zeros((self.size,self.size,3),dtype=np.uint8)
        for key,color in PALETTE.items():
            img[self.grid==key]=color
        img[self.target]=PALETTE[TARGET]
        img[self.pos]=(65,150,247)
        return np.repeat(np.repeat(img,scale,axis=0),scale,axis=1)
