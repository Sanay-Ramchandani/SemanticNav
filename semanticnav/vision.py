"""Semantic segmentation of the RGB observation with palette classification.

NOTE: This uses colour-keyed synthetic observations, not a learned detector or
actual physical RGB-D camera. It isolates a testable perception interface.
"""
from __future__ import annotations
import numpy as np
from .world import PALETTE, UNKNOWN


def segment_rgb(rgb:np.ndarray, max_distance:float=65.0)->np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected H x W x 3 RGB raster')
    colors=np.asarray(list(PALETTE.values()),dtype=np.float32)
    classes=np.asarray(list(PALETTE.keys()),dtype=np.int8)
    distances=np.sum((rgb.astype(np.float32)[:,:,None,:]-colors[None,None,:,:])**2,axis=-1)
    best=distances.argmin(axis=2)
    result=classes[best]
    result[np.min(distances,axis=2)>max_distance**2]=UNKNOWN
    return result


class SemanticMap:
    def __init__(self, size:int):
        self.grid=np.full((size,size),UNKNOWN,dtype=np.int8)
        self.target: tuple[int,int] | None = None

    def update(self, observation:np.ndarray, origin:tuple[int,int]):
        encoded=segment_rgb(observation)
        radius=encoded.shape[0]//2
        for rr,cc in np.argwhere(encoded != UNKNOWN):
            r=origin[0]+int(rr)-radius
            c=origin[1]+int(cc)-radius
            if 0<=r<self.grid.shape[0] and 0<=c<self.grid.shape[1]:
                value=int(encoded[rr,cc])
                self.grid[r,c]=value
                if value==2:
                    self.target=(r,c)
        self.grid[origin]=0

    def collision(self, blocked:tuple[int,int]):
        self.grid[blocked]=1
        if self.target==blocked:
            self.target=None
