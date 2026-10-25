import numpy as np
import pytest
from semanticnav.world import GridWorld, FLOOR, WALL, TARGET, PALETTE, line_of_sight
from semanticnav.vision import SemanticMap, segment_rgb
from semanticnav.planning import astar, find_frontier
from semanticnav.agent import run_episode
from semanticnav.benchmark import run_benchmark, wilson_interval


def test_world_seed_reproducible():
    a=GridWorld(234);b=GridWorld(234)
    assert (a.grid==b.grid).all()
    assert a.start==b.start and a.target==b.target


def test_wall_blocks_rays():
    grid=np.zeros((9,9),dtype=np.int8)
    grid[4,4]=WALL
    assert line_of_sight((4,1),(4,4),grid)
    assert not line_of_sight((4,1),(4,7),grid)


def test_synthetic_camera_never_reveals_distant_object():
    world=GridWorld(100)
    if abs(world.pos[0]-world.target[0])+abs(world.pos[1]-world.target[1])>world.radius*2:
        assert not (segment_rgb(world.observe())==TARGET).any()


def test_palette_segmentation_with_noise():
    image=np.asarray([[PALETTE[FLOOR],PALETTE[WALL],PALETTE[TARGET]]],dtype=np.uint8)
    assert segment_rgb(image).tolist()==[[FLOOR,WALL,TARGET]]
    noisy=np.clip(image.astype(int)+10,0,255).astype(np.uint8)
    assert segment_rgb(noisy).tolist()==[[FLOOR,WALL,TARGET]]


def test_known_map_updates():
    world=GridWorld(11)
    semantic=SemanticMap(world.size)
    semantic.update(world.observe(),world.pos)
    assert semantic.grid[world.pos]==FLOOR
    assert np.sum(semantic.grid!=-1)<world.size*world.size


def test_astar_detour_and_unreachable():
    grid=np.zeros((6,6),dtype=np.int8)
    grid[2,1:5]=WALL
    path=astar(grid,(1,2),(4,2))
    assert path is not None
    assert len(path)>4
    assert all(grid[pos]!=WALL for pos in path)
    grid[2,:]=WALL
    assert astar(grid,(1,2),(4,2)) is None


def test_frontier_reachable():
    grid=np.full((9,9),-1,dtype=np.int8)
    grid[2:6,2:6]=FLOOR
    path=find_frontier(grid,(3,3),3)
    assert path is not None
    assert path[0]==(3,3)
    assert all(grid[pos]==FLOOR for pos in path)


def test_step_validity():
    world=GridWorld(11)
    with pytest.raises(ValueError):
        world.step(world.pos)


def test_agents_and_replanning():
    episodes=[run_episode(i, 'modular',dynamic=True,noise=10) for i in range(15)]
    assert all(e.success for e in episodes)
    assert any(e.collisions>0 for e in episodes)
    assert all(e.steps<=230 for e in episodes)
    assert all(0<e.efficiency<=1 for e in episodes)


def test_reactive_is_deterministic():
    a=run_episode(7,'reactive',noise=0)
    b=run_episode(7,'reactive',noise=0)
    assert a.summary()==b.summary()


def test_matching_seed_benchmark(tmp_path):
    summary=run_benchmark(range(1000,1020),output=tmp_path)
    assert summary['trials_per_policy']==20
    assert summary['policies']['modular']['success_rate']>.8
    assert 0<=summary['policies']['reactive']['success_rate']<=1
    assert (tmp_path/'episodes.csv').exists()
    assert (tmp_path/'benchmark.json').exists()


def test_confidence_interval():
    low,high=wilson_interval(50,100)
    assert .39<low<.41 and .59<high<.61
