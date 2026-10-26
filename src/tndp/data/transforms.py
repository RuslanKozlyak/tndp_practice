"""Преобразования графов для обучения и валидации."""
from dataclasses import dataclass
from torch_geometric.transforms import RandomRotate
from tndp.data.graph import (
    to_batch,
    SpaceScaleTransform,
    DemandScaleTransform,
    RandomFlipCity,
    InsertPosFeatures,
)
from omegaconf import OmegaConf

@dataclass(frozen=True)
class PipelineParams:
    space_scale: float
    demand_scale: float
    flip: bool
    rotate: float
    edit_features: bool

class GraphPipeline:
    def __init__(self, params: PipelineParams):
        if not 0 <= params.space_scale < 1 or not 0 <= params.demand_scale < 1:
            raise ValueError('pipeline scaling must be between zero (included) and one (excluded)')
        self.params = params
        transforms = []
        if params.space_scale:
            transforms.append(SpaceScaleTransform(1 - params.space_scale, 1 + params.space_scale))
        if params.demand_scale:
            transforms.append(DemandScaleTransform(1 - params.demand_scale, 1 + params.demand_scale))
        if params.flip:
            transforms.append(RandomFlipCity())
        if params.rotate:
            transforms.append(RandomRotate(params.rotate))
        if not params.edit_features:
            transforms.append(InsertPosFeatures())
        self.transforms = tuple(transforms)

    def prepare(self, graphs, device=None):
        return to_batch(graphs, device, transforms=self.transforms, edit_features=self.params.edit_features)

    def graph(self, graph):
        return self.prepare([graph]).to_data_list()[0]

class GraphView:
    def __init__(self, graphs, indices, pipeline):
        self.graphs, self.indices, self.pipeline = graphs, tuple(int(i) for i in indices), pipeline

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.pipeline.graph(self.graphs[self.indices[index]])

def pipeline(values):
    return GraphPipeline(PipelineParams(**OmegaConf.to_container(values, resolve=True)))
