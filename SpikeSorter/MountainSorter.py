from SpikeSorter.BaseSorter import BaseSorter
from mountainlab_pytools import mlproc as mlp
from SpikeSorter.mountainsort4_1_0 import sort_dataset as ms4_sort_dataset # MountainSort spike sorting


class MountainSorter(BaseSorter):
    def __init__(self,datafolder,adjacency_radius=-1,detect_threshold=3):
        super().__init__()
        self.datafolder = datafolder
        self.sorterName = 'mountainsort'
        self.pipeline = mlp.initPipeline()
        self.adjacency_radius = adjacency_radius
        self.detect_threshold = detect_threshold

    def run(self):
        with self.pipeline:
            ms4_sort_dataset(self.datafolder,self.datafolder,adjacency_radius,detect_threshold)


    def cleanup(self):
        pass