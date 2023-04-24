# CUHKSZ_AcademicGraph
This repo develops an academic graph / citation graph of the School of Data Science in the Chinese University of Hongkong, Shenzhen, based on @SemanticScholar's S2AG APIs.

#### To load our dataset:
0. Library dependency: pandas, torch, torch_geometric
1. Put `CUHKSZ_AcademicGraph.py` into your module utility folder.
2. Import and create a CUHKSZ_AcademicGraph opject.


```
from utils.CUHKSZ_AcademicGraph import CUHKSZ_AcademicGraph
dataset = CUHKSZ_AcademicGraph(root=dataset_dir, with_title=True, with_label=True)
```