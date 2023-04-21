import torch
import zipfile
import pandas as pd
import os
from torch_geometric.data import Data, InMemoryDataset, download_url


class CUHKSZ_AcademicGraph(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["CUHKSZ_AcademicGraph_Rawdata.zip"]

    @property
    def processed_file_names(self):
        return ['Citations.csv', 'Embedding.csv', 'IndexMapping.csv']

    def download(self):
        # Download to `self.raw_dir`.
        download_url("https://github.com/Linnore/CUHKSZ_AcademicGraph/archive/refs/heads/rawdata_released.zip",
                     self.raw_dir, filename="CUHKSZ_AcademicGraph_Rawdata.zip")
        ...

    def process(self):
        print("1")
        for raw_path in self.raw_paths:
            with zipfile.ZipFile(raw_path,"r") as zip_ref:
                zip_ref.extractall(self.raw_dir)

        raw_papers = pd.read_csv(os.path.join(self.raw_dir, "Raw_Paper_Info.csv"))
        raw_papers.columns = raw_papers.columns.str.strip()
        df_reIndexMapping = pd.DataFrame({"AG_Index":raw_papers.index}, index=raw_papers.Paper_ID)
        reIndexMapping = dict(zip(raw_papers.Paper_ID, raw_papers.index))
        df_reIndexMapping.to_csv(os.path.join(self.processed_dir, "IndexMapping.csv") )

        raw_citations = pd.read_csv(os.path.join(self.raw_dir, "Raw_Citations.csv"))
        raw_citations.columns = raw_citations.columns.str.strip()
        citations = pd.DataFrame({"Paper_ID": raw_citations.Paper_ID.map(reIndexMapping),
                                "Ref_Paper_ID": raw_citations.Ref_Paper_ID.map(reIndexMapping)})
        citations.to_csv(os.path.join(self.raw_dir, "Citations.csv"), index=None)
        citations.to_csv(os.path.join(self.processed_dir, "Citations.csv"), index=None)

        raw_embedding = pd.read_csv(os.path.join(self.raw_dir, "Raw_Paper_Embedding.csv"))
        raw_embedding.Paper_ID = raw_embedding.Paper_ID.map(reIndexMapping)
        embedding = raw_embedding.copy()
        embedding.to_csv(os.path.join(self.processed_dir, "Embedding.csv"), index=None)

        # Read data into huge `Data` list.
        data_list = [...]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
