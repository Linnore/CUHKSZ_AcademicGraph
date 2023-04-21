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
        for raw_path in self.raw_paths:
            with zipfile.ZipFile(raw_path,"r") as zip_ref:
                zip_ref.extractall(self.raw_dir)
        
        unzip_dir = os.path.join(self.raw_dir, "CUHKSZ_AcademicGraph-rawdata_released")
        print(unzip_dir)

        # Process index mapping from semantic to our dataset
        raw_paper_info = pd.read_csv(os.path.join(unzip_dir, "Raw_Paper_Info.csv"))
        raw_paper_info.columns = raw_paper_info.columns.str.strip()
        df_reIndexMapping = pd.DataFrame({"AG_Index":raw_paper_info.index}, index=raw_paper_info.Paper_ID)
        reIndexMapping = dict(zip(raw_paper_info.Paper_ID, raw_paper_info.index))
        df_reIndexMapping.to_csv(os.path.join(self.processed_dir, "IndexMapping.csv") )

        # Process edge information [edge_index]
        raw_citations = pd.read_csv(os.path.join(unzip_dir, "Raw_Citations.csv"))
        raw_citations.columns = raw_citations.columns.str.strip()
        citations = pd.DataFrame({"Paper_ID": raw_citations.Paper_ID.map(reIndexMapping),
                                "Ref_Paper_ID": raw_citations.Ref_Paper_ID.map(reIndexMapping)})
        citations.to_csv(os.path.join(self.processed_dir, "Citations.csv"), index=None)
        edge_index = citations.values.T

        # Process node embeding [x]
        raw_embedding = pd.read_csv(os.path.join(unzip_dir, "Raw_Paper_Embedding.csv"))
        raw_embedding.Paper_ID = raw_embedding.Paper_ID.map(reIndexMapping)
        embedding = raw_embedding.copy()
        embedding.to_csv(os.path.join(self.processed_dir, "Embedding.csv"), index=None)
        x = embedding.drop("Paper_ID", axis=1).values

        # Process node label mapping and node label[y]
        y = None
        raw_paper_info.Field_of_Study

        # Creat Data [Graph]
        cuhksz_ag = Data(x=x, edge_index=edge_index)

        # Read data into huge `Data` list.
        data_list = [cuhksz_ag]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



