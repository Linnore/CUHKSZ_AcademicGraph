import torch
import zipfile
import pandas as pd
import numpy as np
import os
import ast
from torch_geometric.data import Data, InMemoryDataset, download_url


class CUHKSZ_AcademicGraph(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, with_label = True, with_title=True):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.with_label = False
        self.with_title = False
        if self.with_label or self.with_title:
            self.process()
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
            print(raw_path)
            with zipfile.ZipFile(raw_path,"r") as zip_ref:
                zip_ref.extractall(self.raw_dir)
        
        unzip_dir = os.path.join(self.raw_dir, "CUHKSZ_AcademicGraph-rawdata_released")
        print(unzip_dir)
        with zipfile.ZipFile(os.path.join(unzip_dir, "CUHKSZ_AcademicGraph_Rawdata.zip" ),"r") as zip_ref:
            zip_ref.extractall(self.raw_dir)

        if self.with_label:
            raw_paper_info = pd.read_csv(os.path.join(unzip_dir, "Raw_Paper_Info.csv"))
            raw_paper_info.columns = raw_paper_info.columns.str.strip()
            valid = raw_paper_info["fieldsOfStudy"].notna()
            raw_paper_info = raw_paper_info[valid]

            tags = []
            for i, tag_list_str in enumerate(raw_paper_info["fieldsOfStudy"].values):
                tag_list = ast.literal_eval(tag_list_str)
                tags.append(tag_list[0])
            raw_paper_info["fieldsOfStudy"] = tags
            top8_labels = raw_paper_info["fieldsOfStudy"].value_counts()[:8]
            in_top8 = raw_paper_info["fieldsOfStudy"].map(lambda x:x in top8_labels)
            raw_paper_info = raw_paper_info[in_top8]
            raw_paper_info = raw_paper_info.reset_index()

            top8_labels_map = {}
            for i, field in enumerate(top8_labels.index):
                top8_labels_map[field] = i
            raw_paper_info["fieldsOfStudy"] = raw_paper_info["fieldsOfStudy"].map(lambda x: top8_labels_map[x])
            y = raw_paper_info["fieldsOfStudy"].values

            valid_paper_set = set()
            for paperId in raw_paper_info["paperId"]:
                valid_paper_set.add(paperId)

            # Process index mapping from semantic to our dataset
            df_reIndexMapping = pd.DataFrame({"AG_Index":raw_paper_info.index}, index=raw_paper_info["paperId"])
            reIndexMapping = dict(zip(raw_paper_info["paperId"], raw_paper_info.index))
            df_reIndexMapping.to_csv(os.path.join(self.processed_dir, "IndexMapping.csv") )


            # Process node embeding [x]
            raw_embedding = pd.read_csv(os.path.join(unzip_dir, "Raw_Paper_Embedding.csv"))
            raw_embedding = raw_embedding[valid]
            raw_embedding = raw_embedding[in_top8]
            raw_embedding["paperId"] = raw_embedding["paperId"].map(reIndexMapping)
            embedding = raw_embedding.copy()
            embedding.to_csv(os.path.join(self.processed_dir, "Embedding.csv"), index=None)
            x = embedding.drop("paperId", axis=1).values


            # Process edge information [edge_index]
            raw_citations = pd.read_csv(os.path.join(unzip_dir, "Raw_Citations.csv"))
            raw_citations.columns = raw_citations.columns.str.strip()
            valid = np.ones(raw_citations.shape[0], dtype=bool)
            i = 0
            for paperId, ref_paperId in raw_citations.values:
                if not paperId in valid_paper_set or not ref_paperId in valid_paper_set:
                    valid[i] = False
                i += 1
            print(valid)
            raw_citations = raw_citations[valid]
            raw_citations = raw_citations.reset_index()

            citations = pd.DataFrame({"paperId": raw_citations["paperId"].map(reIndexMapping),
                                    "ref_paperId": raw_citations["ref_paperId"].map(reIndexMapping)})
            citations.to_csv(os.path.join(self.processed_dir, "Citations.csv"), index=None)
            edge_index = citations.values.T

        else:
            y = None
            
            # Process index mapping from semantic to our dataset
            raw_paper_info = pd.read_csv(os.path.join(unzip_dir, "Raw_Paper_Info.csv"))
            raw_paper_info.columns = raw_paper_info.columns.str.strip()
            df_reIndexMapping = pd.DataFrame({"AG_Index":raw_paper_info.index}, index=raw_paper_info["paperId"])
            reIndexMapping = dict(zip(raw_paper_info["paperId"], raw_paper_info.index))
            df_reIndexMapping.to_csv(os.path.join(self.processed_dir, "IndexMapping.csv") )

            # Process edge information [edge_index]
            raw_citations = pd.read_csv(os.path.join(unzip_dir, "Raw_Citations.csv"))
            raw_citations.columns = raw_citations.columns.str.strip()
            citations = pd.DataFrame({"paperId": raw_citations["paperId"].map(reIndexMapping),
                                    "ref_paperId": raw_citations["ref_paperId"].map(reIndexMapping)})
            citations.to_csv(os.path.join(self.processed_dir, "Citations.csv"), index=None)
            edge_index = citations.values.T

            # Process node embeding [x]
            raw_embedding = pd.read_csv(os.path.join(unzip_dir, "Raw_Paper_Embedding.csv"))
            raw_embedding["paperId"] = raw_embedding["paperId"].map(reIndexMapping)
            embedding = raw_embedding.copy()
            embedding.to_csv(os.path.join(self.processed_dir, "Embedding.csv"), index=None)
            x = embedding.drop("paperId", axis=1).values

        if self.with_title:
            title = raw_paper_info['title'].values
        else:
            title = None
        
        print(y, title)
        # Creat Data [Graph]
        cuhksz_ag = Data(x=x, edge_index=edge_index, y=y, title=title)
        print(cuhksz_ag.y)


        # Read data into huge `Data` list.
        data_list = [cuhksz_ag]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



