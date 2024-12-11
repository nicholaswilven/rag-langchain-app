import logging
import os
from math import ceil
from typing import List
from tqdm import tqdm

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

class StellaEmbeddingModel:
    def __init__(self, model_name: str = "dunzhang/stella_en_400M_v5", vector_dim: int = 256, query_prompt = None):
        self.model_name = model_name
        self.vector_dim = vector_dim
        if query_prompt is None:
            self.query_prompt = "Instruct: Given a menu order query, retrieve relevant menu to the query.\nQuery: "

    def download_model(self):        
        self.local_dir = os.path.join("./data/models", self.model_name.replace("/", "--"))
        os.makedirs(self.local_dir, exist_ok = True)
        
        api = HfApi()
        files = api.list_repo_files(repo_id = self.model_name)

        # Download base files
        for file in files:
            if "/" not in file:  # Exclude folders (no '/' in file path)
                complete_path = os.path.join(self.local_dir, file)
                if not os.path.exists(complete_path):
                    foldername = os.path.dirname(complete_path)
                    filename = os.path.basename(complete_path)
                    downloaded_file = hf_hub_download(repo_id = self.model_name, filename = filename, local_dir = foldername)
                    logger.info(f"Downloaded: {downloaded_file}")
        
        # Download pooling layer
        for file in files:
            if file.startswith("1_Pooling/"):
                complete_path = os.path.join(self.local_dir, file)
                if not os.path.exists(complete_path):
                    foldername = os.path.dirname(complete_path)
                    filename = os.path.basename(complete_path)
                    downloaded_file = hf_hub_download(repo_id = self.model_name, filename = filename, local_dir = foldername)
                    logger.info(f"Downloaded: {downloaded_file}")     
        
        # Download specific dense layer
        for file in files:
            if file.startswith(f"2_Dense_{self.vector_dim}/"):
                complete_path = os.path.join(self.local_dir, file)
                if not os.path.exists(complete_path):
                    foldername = os.path.dirname(complete_path)
                    filename = os.path.basename(complete_path)
                    downloaded_file = hf_hub_download(repo_id = self.model_name, filename = filename, local_dir = foldername)
                    logger.info(f"Downloaded: {downloaded_file}")
                
        logger.info(f"All files downloaded to: {self.local_dir}")

    def load_model(self):
        vector_linear_directory = f"2_Dense_{self.vector_dim}"
        
        self.model = AutoModel.from_pretrained(self.local_dir, trust_remote_code = True, device_map = "cuda").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir, trust_remote_code = True)
        self.vector_linear = torch.nn.Linear(in_features = self.model.config.hidden_size, out_features = self.vector_dim)
        
        vector_linear_dict = {
            k.replace("linear.", ""): v for k, v in
            torch.load(os.path.join(self.local_dir, f"{vector_linear_directory}/pytorch_model.bin")).items()
        }
        self.vector_linear.load_state_dict(vector_linear_dict)
        self.vector_linear.cuda()

    def encode(self, queries: List[str], infer_batch_size: int = 32, use_prompt: bool = True)->torch.Tensor:
        if use_prompt:
            queries = [self.query_prompt + query for query in queries]
        query_embeddings = []
        batch_num = ceil(len(queries)/infer_batch_size)
        for idx in tqdm(range(batch_num), desc = f"Encoding text queries using {self.model_name}"):
            sub_queries = list(queries[infer_batch_size*idx:infer_batch_size*(idx+1)])
            with torch.no_grad():
                input_data = self.tokenizer(sub_queries, padding = "longest", truncation = True, max_length =  512, return_tensors = "pt")
                input_data = {k: v.cuda() for k, v in input_data.items()}
                attention_mask = input_data["attention_mask"]
                last_hidden_state = self.model(**input_data)[0]
                last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
                embeddings = last_hidden.sum(dim = 1) / attention_mask.sum(dim = 1)[..., None]
                embeddings = normalize(self.vector_linear(embeddings).cpu().numpy())
            query_embeddings.append(embeddings)
        return np.vstack(query_embeddings, dim = 0)
    
    def similarity(self, query_embeddings: np.ndarray, doc_embeddings: np.ndarray)->np.ndarray:
        return self.model.similarity(query_embeddings, doc_embeddings)

