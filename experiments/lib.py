# pip requirements:
# accelerate?

import numpy as np

import chunked_pooling
from chunked_pooling import chunked_pooling
from chunked_pooling.chunking import Chunker

from typing import List, Tuple
from transformers import AutoModel, AutoTokenizer, pipeline

import requests
import os

def request_anthropic_api(prompt: str):
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 2048,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()["content"][0]["text"]

def cosine_similarity(vector1, vector2):
    vector1_norm = vector1 / np.linalg.norm(vector1)
    vector2_norm = vector2 / np.linalg.norm(vector2)
    return np.dot(vector1_norm, vector2_norm)

class LateChunkingEmbedder:

    def __init__(self, 
            model: AutoModel,
            tokenizer: AutoTokenizer, 
            chunking_strategy: str = "sentences",
            n_sentences: int = 1
        ):

        self.model = model
        self.tokenizer = tokenizer

        self.chunker = Chunker(chunking_strategy = chunking_strategy)
        self.n_sentences = n_sentences

    
    def run(self, document: str):
        annotations = [self.chunker.chunk(text=document, tokenizer=self.tokenizer, n_sentences=self.n_sentences)]
        model_inputs = self.tokenizer(
            document,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=8192,
        )
        model_outputs = self.model(**model_inputs)
        self.output_embs = chunked_pooling(
            model_outputs, annotations, max_length=8192, 
        )[0]
        return self.output_embs

    def query(self, query: str):
        if "output_embs" not in dir(self):
            raise ValueError("no embeddings calculated, use .run(document) to create chunk embeddings")
        query_embedding = self.model.encode(query)
        similarities = []
        for emb in self.output_embs:
            similarities.append(cosine_similarity(query_embedding, emb))
        
        return similarities


class ContextualRetrievalEmbedder():
    def __init__(self, 
            model: AutoModel,
            tokenizer: AutoTokenizer, 
            llm_name: str = "meta-llama/Meta-Llama-3.1-8B",
            chunking_strategy: str = "fixed"
        ):
        # self.llm = pipeline(
        #     "text-generation", model=llm_name, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto",
        #     max_length = 1000
        # )
        self.llm = request_anthropic_api

        self.prompt = """
        <document> 
        {{WHOLE_DOCUMENT}} 
        </document> 
        Here is the chunk we want to situate within the whole document 
        <chunk> 
        {{CHUNK_CONTENT}} 
        </chunk> 
        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. 
        """.strip()

        self.model = model
        self.tokenizer = tokenizer

        self.chunker = Chunker(chunking_strategy = chunking_strategy)


    def _add_context(self, chunk: str, document: str):
        prompt = self.prompt.replace("{{WHOLE_DOCUMENT}}", document).replace("{{CHUNK_CONTENT}}", chunk)
        extra_context = self.llm(prompt)
        return extra_context + " " + chunk

    def _tokens_to_text(self, text: str, annotations: List[Tuple[int, int]]):
        tokens = self.tokenizer.encode_plus(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
        token_offsets = tokens.offset_mapping
        chunks = []
        for start, end in annotations:
            chunk = text[token_offsets[start][0]:token_offsets[end-1][1]]
            chunks.append(chunk)
        return chunks
    
    def run(self, document: str):
        annotations = [self.chunker.chunk(text=document, tokenizer=self.tokenizer, n_sentences=1)]
        self.chunks = self._tokens_to_text(text=document, annotations=annotations[0])
        self.chunks = [self._add_context(chunk, document) for chunk in self.chunks]

        model_outputs = self.model.encode(self.chunks)
        self.output_embs = [model_outputs[i, :] for i in range(len(self.chunks))]
        return self.output_embs

    def query(self, query: str):
        if "output_embs" not in dir(self):
            raise ValueError("no embeddings calculated, use .run(document) to create chunk embeddings")
        query_embedding = self.model.encode(query)
        similarities = []
        for emb in self.output_embs:
            similarities.append(cosine_similarity(query_embedding, emb))
        
        return similarities

        