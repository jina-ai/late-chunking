import bisect
import logging
from typing import Dict, List, Optional, Tuple, Union

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer

# Set the logging level to WARNING to suppress INFO and DEBUG messages
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

MIN_TOKENS = 10
DEFAULT_CHUNK_SIZE = 256
BUFFER_SIZE = 1
BREAKPOINT_PERCENTILE_THRESHOLD = 0.98
EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v2-small-en"


def setup_semantic_chunking(
    self,
    buffer_size: int = BUFFER_SIZE,
    breakpoint_percentile_threshold: float = BREAKPOINT_PERCENTILE_THRESHOLD,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
):
    self.buffer_size = buffer_size
    self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
    self.embed_model = HuggingFaceEmbedding(
        model_name=embedding_model_name,
        max_length=512,
        trust_remote_code=True,
    )
    self.splitter = SemanticSplitterNodeParser(
        buffer_size=self.buffer_size,
        breakpoint_percentile_threshold=self.breakpoint_percentile_threshold,
        embed_model=self.embed_model,
        show_progress=False,
    )


class Chunker:
    def __init__(
        self,
        chunking_strategy: str = 'fixed',
        tokenizer: Optional[Union[str, 'AutoTokenizer']] = None,
        buffer_size: int = BUFFER_SIZE,
        breakpoint_percentile_threshold: float = BREAKPOINT_PERCENTILE_THRESHOLD,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        self.chunking_strategy = chunking_strategy
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer,
                force_download=True,
                trust_remote_code=True,
            )
        else:
            self.tokenizer = tokenizer
        if self.chunking_strategy == "semantic":
            self.buffer_size = buffer_size
            self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
            self.embed_model = HuggingFaceEmbedding(
                model_name=embedding_model_name,
                max_length=512,
                trust_remote_code=True,
            )
            self.splitter = SemanticSplitterNodeParser(
                buffer_size=self.buffer_size,
                breakpoint_percentile_threshold=self.breakpoint_percentile_threshold,
                embed_model=self.embed_model,
                show_progress=False,
            )
        elif self.chunking_strategy == "fixed":
            self.chunk_size = chunk_size
        else:
            raise ValueError("Unsupported chunking strategy")
        self.min_tokens = MIN_TOKENS

    def chunk_semantically(
        self, text: str, min_tokens: Optional[int] = None
    ) -> List[Tuple[int, int, int]]:
        if self.embed_model is None:
            setup_semantic_chunking()

        min_tokens = min_tokens or self.min_tokens

        nodes = [
            (node.start_char_idx, node.end_char_idx)
            for node in self.splitter.get_nodes_from_documents(
                [Document(text=text)], show_progress=False
            )
        ]
        # Tokenize the entire text
        tokens = self.tokenizer.encode_plus(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            max_length=512,
            padding=True,
            truncation=True,
        )
        token_offsets = tokens.offset_mapping

        chunk_spans = []

        if len(token_offsets) < min_tokens:
            # If the entire text has fewer than 10 tokens, return it as a single chunk
            chunk_spans.append((0, len(token_offsets) - 1))
            return chunk_spans

        i = 0
        while i < len(nodes):
            char_start, char_end = nodes[i]

            # convert char_start and char_end to token indices
            start_chunk_index = bisect.bisect_left(
                [offset[0] for offset in token_offsets], char_start
            )
            end_chunk_index = (
                bisect.bisect_right([offset[1] for offset in token_offsets], char_end)
                - 1
            )

            # Ensure each chunk has at least min_tokens tokens
            while (
                end_chunk_index - start_chunk_index + 1 < min_tokens
                and i < len(nodes) - 1
            ):
                # Merge with the next node
                i += 1
                char_end = nodes[i][1]
                end_chunk_index = (
                    bisect.bisect_right(
                        [offset[1] for offset in token_offsets], char_end
                    )
                    - 1
                )

            # If the chunk is still less than min_tokens and it's the last node, handle it explicitly
            if (
                end_chunk_index - start_chunk_index + 1 < min_tokens
                and i == len(nodes) - 1
            ):
                end_chunk_index = min(
                    start_chunk_index + min_tokens - 1, len(token_offsets) - 1
                )

            # If the chunk is outside of the tokenized text, break out of loop
            if start_chunk_index >= len(token_offsets) or end_chunk_index >= len(
                token_offsets
            ):
                break

            chunk_spans.append((start_chunk_index, end_chunk_index))
            i += 1

        return chunk_spans

    def chunk_by_tokens(
        self,
        text: str,
        chunk_size: Optional[int] = None,
    ) -> List[Tuple[int, int, int]]:
        chunk_size = chunk_size or self.chunk_size
        tokens = self.tokenizer.encode_plus(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
        token_offsets = tokens.offset_mapping

        chunk_spans = []
        for i in range(0, len(token_offsets), chunk_size):
            chunk_end = min(i + chunk_size, len(token_offsets) - 1)
            if chunk_end - i > 0:
                chunk_spans.append((i, chunk_end))

        return chunk_spans

    def chunk(
        self,
        text: str,
        tokenizer: 'AutoTokenizer' = None,
        chunking_strategy: str = None,
        chunk_size: Optional[int] = None,
    ):
        if chunk_size < 10:
            raise ValueError("Chunk size must be greater than 10.")

        if tokenizer and not self.tokenizer:
            self.tokenizer = tokenizer
        if chunking_strategy == "semantic":
            return self.chunk_semantically(text)
        elif chunking_strategy == "fixed":
            return self.chunk_by_tokens(text, chunk_size=chunk_size)
        else:
            raise ValueError("Unsupported chunking strategy")
