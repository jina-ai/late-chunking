import sys
import os
import pytest

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chunked_pooling.chunking import Chunker, MIN_TOKENS, DEFAULT_CHUNK_SIZE

# Test data
sample_text = """
This is a sample text. It contains multiple sentences.
This is the second paragraph. It also has multiple sentences.

This is the third paragraph. It's shorter.
"""

def test_chunk_by_tokens():
    chunker = Chunker(chunking_strategy="fixed", chunk_size=10)
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 1
    for start, end in chunks:
        assert end - start <= 10

def test_chunk_semantically():
    chunker = Chunker(chunking_strategy="semantic")
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0

def test_chunker_class():
    chunker = Chunker(chunking_strategy="fixed", chunk_size=10)
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 1
    for start, end in chunks:
        assert end - start <= 10

    chunker = Chunker(chunking_strategy="semantic")
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0

def test_invalid_chunking_strategy():
    try:
        Chunker(chunking_strategy="invalid")
    except ValueError:
        assert True
    else:
        assert False, "Should have raised ValueError"

def test_small_chunk_size():
    with pytest.raises(ValueError):
        Chunker(chunking_strategy="fixed", chunk_size=MIN_TOKENS - 1)

def test_empty_input():
    chunker = Chunker(chunking_strategy="fixed", chunk_size=10)
    chunks = chunker.chunk("")
    assert len(chunks) == 0

def test_input_shorter_than_chunk_size():
    short_text = "Short text."
    chunker = Chunker(chunking_strategy="fixed", chunk_size=20)
    chunks = chunker.chunk(short_text)
    assert len(chunks) == 1
    assert chunks[0] == (0, len(chunker.tokenizer.encode(short_text, add_special_tokens=False)))

def test_chunk_size_equal_to_input():
    text = "This is exactly ten tokens long for testing."
    chunker = Chunker(chunking_strategy="fixed", chunk_size=10)
    chunks = chunker.chunk(text)
    assert len(chunks) == 1
    assert chunks[0] == (0, 10)

def test_large_chunk_size():
    chunker = Chunker(chunking_strategy="fixed", chunk_size=1000)
    chunks = chunker.chunk(sample_text)
    assert len(chunks) == 1

def test_default_chunk_size():
    chunker = Chunker(chunking_strategy="fixed")
    assert chunker.chunk_size == DEFAULT_CHUNK_SIZE

def test_semantic_chunking_min_tokens():
    short_text = "This is a short semantic text for testing purposes. It needs to be long enough to meet the minimum token requirement."
    chunker = Chunker(chunking_strategy="semantic")
    chunks = chunker.chunk(short_text)
    assert len(chunks) == 1
    assert chunks[0][1] - chunks[0][0] >= MIN_TOKENS

def test_chunking_with_different_tokenizers():
    text = "This is a test for different tokenizers."
    chunker1 = Chunker(chunking_strategy="fixed", chunk_size=MIN_TOKENS, tokenizer="bert-base-uncased")
    chunker2 = Chunker(chunking_strategy="fixed", chunk_size=MIN_TOKENS, tokenizer="gpt2")
    
    chunks1 = chunker1.chunk(text)
    chunks2 = chunker2.chunk(text)
    
    assert len(chunks1) > 0
    assert len(chunks2) > 0
    assert chunks1 != chunks2  # Different tokenizers should produce different chunks

@pytest.mark.parametrize("chunk_size", [10, 20, 50])
def test_various_chunk_sizes(chunk_size):
    chunker = Chunker(chunking_strategy="fixed", chunk_size=chunk_size)
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0
    for start, end in chunks:
        assert end - start <= chunk_size

def test_chunk_method_with_different_strategy():
    chunker = Chunker(chunking_strategy="fixed", chunk_size=MIN_TOKENS)
    fixed_chunks = chunker.chunk(sample_text)
    semantic_chunks = chunker.chunk(sample_text, chunking_strategy="semantic")
    assert fixed_chunks != semantic_chunks

def test_chunk_method_with_custom_chunk_size():
    chunker = Chunker(chunking_strategy="fixed", chunk_size=10)
    default_chunks = chunker.chunk(sample_text)
    custom_chunks = chunker.chunk(sample_text, chunk_size=20)
    assert len(default_chunks) > len(custom_chunks)

def test_chunker_with_custom_tokenizer():
    from transformers import AutoTokenizer
    custom_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    chunker = Chunker(chunking_strategy="fixed", chunk_size=10, tokenizer=custom_tokenizer)
    chunks = chunker.chunk(sample_text)
    assert len(chunks) > 0

# Add these tests if you want to check for specific exceptions
def test_invalid_chunk_size_type():
    with pytest.raises(ValueError):
        Chunker(chunking_strategy="fixed", chunk_size="10")

def test_negative_chunk_size():
    with pytest.raises(ValueError):
        Chunker(chunking_strategy="fixed", chunk_size=-5)

def test_zero_chunk_size():
    with pytest.raises(ValueError):
        Chunker(chunking_strategy="fixed", chunk_size=0)

