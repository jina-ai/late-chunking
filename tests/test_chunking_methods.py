import pytest

from transformers import AutoTokenizer, AutoModel

from chunked_pooling.chunking import Chunker
from chunked_pooling.mteb_chunked_eval import AbsTaskChunkedRetrieval

EXAMPLE_TEXT_1 = "Berlin is the capital and largest city of Germany, both by area and by population. Its more than 3.85 million inhabitants make it the European Union's most populous city, as measured by population within city limits. The city is also one of the states of Germany, and is the third smallest state in the country in terms of area."
PUNCTATIONS = ('.', '!', '?')


@pytest.mark.parametrize("n_sentences", [1, 2, 3, 4])
def test_chunk_by_sentences(n_sentences):
    strategy = 'sentences'
    model_name = 'jinaai/jina-embeddings-v2-small-en'
    chunker = Chunker(strategy)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    boundary_cues = chunker.chunk(
        text=EXAMPLE_TEXT_1,
        tokenizer=tokenizer,
        chunking_strategy=strategy,
        n_sentences=n_sentences,
    )
    extended_boundary_cues = AbsTaskChunkedRetrieval._extend_special_tokens(
        boundary_cues
    )
    model_inputs = tokenizer(
        EXAMPLE_TEXT_1,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=8192,
    )

    # check that the cues start with 0 and and with the last token
    assert extended_boundary_cues[0][0] == 0
    assert len(model_inputs.tokens(0)) == extended_boundary_cues[-1][1]

    # check that all chunks but the last one end with a punctation
    assert all(
        model_inputs.tokens(0)[x:y][-1] in PUNCTATIONS
        for (x, y) in extended_boundary_cues[:-1]
    )

    # check that the last chunk ends with a "[SEP]" token
    last_cue = extended_boundary_cues[-1]
    assert model_inputs.tokens(0)[last_cue[0] : last_cue[1]][-1] == "[SEP]"

    # check that the boundary cues are continues (no token is missing)
    assert all(
        [
            extended_boundary_cues[i][1] == extended_boundary_cues[i + 1][0]
            for i in range(len(extended_boundary_cues) - 1)
        ]
    )


@pytest.mark.parametrize(
    "boundary_cues", [[(0, 17), (17, 44), (44, 69)], [(0, 44), (44, 69)]]
)
def test_token_equivalence(boundary_cues):
    model_name = 'jinaai/jina-embeddings-v2-small-en'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokens = tokenizer.encode_plus(
        EXAMPLE_TEXT_1, add_special_tokens=False, return_offsets_mapping=True
    )
    for start_token_idx, end_token_idx in boundary_cues:
        decoded_text_chunk = tokenizer.decode(
            tokens.encodings[0].ids[start_token_idx:end_token_idx]
        )

        original_text_chunk = EXAMPLE_TEXT_1[
            tokens.offset_mapping[start_token_idx][0] : tokens.offset_mapping[
                end_token_idx - 1
            ][1]
        ]
        chunk_tokens_original = tokenizer.encode_plus(original_text_chunk)
        chunk_tokens_decoded = tokenizer.encode_plus(decoded_text_chunk)
        assert chunk_tokens_original == chunk_tokens_decoded