# soft_token_boundary.py
#
# long context embedding models (like jina-embeddings-v2-base-en) have a limit of 8192 tokens
# for longer context documents, such as those in LongEmbed benchmarks, how do we embed them?
#
# method 1: truncate the document at the token boundary
# method 2: embed twice, before and after the truncation point as many times as needed
# method 3: same as method 2 but with overlap
#
# obviously method 1 is not great. any information after the 8192nd token is discarded.
# method 3 should perform the best, but how much better is it?
# 
# and do we need to look into reducing the size of context window?
#
# let's use the WikimQA dataset to test these ideas, just method 2 and 3.
#
# let's also try to use the mteb benchmark to evaluate the results.

from chunked_pooling.wrappers import load_model
from transformers import AutoModel, AutoTokenizer, pipeline
# from experiments.lib import ContextualRetrievalEmbedder, LateChunkingEmbedder


import click
import torch.cuda
from mteb import MTEB
from transformers import AutoModel, AutoTokenizer

from chunked_pooling.chunked_eval_tasks import (
    LEMBWikimQARetrievalChunked, 
    LEMBNeedleRetrievalChunked, 
    LEMBNarrativeQARetrievalChunked,
    LEMBQMSumRetrievalChunked,
    LEMBSummScreenFDRetrievalChunked
)
from chunked_pooling.wrappers import load_model

DEFAULT_CHUNKING_STRATEGY = 'fixed'
DEFAULT_CHUNK_SIZE = 256
DEFAULT_N_SENTENCES = 5
BATCH_SIZE = 1
DEFAULT_SOFT_BOUNDARY_EMBED_SIZE = 8192
DEFAULT_HARD_BOUNDARY_EMBED_SIZE = 0



@click.command()
@click.option(
    '--model-name',
    default='jinaai/jina-embeddings-v2-small-en',
    help='The name of the model to use.',
)
@click.option(
    '--strategy',
    default=DEFAULT_CHUNKING_STRATEGY,
    help='The chunking strategy to be applied.',
)
@click.option(
    '--task-name', default='LEMBWikimQARetrievalChunked', help='The evaluation task to perform.'
)
@click.option(
    '--eval-split', default='test', help='The name of the evaluation split in the task.'
)
@click.option(
    '--chunking-model',
    default=None,
    required=False,
    help='The name of the model used for semantic chunking.',
)
@click.option(
    '--truncate-max-length',
    default=8192,
    type=int,
    help='Maximum number of tokens; By default, no truncation is done.',
)
@click.option(
    '--chunk-size',
    default=DEFAULT_CHUNK_SIZE,
    type=int,
    help='Number of tokens per chunk for fixed strategy.',
)
@click.option(
    '--n-sentences',
    default=DEFAULT_N_SENTENCES,
    type=int,
    help='Number of sentences per chunk for sentence strategy.',
)
@click.option(
    '--soft-boundary-embed-size',
    default=DEFAULT_SOFT_BOUNDARY_EMBED_SIZE,
    type=int,
    help='Token length of the embeddings that come before/after soft boundaries (i.e. overlapping embeddings). Above zero, soft boundaries are used.',
)
@click.option(
    '--hard-boundary-embed-size',
    default=DEFAULT_HARD_BOUNDARY_EMBED_SIZE,
    type=int,
    help='Token length of the embeddings that come before/after hard boundaries. Above zero, hard boundaries are used.',
)

def main(
    model_name,
    strategy,
    task_name,
    eval_split,
    chunking_model,
    truncate_max_length,
    chunk_size,
    n_sentences,
    soft_boundary_embed_size,
    hard_boundary_embed_size,
):
    try:
        task_cls = globals()[task_name]
    except:
        raise ValueError(f'Unknown task name: {task_name}')

    model, has_instructions = load_model(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    chunking_args = {
        'chunk_size': chunk_size,
        'n_sentences': n_sentences,
        'chunking_strategy': strategy,
        'model_has_instructions': has_instructions,
        'embedding_model_name': chunking_model if chunking_model else model_name,
    }

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    overlap_sizes = [32, 64, 128, 256, 512]
    for overlap_size in overlap_sizes:


        # Evaluate with soft boundary
        tasks = [
            task_cls(
                chunked_pooling_enabled=True,
                tokenizer=tokenizer,
                prune_size=None,
                truncate_max_length=0,
                soft_boundary_embed_size=soft_boundary_embed_size,
                soft_boundary_overlap_size=overlap_size,
                hard_boundary_embed_size=0,
                **chunking_args,
            )
        ]

        evaluation = MTEB(
            tasks=tasks,
            chunked_pooling_enabled=True,
            tokenizer=tokenizer,
            prune_size=None,
            **chunking_args,
        )
        evaluation.run(
            model,
            output_folder=f'results-soft-boundary/embed_size_{soft_boundary_embed_size}/overlap_{overlap_size}',
            eval_splits=[eval_split],
            overwrite_results=True,
            batch_size=BATCH_SIZE,
            encode_kwargs={'batch_size': BATCH_SIZE},
        )
    

    # Evaluate with hard boundary
    tasks = [
        task_cls(
            chunked_pooling_enabled=True,
            tokenizer=tokenizer,
            prune_size=None,
            truncate_max_length=0,
            soft_boundary_embed_size=0,
            hard_boundary_embed_size=hard_boundary_embed_size,
            **chunking_args,
        )
    ]

    evaluation = MTEB(
        tasks=tasks,
        chunked_pooling_enabled=True,
        tokenizer=tokenizer,
        prune_size=None,
        **chunking_args,
    )

    evaluation.run(
        model,
        output_folder=f'results-hard-boundary/embed_size_{hard_boundary_embed_size}',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )

    # Evaluate with no boundary (truncation)
    tasks = [
        task_cls(
            chunked_pooling_enabled=True,
            tokenizer=tokenizer,
            prune_size=None,
            truncate_max_length=truncate_max_length,
            soft_boundary_embed_size=0,
            hard_boundary_embed_size=0,
            **chunking_args,
        )
    ]

    evaluation = MTEB(
        tasks=tasks,
        chunked_pooling_enabled=True,
        tokenizer=tokenizer,
        prune_size=None,
        **chunking_args,
    )
    evaluation.run(
        model,
        output_folder=f'results-truncation/embed_size_{truncate_max_length}',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )

    x=1
    
if __name__ == '__main__':
    main()
