import click
import torch.cuda
from mteb import MTEB
from transformers import AutoModel, AutoTokenizer

from chunked_pooling.chunked_eval_tasks import *
from chunked_pooling.wrappers import load_model

DEFAULT_CHUNKING_STRATEGY = 'fixed'
DEFAULT_N_SENTENCES = 5
BATCH_SIZE = 1
DEFAULT_OVERLAP_SIZE = 256
DEFAULT_SOFT_BOUNDARY_EMBED_SIZE = 8192 # set to 0 to disable soft boundaries
DEFAULT_HARD_BOUNDARY_EMBED_SIZE = 0    # set to 0 to disable hard boundaries


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
    default=None,
    type=int,
    help='Maximum number of tokens; By default, no truncation is done.',
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
@click.option(
    '--overlap-size',
    default=DEFAULT_OVERLAP_SIZE,
    type=int,
    help='Number of tokens per chunk for fixed strategy.',
)

def main(
    model_name,
    strategy,
    task_name,
    eval_split,
    chunking_model,
    truncate_max_length,
    n_sentences,
    soft_boundary_embed_size,
    hard_boundary_embed_size,
    overlap_size,
):
    try:
        task_cls = globals()[task_name]
    except:
        raise ValueError(f'Unknown task name: {task_name}')

    model, has_instructions = load_model(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    chunk_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
    for chunk_size in chunk_sizes:

        print(f'\n\n\n\nEvaluating chunk size: {chunk_size} ({chunk_sizes.index(chunk_size) + 1}/{len(chunk_sizes)}) \n\n\n\n')

        chunking_args = {
            'chunk_size': chunk_size,
            'n_sentences': n_sentences,
            'chunking_strategy': strategy,
            'model_has_instructions': has_instructions,
            'embedding_model_name': chunking_model if chunking_model else model_name,
        }

        #  == Late Chunking ==
        # tasks = [
        #     task_cls(
        #         chunked_pooling_enabled=True,
        #         tokenizer=tokenizer,
        #         prune_size=None,
        #         truncate_max_length=truncate_max_length,
        #         soft_boundary_embed_size=soft_boundary_embed_size,
        #         soft_boundary_overlap_size=overlap_size,
        #         hard_boundary_embed_size=hard_boundary_embed_size,
        #         **chunking_args,
        #     )
        # ]

        # evaluation = MTEB(
        #     tasks=tasks,
        #     chunked_pooling_enabled=True,
        #     tokenizer=tokenizer,
        #     prune_size=None,
        #     **chunking_args,
        # )
        # evaluation.run(
        #     model,
        #     output_folder=f'results-chunked-pooling/chunk_size_{chunk_size}',
        #     eval_splits=[eval_split],
        #     overwrite_results=True,
        #     batch_size=BATCH_SIZE,
        #     encode_kwargs={'batch_size': BATCH_SIZE},
        # )

        #  == Naive Chunking ==
        # naive chunking does not need soft boundaries because chunk size is guaranteed to be <8192 tokens
        tasks = [
            task_cls(
                chunked_pooling_enabled=False,
                tokenizer=tokenizer,
                prune_size=None,
                truncate_max_length=truncate_max_length,
                **chunking_args,
            )
        ]

        evaluation = MTEB(
            tasks=tasks,
            chunked_pooling_enabled=False,
            tokenizer=tokenizer,
            prune_size=None,
            **chunking_args,
        )
        evaluation.run(
            model,
            output_folder=f'results-normal-pooling/chunk_size_{chunk_size}',
            eval_splits=[eval_split],
            overwrite_results=True,
            batch_size=BATCH_SIZE,
            encode_kwargs={'batch_size': BATCH_SIZE},
        )


if __name__ == '__main__':
    main()
