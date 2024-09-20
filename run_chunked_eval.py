import click
import torch.cuda
from mteb import MTEB
from transformers import AutoModel, AutoTokenizer

from chunked_pooling.chunked_eval_tasks import *
from chunked_pooling.wrappers import load_model

DEFAULT_CHUNKING_STRATEGY = 'fixed'
DEFAULT_CHUNK_SIZE = 256
DEFAULT_N_SENTENCES = 5
BATCH_SIZE = 1


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
    '--task-name', default='SciFactChunked', help='The evaluation task to perform.'
)
@click.option(
    '--eval-split', default='test', help='The name of the evaluation split in the task.'
)
def main(model_name, strategy, task_name, eval_split):
    try:
        task_cls = globals()[task_name]
    except:
        raise ValueError(f'Unknown task name: {task_name}')

    model, has_instructions = load_model(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    chunking_args = {
        'chunk_size': DEFAULT_CHUNK_SIZE,
        'n_sentences': DEFAULT_N_SENTENCES,
        'chunking_strategy': strategy,
        'model_has_instructions': has_instructions,
    }

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    # Evaluate with late chunking
    tasks = [
        task_cls(
            chunked_pooling_enabled=True,
            tokenizer=tokenizer,
            prune_size=None,
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
        output_folder='results-chunked-pooling',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )

    # Encode without late chunking
    tasks = [
        task_cls(
            chunked_pooling_enabled=False,
            tokenizer=tokenizer,
            prune_size=None,
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
        output_folder='results-normal-pooling',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
    )


if __name__ == '__main__':
    main()
