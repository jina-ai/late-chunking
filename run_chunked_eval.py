import click
import torch.cuda
from mteb import MTEB
from transformers import AutoModel, AutoTokenizer

from chunked_pooling.chunked_eval_tasks import (FiQA2018Chunked,
                                                LEMBWikimQARetrievalChunked,
                                                NFCorpusChunked, QuoraChunked,
                                                SciFactChunked,
                                                TRECCOVIDChunked)

DEFAULT_CHUNKING_STRATEGY = 'fixed'
DEFAULT_CHUNK_SIZE = 256
DEFAULT_N_SENTENCES = 5


def remove_prompt_name(original_encode):
    def wrapper(self, *args, **kwargs):
        # Remove 'prompt_name' from kwargs if present
        kwargs.pop('prompt_name', None)
        return original_encode(self, *args, **kwargs)

    return wrapper


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
    '--task-name', default='SciFactChunked', help='The evaluationtask to perform.'
)
def main(model_name, strategy, task_name):
    try:
        task_cls = globals()[task_name]
    except:
        raise ValueError(f'Unknown task name: {task_name}')

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    if model_name == 'jinaai/jina-embeddings-v2-small-en':
        print("Overwriting encode")
        model.encode = remove_prompt_name(model.encode)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    chunking_args = {
        'chunk_size': DEFAULT_CHUNK_SIZE,
        'n_sentences': DEFAULT_N_SENTENCES,
        'chunking_strategy': strategy,
    }

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    # Evaluate with chunking
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
        eval_splits=['test'],
        overwrite_results=True,
        encode_kwargs={'batch_size': 1},
    )

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
        eval_splits=['test'],
        overwrite_results=True,
        encode_kwargs={'batch_size': 1},
    )


if __name__ == '__main__':
    main()
