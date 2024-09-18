import click
import torch.cuda

from transformers import AutoModel, AutoTokenizer
from mteb import MTEB

from chunked_pooling.chunked_eval_tasks import (
    SciFactChunked,
    TRECCOVIDChunked,
    FiQA2018Chunked,
    NFCorpusChunked,
    QuoraChunked,
    LEMBWikimQARetrievalChunked,
)

from chunked_pooling.wrappers import MODEL_WRAPPERS

DEFAULT_CHUNKING_STRATEGY = 'fixed'
DEFAULT_CHUNK_SIZE = 256
DEFAULT_N_SENTENCES = 5


def load_model(model_name):
    if model_name in MODEL_WRAPPERS:
        return (
            MODEL_WRAPPERS[model_name](model_name),
            MODEL_WRAPPERS[model_name].has_instructions(),
        )
    else:
        return AutoModel.from_pretrained(model_name, trust_remote_code=True), False


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
        batch_size=1,
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
        batch_size=1,
    )


if __name__ == '__main__':
    main()
