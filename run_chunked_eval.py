import sys
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

DEFAULT_CHUNKING_STRATEGY = 'fixed'
DEFAULT_CHUNK_SIZE = 256
DEFAULT_N_SENTENCES = 5


def main(task_name):
    try:
        task_cls = globals()[task_name]
    except:
        raise ValueError(f'Unknown task name: {task_name}')

    model = AutoModel.from_pretrained(
        'jinaai/jina-embeddings-v2-small-en', trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        'jinaai/jina-embeddings-v2-small-en', trust_remote_code=True
    )

    chunking_args = {
        'chunk_size': DEFAULT_CHUNK_SIZE,
        'n_sentences': DEFAULT_N_SENTENCES,
        'chunking_strategy': DEFAULT_CHUNKING_STRATEGY,
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
    task_name = sys.argv[1] if len(sys.argv) > 1 else 'SciFactChunked'
    main(task_name)
