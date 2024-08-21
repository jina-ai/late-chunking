import datasets
from mteb.abstasks.TaskMetadata import TaskMetadata

from chunked_pooling.mteb_chunked_eval import AbsTaskChunkedRetrieval


class SciFactChunked(AbsTaskChunkedRetrieval):
    metadata = TaskMetadata(
        name='SciFactChunked',
        dataset={
            'path': 'mteb/scifact',
            'revision': '0228b52cf27578f30900b9e5271d331663a030d7',
            'name': 'SciFact',
        },
        description=(
            'SciFact verifies scientific claims using evidence from the '
            'research literature containing scientific paper abstracts.'
        ),
        reference='https://github.com/allenai/scifact',
        type='Retrieval',
        category='s2p',
        eval_splits=['test'],
        eval_langs=['eng-Latn'],
        main_score='ndcg_at_10',
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class NarrativeQAChunked(AbsTaskChunkedRetrieval):
    metadata = TaskMetadata(
        name='NarrativeQAChunked',
        dataset={
            'path': 'narrativeqa',
            'revision': '2e643e7363944af1c33a652d1c87320d0871c4e4',
            'name': 'NarrativeQARetrieval',
        },
        reference='https://metatext.io/datasets/narrativeqa',
        description=(
            'NarrativeQA is a dataset for the task of question answering '
            'on long narratives. It consists of realistic QA instances '
            'collected from literature (fiction and non-fiction) '
            'and movie scripts. '
        ),
        type='Retrieval',
        category='s2p',
        eval_splits=['test'],
        eval_langs=['eng-Latn'],
        main_score='ndcg_at_10',
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class NFCorpusChunked(AbsTaskChunkedRetrieval):
    metadata = TaskMetadata(
        name="NFCorpusChunked",
        dataset={
            "path": "mteb/nfcorpus",
            "revision": "ec0fa4fe99da2ff19ca1214b7966684033a58814",
            'name': 'NFCorpus',
        },
        description="NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval",
        reference="https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/",
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class QuoraChunked(AbsTaskChunkedRetrieval):
    metadata = TaskMetadata(
        name="QuoraChunked",
        dataset={
            "path": "mteb/quora",
            "revision": "e4e08e0b7dbe3c8700f0daef558ff32256715259",
            "name": "QuoraRetrieval",
        },
        description=(
            "QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a"
            " question, find other (duplicate) questions."
        ),
        reference="https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs",
        type="Retrieval",
        category="s2s",
        eval_splits=["dev", "test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FiQA2018Chunked(AbsTaskChunkedRetrieval):
    metadata = TaskMetadata(
        name="FiQA2018Chunked",
        description="Financial Opinion Mining and Question Answering",
        reference="https://sites.google.com/view/fiqa/",
        dataset={
            "path": "mteb/fiqa",
            "revision": "27a168819829fe9bcd655c2df245fb19452e8e06",
            'name': 'FiQA2018',
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["train", "dev", "test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TRECCOVIDChunked(AbsTaskChunkedRetrieval):
    metadata = TaskMetadata(
        name='TRECCOVIDChunked',
        description=(
            'TRECCOVID is an ad-hoc search challenge based on the '
            'COVID-19 dataset containing scientific articles '
            'related to the COVID-19 pandemic.'
        ),
        reference='https://ir.nist.gov/covidSubmit/index.html',
        dataset={
            'path': 'mteb/trec-covid',
            'revision': 'bb9466bac8153a0349341eb1b22e06409e78ef4e',
            'name': 'TRECCOVID',
        },
        type='Retrieval',
        category='s2p',
        eval_splits=['test'],
        eval_langs=['eng-Latn'],
        main_score='ndcg_at_10',
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class GerDaLIRChunked(AbsTaskChunkedRetrieval):
    _EVAL_SPLIT = 'test'

    metadata = TaskMetadata(
        name='GerDaLIRChunked',
        description=(
            'GerDaLIR is a legal information retrieval dataset '
            'created from the Open Legal Data platform.'
        ),
        reference='https://github.com/lavis-nlp/GerDaLIR',
        dataset={
            'path': 'jinaai/ger_da_lir',
            'revision': '0bb47f1d73827e96964edb84dfe552f62f4fd5eb',
            'name': 'GerDaLIR',
        },
        type='Retrieval',
        category='s2p',
        eval_splits=[_EVAL_SPLIT],
        eval_langs=['deu-Latn'],
        main_score='ndcg_at_10',
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        dataset_metadata = self.metadata_dict['dataset']
        dataset_metadata.pop('name')

        query_rows = datasets.load_dataset(
            name='queries',
            split=self._EVAL_SPLIT,
            **dataset_metadata,
        )
        corpus_rows = datasets.load_dataset(
            name='corpus',
            split=self._EVAL_SPLIT,
            **dataset_metadata,
        )
        qrels_rows = datasets.load_dataset(
            name='qrels',
            split=self._EVAL_SPLIT,
            **dataset_metadata,
        )

        self.queries = {
            self._EVAL_SPLIT: {row['_id']: row['text'] for row in query_rows}
        }
        self.corpus = {self._EVAL_SPLIT: {row['_id']: row for row in corpus_rows}}
        self.relevant_docs = {
            self._EVAL_SPLIT: {
                row['_id']: {v: 1 for v in row['text'].split(' ')} for row in qrels_rows
            }
        }

        self.data_loaded = True
