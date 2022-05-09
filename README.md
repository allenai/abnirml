# ABNIRML

Code for building and running neural IR model probing experiments.

For details, see our TACL paper: Sean MacAvaney, Sergey Feldman, Nazli Goharian, Doug Downey, and Arman Cohan. _ABNIRML: Analyzing the Behavior of Neural IR Models_ ([PDF](https://arxiv.org/pdf/2011.00696.pdf)).

## Installing

```bash
git clone https://github.com/allenai/abnirml.git
cd abnirml
pip install -r requirements.txt
```

## Reproducing ABNIRML Results

The following code runs all defined probes for all defined models.

Datasets that are downloadable will be automatically downloaded. Datasets that
require a manual process will give instructions. The first time you import the library,
hang tight as it downloads the MS MARCO pasage corpus and builds an index.

Note that this process will take a long time in total. It caches what it can, but there's
just a lot to run.

```bash
python -m abnirml
```

You can also run for specific scorers and probes using:

```bash
python -m abnirml --scorer SCORER_NAME --probe PROBE_NAME
```

Available models are: 
 - `BM25-dl19` -- BM25 using default parameters over the MSMARCO corpus
 - `DocTTTTTQuery-BM25-dl19` -- BM25 expanded using docT5Query
 - `T5` -- monoT5 (base)
 - `T5-l` -- monoT5 (large)
 - `VBERT` -- Vanilla BERT
 - `EPIC` -- EPIC
 - `ColBERT` -- ColBERT
 - `ANCE` -- ANCE
 - `WMD` -- Word Mover's Distance
 - `SBERT` -- SBERT (transfer)
 - `KNRM` -- KNRM
 - `ConvKNRM` -- ConvKNRM
 - `S2` -- Semantic Scholar's LtR ranker (lexical component)

Available probes are:
 - `CV-[DL19,ANT]-Rel-[Len,Tf,SumTf,Overlap]` -- Measure-and-Match Probe (MMP), for TREC DL19 or ANTique, holding relevance constant
 - `CV-[DL19,ANT]-Len-[Rel,Tf,SumTf,Overlap]` -- Measure-and-Match Probe (MMP), for TREC DL19 or ANTique, holding document length constant
 - `CV-[DL19,ANT]-Tf-[Rel,Len,SumTf,Overlap]` -- Measure-and-Match Probe (MMP), for TREC DL19 or ANTique, holding term frequence constant
 - `CV-[DL19,ANT]-SumTf-[Rel,Len,Tf,Overlap]` -- Measure-and-Match Probe (MMP), for TREC DL19 or ANTique, holding the sum of term frequencies constant
 - `CV-[DL19,ANT]-Overlap-[Rel,Len,Tf,SumTf]` -- Measure-and-Match Probe (MMP), for TREC DL19 or ANTique, holding overlap constant
 - `TR-[ANT,DL19][,-rel,-nrel]-[ShufWords,ShufSents,ReverseSents,ReverseWords,ShufWordsKeepSents,ShufWordsKeepSentsAndNPs,ShufWordsKeepNPs,ShufNPSlots,ShufPrepositions,ReverseNPSlots,SwapNumNPSlots2,CaseFold,Lemma,DelPunct,DelSent-start,DelSent-end,DelSent-rand,AddSent-start,AddSent-end,DocTTTTTQuery,Query,Typo,DelStops,DelStops-DelPunct]` --  Textual Manipulation Probe (TMP), for TREC DL19 or ANTique, over all judged documents, only relevant documents, or only non-relevant documents
 - `DS-Jfleg` -- Dataset Transfer Probe (DTP), for JFLEG Fluency
 - `DS-Jfleg-sp` -- Dataset Transfer Probe (DTP), for JFLEG Fluency (corrected for spelling)
 - `DS-Gyafc` -- Dataset Transfer Probe (DTP), for GYAFC Formality
 - `DS-Gyafc-[family,enter]` -- Dataset Transfer Probe (DTP), for GYAFC Formality (family or entertainment only)
 - `DS-CnnDmProbe-[cnn,dm]` -- Dataset Transfer Probe (DTP), for CNN/Daily Mail Succinctness
 - `DS-XSum` -- Dataset Transfer Probe (DTP), for XSum Succinctness
 - `DS-Paraphrase-mspc` -- Dataset Transfer Probe (DTP), for MSPC Paraphrases
 - `DS-Factuality-nq-tertok-train-[PERSON,GPE,LOC,NORP,ORG]` -- Dataset Transfer Probe (DTP), for NQ Factuality (over each entity type)
 - `DS-Simplification-wikiturk` -- Dataset Transfer Probe (DTP), for WikiTurk Simplification

Some models require a separate environment due to conflicting dependency versions
(e.g., huggingface transformers).

To use ColBERT, run:

```bash
# for colbert
pip install --upgrade git+https://github.com/terrierteam/pyterrier_colbert.git
```

To use ANCE, run:

```bash
# for ance
pip install --upgrade git+https://github.com/seanmacavaney/pyterrier_ance.git
conda install -c pytorch faiss-cpu
```

## Python API

`ProbeExperiment` provides the core evaluation functionality. Its arguments are:
 - `scorer`: an object that provides a score for query-document pairs. Can be either an instance of
   `abnirml.Scorer` or a [PyTerrier re-ranking-like transformer](https://pyterrier.readthedocs.io/en/latest/transformer.html).
 - `probe`: an object that provides the probe samples. Should be an instance of `abnirml.Probe`.
 - (optional) `delta`: minimum score to treat difference as non-neural. Default: `0`. See Section 3.3 for recommended delta settings.

Example: JFLEG probing of monoT5.

```python
from abnirml import ProbeExperiment, JflegProbe
from pyterrier_t5 import MonoT5ReRanker

scorer = MonoT5ReRanker(verbose=False)
probe = JflegProbe()

ProbeExperiment(scorer, probe, delta=0.0038296878337860107)  # delta based on median difference of top TREC DL'19 results.
# Results:
{
	'pos': 1834,
	'neg': 2362,
	'neu': 877,
	'score': -0.10408042578356003,
	'count': 5073,
	'mean_diff': -0.22396814961192174,
	'median_diff': 0.0,
	'p_val': 3.2449154236357972e-34,
	'hash': 'f67b10889511af74699468d56f9309b2'
}
{
	'pos': 1898, # count of positive score differences
	'neg': 2426, # count of negative score differences
	'neu': 749, # count of neural score differences (within delta)
	'count': 5073, # total number of samples (pos+neg+neu)
	'score': -0.10408042578356003, # normalised score ((pos-neg)/count)
	'mean_diff': -0.22396814961192174, # mean score difference
	'median_diff': 0.0, # median score difference
	'p_val': 3.2449154236357972e-34, # p-value of a two-sided paired t-test (note: when running multiple tests, you should perform a correction like Bonferroni)
	'hash': 'f67b10889511af74699468d56f9309b2' # md5 hash of the samples from the probe (useful to verify that probes contain the exact same data across systems)
}
```
