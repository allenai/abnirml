import ir_datasets
_logger = ir_datasets.log.easy()
_logger.logger().setLevel(20) # INFO
from . import datasets
from . import probes
from . import scorers
from . import indices
from . import eval as ev
from . import util
from pathlib import Path
import pyterrier as pt
from abnirml.java import J
J.initialize()

ProbeExperiment = ev.ProbeExperiment


Probe = probes.Probe
Scorer = scorers.Scorer
NeuralScorer = scorers.NeuralScorer
PyTerrierScorer = scorers.PyTerrierScorer
CachedScorer = scorers.CachedScorer


JflegProbe = probes.JflegProbe


trecdl19 = indices.PtIndexWrapper(ir_datasets.load('msmarco-passage/trec-dl-2019'))
trecdl19_index = trecdl19.docs_ptindex()
antiquetest = indices.PtIndexWrapper(ir_datasets.load('antique/test'))


base_path = Path.home()/'.abnirml'
if not base_path.exists():
    base_path.mkdir(parents=True, exist_ok=True)
if not (base_path/'cache'/'scorers').exists():
    (base_path/'cache'/'scorers').mkdir(parents=True, exist_ok=True)
if not (base_path/'cache'/'probes').exists():
    (base_path/'cache'/'probes').mkdir(parents=True, exist_ok=True)



PROBES = {
    'CV-DL19-Rel-Len': probes.const_var.ConstVarQrelsProbe(trecdl19, const=probes.const_var.Rel, var=probes.const_var.Len),
    'CV-DL19-Rel-Tf': probes.const_var.ConstVarQrelsProbe(trecdl19, const=probes.const_var.Rel, var=probes.const_var.Tf),
    'CV-DL19-Rel-SumTf': probes.const_var.ConstVarQrelsProbe(trecdl19, const=probes.const_var.Rel, var=probes.const_var.SumTf),
    'CV-DL19-Rel-Overlap': probes.const_var.ConstVarQrelsProbe(trecdl19, const=probes.const_var.Rel, var=probes.const_var.Overlap),
    'CV-DL19-Len-Rel': probes.const_var.ConstVarQrelsProbe(trecdl19, const=probes.const_var.Len, var=probes.const_var.Rel),
    'CV-DL19-Len-Tf': probes.const_var.ConstVarQrelsProbe(trecdl19, const=probes.const_var.Len, var=probes.const_var.Tf),
    'CV-DL19-Len-SumTf': probes.const_var.ConstVarQrelsProbe(trecdl19, const=probes.const_var.Len, var=probes.const_var.SumTf),
    'CV-DL19-Len-Overlap': probes.const_var.ConstVarQrelsProbe(trecdl19, const=probes.const_var.Len, var=probes.const_var.Overlap),
    'CV-DL19-Tf-Rel': probes.const_var.ConstVarQrelsProbe(trecdl19, const=probes.const_var.Tf, var=probes.const_var.Rel),
    'CV-DL19-Tf-Len': probes.const_var.ConstVarQrelsProbe(trecdl19, const=probes.const_var.Tf, var=probes.const_var.Len),
    'CV-DL19-Tf-SumTf': probes.const_var.ConstVarQrelsProbe(trecdl19, const=probes.const_var.Tf, var=probes.const_var.SumTf),
    'CV-DL19-Tf-Overlap': probes.const_var.ConstVarQrelsProbe(trecdl19, const=probes.const_var.Tf, var=probes.const_var.Overlap),
    'CV-DL19-SumTf-Rel': probes.const_var.ConstVarQrelsProbe(trecdl19, const=probes.const_var.SumTf, var=probes.const_var.Rel),
    'CV-DL19-SumTf-Len': probes.const_var.ConstVarQrelsProbe(trecdl19, const=probes.const_var.SumTf, var=probes.const_var.Len),
    'CV-DL19-SumTf-Tf': probes.const_var.ConstVarQrelsProbe(trecdl19, const=probes.const_var.SumTf, var=probes.const_var.Tf),
    'CV-DL19-SumTf-Overlap': probes.const_var.ConstVarQrelsProbe(trecdl19, const=probes.const_var.SumTf, var=probes.const_var.Overlap),
    'CV-DL19-Overlap-Rel': probes.const_var.ConstVarQrelsProbe(trecdl19, const=probes.const_var.Overlap, var=probes.const_var.Rel),
    'CV-DL19-Overlap-Len': probes.const_var.ConstVarQrelsProbe(trecdl19, const=probes.const_var.Overlap, var=probes.const_var.Len),
    'CV-DL19-Overlap-Tf': probes.const_var.ConstVarQrelsProbe(trecdl19, const=probes.const_var.Overlap, var=probes.const_var.Tf),
    'CV-DL19-Overlap-SumTf': probes.const_var.ConstVarQrelsProbe(trecdl19, const=probes.const_var.Overlap, var=probes.const_var.SumTf),    

    'TR-DL19-ShufWords': probes.transform.TransformProbe(trecdl19, probes.transform.ShufWords()),
    'TR-DL19-ShufSents': probes.transform.TransformProbe(trecdl19, probes.transform.ShufSents()),
    'TR-DL19-ReverseSents': probes.transform.TransformProbe(trecdl19, probes.transform.ReverseSents()),
    'TR-DL19-ReverseWords': probes.transform.TransformProbe(trecdl19, probes.transform.ReverseWords()),
    'TR-DL19-ShufWordsKeepSents': probes.transform.TransformProbe(trecdl19, probes.transform.ShufWordsKeepSents()),
    'TR-DL19-ShufWordsKeepSentsAndNPs': probes.transform.TransformProbe(trecdl19, probes.transform.ShufWordsKeepSentsAndNPs()),
    'TR-DL19-ShufWordsKeepNPs': probes.transform.TransformProbe(trecdl19, probes.transform.ShufWordsKeepNPs()),
    'TR-DL19-ShufNPSlots': probes.transform.TransformProbe(trecdl19, probes.transform.ShufNPSlots()),
    'TR-DL19-ShufPrepositions': probes.transform.TransformProbe(trecdl19, probes.transform.ShufPrepositions()),
    'TR-DL19-ReverseNPSlots': probes.transform.TransformProbe(trecdl19, probes.transform.ReverseNPSlots()),
    'TR-DL19-SwapNumNPSlots2': probes.transform.TransformProbe(trecdl19, probes.transform.SwapNumNPSlots2()),
    'TR-DL19-CaseFold': probes.transform.TransformProbe(trecdl19, probes.transform.CaseFold()),
    'TR-DL19-Lemma': probes.transform.TransformProbe(trecdl19, probes.transform.Lemmatize()),
    'TR-DL19-DelPunct': probes.transform.TransformProbe(trecdl19, probes.transform.DelPunct()),
    'TR-DL19-DelSent-start': probes.transform.TransformProbe(trecdl19, probes.transform.DelSent('start')),
    'TR-DL19-DelSent-end': probes.transform.TransformProbe(trecdl19, probes.transform.DelSent('end')),
    'TR-DL19-DelSent-rand': probes.transform.TransformProbe(trecdl19, probes.transform.DelSent('rand')),
    'TR-DL19-AddSent-start': probes.transform.TransformProbe(trecdl19, probes.transform.AddSent('start')),
    'TR-DL19-AddSent-end': probes.transform.TransformProbe(trecdl19, probes.transform.AddSent('end')),
    'TR-DL19-DocTTTTTQuery': probes.transform.TransformProbe(trecdl19, probes.transform.DocTTTTTQuery()),
    'TR-DL19-Query': probes.transform.TransformProbe(trecdl19, probes.transform.Query()),
    'TR-DL19-Typo': probes.transform.TransformProbe(trecdl19, probes.transform.Typo()),
    'TR-DL19-Typo-nostops': probes.transform.TransformProbe(trecdl19, probes.transform.Typo(no_stops=True)),
    'TR-DL19-DelStops': probes.transform.TransformProbe(trecdl19, probes.transform.RmStops()),
    'TR-DL19-DelStops-DelPunct': probes.transform.TransformProbe(trecdl19, probes.transform.Multi([probes.transform.RmStops(), probes.transform.DelPunct()])),

    'TR-DL19-nrel-ShufWords': probes.transform.TransformProbe(trecdl19, probes.transform.ShufWords(), rel_range=(0, 1)),
    'TR-DL19-nrel-ShufSents': probes.transform.TransformProbe(trecdl19, probes.transform.ShufSents(), rel_range=(0, 1)),
    'TR-DL19-nrel-ReverseSents': probes.transform.TransformProbe(trecdl19, probes.transform.ReverseSents(), rel_range=(0, 1)),
    'TR-DL19-nrel-ReverseWords': probes.transform.TransformProbe(trecdl19, probes.transform.ReverseWords(), rel_range=(0, 1)),
    'TR-DL19-nrel-ShufWordsKeepSents': probes.transform.TransformProbe(trecdl19, probes.transform.ShufWordsKeepSents(), rel_range=(0, 1)),
    'TR-DL19-nrel-ShufWordsKeepSentsAndNPs': probes.transform.TransformProbe(trecdl19, probes.transform.ShufWordsKeepSentsAndNPs(), rel_range=(0, 1)),
    'TR-DL19-nrel-ShufWordsKeepNPs': probes.transform.TransformProbe(trecdl19, probes.transform.ShufWordsKeepNPs(), rel_range=(0, 1)),
    'TR-DL19-nrel-ShufNPSlots': probes.transform.TransformProbe(trecdl19, probes.transform.ShufNPSlots(), rel_range=(0, 1)),
    'TR-DL19-nrel-ShufPrepositions': probes.transform.TransformProbe(trecdl19, probes.transform.ShufPrepositions(), rel_range=(0, 1)),
    'TR-DL19-nrel-ReverseNPSlots': probes.transform.TransformProbe(trecdl19, probes.transform.ReverseNPSlots(), rel_range=(0, 1)),
    'TR-DL19-nrel-SwapNumNPSlots2': probes.transform.TransformProbe(trecdl19, probes.transform.SwapNumNPSlots2(), rel_range=(0, 1)),
    'TR-DL19-nrel-CaseFold': probes.transform.TransformProbe(trecdl19, probes.transform.CaseFold(), rel_range=(0, 1)),
    'TR-DL19-nrel-Lemma': probes.transform.TransformProbe(trecdl19, probes.transform.Lemmatize(), rel_range=(0, 1)),
    'TR-DL19-nrel-DelPunct': probes.transform.TransformProbe(trecdl19, probes.transform.DelPunct(), rel_range=(0, 1)),
    'TR-DL19-nrel-DelSent-start': probes.transform.TransformProbe(trecdl19, probes.transform.DelSent('start'), rel_range=(0, 1)),
    'TR-DL19-nrel-DelSent-end': probes.transform.TransformProbe(trecdl19, probes.transform.DelSent('end'), rel_range=(0, 1)),
    'TR-DL19-nrel-DelSent-rand': probes.transform.TransformProbe(trecdl19, probes.transform.DelSent('rand'), rel_range=(0, 1)),
    'TR-DL19-nrel-AddSent-start': probes.transform.TransformProbe(trecdl19, probes.transform.AddSent('start'), rel_range=(0, 1)),
    'TR-DL19-nrel-AddSent-end': probes.transform.TransformProbe(trecdl19, probes.transform.AddSent('end'), rel_range=(0, 1)),
    'TR-DL19-nrel-DocTTTTTQuery': probes.transform.TransformProbe(trecdl19, probes.transform.DocTTTTTQuery(), rel_range=(0, 1)),
    'TR-DL19-nrel-Query': probes.transform.TransformProbe(trecdl19, probes.transform.Query(), rel_range=(0, 1)),
    'TR-DL19-nrel-Typo': probes.transform.TransformProbe(trecdl19, probes.transform.Typo(), rel_range=(0, 1)),
    'TR-DL19-nrel-DelStops': probes.transform.TransformProbe(trecdl19, probes.transform.RmStops(), rel_range=(0, 1)),
    'TR-DL19-nrel-DelStops-DelPunct': probes.transform.TransformProbe(trecdl19, probes.transform.Multi([probes.transform.RmStops(), probes.transform.DelPunct()]), rel_range=(0, 1)),

    'TR-DL19-rel-ShufWords': probes.transform.TransformProbe(trecdl19, probes.transform.ShufWords(), rel_range=(2, 3)),
    'TR-DL19-rel-ShufSents': probes.transform.TransformProbe(trecdl19, probes.transform.ShufSents(), rel_range=(2, 3)),
    'TR-DL19-rel-ReverseSents': probes.transform.TransformProbe(trecdl19, probes.transform.ReverseSents(), rel_range=(2, 3)),
    'TR-DL19-rel-ReverseWords': probes.transform.TransformProbe(trecdl19, probes.transform.ReverseWords(), rel_range=(2, 3)),
    'TR-DL19-rel-ShufWordsKeepSents': probes.transform.TransformProbe(trecdl19, probes.transform.ShufWordsKeepSents(), rel_range=(2, 3)),
    'TR-DL19-rel-ShufWordsKeepSentsAndNPs': probes.transform.TransformProbe(trecdl19, probes.transform.ShufWordsKeepSentsAndNPs(), rel_range=(2, 3)),
    'TR-DL19-rel-ShufWordsKeepNPs': probes.transform.TransformProbe(trecdl19, probes.transform.ShufWordsKeepNPs(), rel_range=(2, 3)),
    'TR-DL19-rel-ShufNPSlots': probes.transform.TransformProbe(trecdl19, probes.transform.ShufNPSlots(), rel_range=(2, 3)),
    'TR-DL19-rel-ShufPrepositions': probes.transform.TransformProbe(trecdl19, probes.transform.ShufPrepositions(), rel_range=(2, 3)),
    'TR-DL19-rel-ReverseNPSlots': probes.transform.TransformProbe(trecdl19, probes.transform.ReverseNPSlots(), rel_range=(2, 3)),
    'TR-DL19-rel-SwapNumNPSlots2': probes.transform.TransformProbe(trecdl19, probes.transform.SwapNumNPSlots2(), rel_range=(2, 3)),
    'TR-DL19-rel-CaseFold': probes.transform.TransformProbe(trecdl19, probes.transform.CaseFold(), rel_range=(2, 3)),
    'TR-DL19-rel-Lemma': probes.transform.TransformProbe(trecdl19, probes.transform.Lemmatize(), rel_range=(2, 3)),
    'TR-DL19-rel-DelPunct': probes.transform.TransformProbe(trecdl19, probes.transform.DelPunct(), rel_range=(2, 3)),
    'TR-DL19-rel-DelSent-start': probes.transform.TransformProbe(trecdl19, probes.transform.DelSent('start'), rel_range=(2, 3)),
    'TR-DL19-rel-DelSent-end': probes.transform.TransformProbe(trecdl19, probes.transform.DelSent('end'), rel_range=(2, 3)),
    'TR-DL19-rel-DelSent-rand': probes.transform.TransformProbe(trecdl19, probes.transform.DelSent('rand'), rel_range=(2, 3)),
    'TR-DL19-rel-AddSent-start': probes.transform.TransformProbe(trecdl19, probes.transform.AddSent('start'), rel_range=(2, 3)),
    'TR-DL19-rel-AddSent-end': probes.transform.TransformProbe(trecdl19, probes.transform.AddSent('end'), rel_range=(2, 3)),
    'TR-DL19-rel-DocTTTTTQuery': probes.transform.TransformProbe(trecdl19, probes.transform.DocTTTTTQuery(), rel_range=(2, 3)),
    'TR-DL19-rel-Query': probes.transform.TransformProbe(trecdl19, probes.transform.Query(), rel_range=(2, 3)),
    'TR-DL19-rel-Typo': probes.transform.TransformProbe(trecdl19, probes.transform.Typo(), rel_range=(2, 3)),
    'TR-DL19-rel-DelStops': probes.transform.TransformProbe(trecdl19, probes.transform.RmStops(), rel_range=(2, 3)),
    'TR-DL19-rel-DelStops-DelPunct': probes.transform.TransformProbe(trecdl19, probes.transform.Multi([probes.transform.RmStops(), probes.transform.DelPunct()]), rel_range=(2, 3)),



    'CV-ANT-Rel-Len': probes.const_var.ConstVarQrelsProbe(antiquetest, const=probes.const_var.Rel, var=probes.const_var.Len),
    'CV-ANT-Rel-Tf': probes.const_var.ConstVarQrelsProbe(antiquetest, const=probes.const_var.Rel, var=probes.const_var.Tf),
    'CV-ANT-Rel-SumTf': probes.const_var.ConstVarQrelsProbe(antiquetest, const=probes.const_var.Rel, var=probes.const_var.SumTf),
    'CV-ANT-Rel-Overlap': probes.const_var.ConstVarQrelsProbe(antiquetest, const=probes.const_var.Rel, var=probes.const_var.Overlap),
    'CV-ANT-Len-Rel': probes.const_var.ConstVarQrelsProbe(antiquetest, const=probes.const_var.Len, var=probes.const_var.Rel),
    'CV-ANT-Len-Tf': probes.const_var.ConstVarQrelsProbe(antiquetest, const=probes.const_var.Len, var=probes.const_var.Tf),
    'CV-ANT-Len-SumTf': probes.const_var.ConstVarQrelsProbe(antiquetest, const=probes.const_var.Len, var=probes.const_var.SumTf),
    'CV-ANT-Len-Overlap': probes.const_var.ConstVarQrelsProbe(antiquetest, const=probes.const_var.Len, var=probes.const_var.Overlap),
    'CV-ANT-Tf-Rel': probes.const_var.ConstVarQrelsProbe(antiquetest, const=probes.const_var.Tf, var=probes.const_var.Rel),
    'CV-ANT-Tf-Len': probes.const_var.ConstVarQrelsProbe(antiquetest, const=probes.const_var.Tf, var=probes.const_var.Len),
    'CV-ANT-Tf-SumTf': probes.const_var.ConstVarQrelsProbe(antiquetest, const=probes.const_var.Tf, var=probes.const_var.SumTf),
    'CV-ANT-Tf-Overlap': probes.const_var.ConstVarQrelsProbe(antiquetest, const=probes.const_var.Tf, var=probes.const_var.Overlap),
    'CV-ANT-SumTf-Rel': probes.const_var.ConstVarQrelsProbe(antiquetest, const=probes.const_var.SumTf, var=probes.const_var.Rel),
    'CV-ANT-SumTf-Len': probes.const_var.ConstVarQrelsProbe(antiquetest, const=probes.const_var.SumTf, var=probes.const_var.Len),
    'CV-ANT-SumTf-Tf': probes.const_var.ConstVarQrelsProbe(antiquetest, const=probes.const_var.SumTf, var=probes.const_var.Tf),
    'CV-ANT-SumTf-Overlap': probes.const_var.ConstVarQrelsProbe(antiquetest, const=probes.const_var.SumTf, var=probes.const_var.Overlap),
    'CV-ANT-Overlap-Rel': probes.const_var.ConstVarQrelsProbe(antiquetest, const=probes.const_var.Overlap, var=probes.const_var.Rel),
    'CV-ANT-Overlap-Len': probes.const_var.ConstVarQrelsProbe(antiquetest, const=probes.const_var.Overlap, var=probes.const_var.Len),
    'CV-ANT-Overlap-Tf': probes.const_var.ConstVarQrelsProbe(antiquetest, const=probes.const_var.Overlap, var=probes.const_var.Tf),
    'CV-ANT-Overlap-SumTf': probes.const_var.ConstVarQrelsProbe(antiquetest, const=probes.const_var.Overlap, var=probes.const_var.SumTf),    

    'TR-ANT-ShufWords': probes.transform.TransformProbe(antiquetest, probes.transform.ShufWords()),
    'TR-ANT-ShufSents': probes.transform.TransformProbe(antiquetest, probes.transform.ShufSents()),
    'TR-ANT-ReverseSents': probes.transform.TransformProbe(antiquetest, probes.transform.ReverseSents()),
    'TR-ANT-ReverseWords': probes.transform.TransformProbe(antiquetest, probes.transform.ReverseWords()),
    'TR-ANT-ShufWordsKeepSents': probes.transform.TransformProbe(antiquetest, probes.transform.ShufWordsKeepSents()),
    'TR-ANT-ShufWordsKeepSentsAndNPs': probes.transform.TransformProbe(antiquetest, probes.transform.ShufWordsKeepSentsAndNPs()),
    'TR-ANT-ShufWordsKeepNPs': probes.transform.TransformProbe(antiquetest, probes.transform.ShufWordsKeepNPs()),
    'TR-ANT-ShufNPSlots': probes.transform.TransformProbe(antiquetest, probes.transform.ShufNPSlots()),
    'TR-ANT-ShufPrepositions': probes.transform.TransformProbe(antiquetest, probes.transform.ShufPrepositions()),
    'TR-ANT-ReverseNPSlots': probes.transform.TransformProbe(antiquetest, probes.transform.ReverseNPSlots()),
    'TR-ANT-SwapNumNPSlots2': probes.transform.TransformProbe(antiquetest, probes.transform.SwapNumNPSlots2()),
    'TR-ANT-CaseFold': probes.transform.TransformProbe(antiquetest, probes.transform.CaseFold()),
    'TR-ANT-Lemma': probes.transform.TransformProbe(antiquetest, probes.transform.Lemmatize()),
    'TR-ANT-DelPunct': probes.transform.TransformProbe(antiquetest, probes.transform.DelPunct()),
    'TR-ANT-DelSent-start': probes.transform.TransformProbe(antiquetest, probes.transform.DelSent('start')),
    'TR-ANT-DelSent-end': probes.transform.TransformProbe(antiquetest, probes.transform.DelSent('end')),
    'TR-ANT-DelSent-rand': probes.transform.TransformProbe(antiquetest, probes.transform.DelSent('rand')),
    'TR-ANT-AddSent-start': probes.transform.TransformProbe(antiquetest, probes.transform.AddSent('start')),
    'TR-ANT-AddSent-end': probes.transform.TransformProbe(antiquetest, probes.transform.AddSent('end')),
    'TR-ANT-AddSent1-end': probes.transform.TransformProbe(antiquetest, probes.transform.AddSent('end', rel=1)),
    'TR-ANT-DocTTTTTQuery': probes.transform.TransformProbe(antiquetest, probes.transform.DocTTTTTQuery()),
    'TR-ANT-Query': probes.transform.TransformProbe(antiquetest, probes.transform.Query()),
    'TR-ANT-Typo': probes.transform.TransformProbe(antiquetest, probes.transform.Typo()),
    'TR-ANT-Typo-nostops': probes.transform.TransformProbe(antiquetest, probes.transform.Typo(no_stops=True)),
    'TR-ANT-DelStops': probes.transform.TransformProbe(antiquetest, probes.transform.RmStops()),
    'TR-ANT-DelStops-DelPunct': probes.transform.TransformProbe(antiquetest, probes.transform.Multi([probes.transform.RmStops(), probes.transform.DelPunct()])),

    'TR-ANT-nrel-ShufWords': probes.transform.TransformProbe(antiquetest, probes.transform.ShufWords(), rel_range=(0, 1)),
    'TR-ANT-nrel-ShufSents': probes.transform.TransformProbe(antiquetest, probes.transform.ShufSents(), rel_range=(0, 1)),
    'TR-ANT-nrel-ReverseSents': probes.transform.TransformProbe(antiquetest, probes.transform.ReverseSents(), rel_range=(0, 1)),
    'TR-ANT-nrel-ReverseWords': probes.transform.TransformProbe(antiquetest, probes.transform.ReverseWords(), rel_range=(0, 1)),
    'TR-ANT-nrel-ShufWordsKeepSents': probes.transform.TransformProbe(antiquetest, probes.transform.ShufWordsKeepSents(), rel_range=(0, 1)),
    'TR-ANT-nrel-ShufWordsKeepSentsAndNPs': probes.transform.TransformProbe(antiquetest, probes.transform.ShufWordsKeepSentsAndNPs(), rel_range=(0, 1)),
    'TR-ANT-nrel-ShufWordsKeepNPs': probes.transform.TransformProbe(antiquetest, probes.transform.ShufWordsKeepNPs(), rel_range=(0, 1)),
    'TR-ANT-nrel-ShufNPSlots': probes.transform.TransformProbe(antiquetest, probes.transform.ShufNPSlots(), rel_range=(0, 1)),
    'TR-ANT-nrel-ShufPrepositions': probes.transform.TransformProbe(antiquetest, probes.transform.ShufPrepositions(), rel_range=(0, 1)),
    'TR-ANT-nrel-ReverseNPSlots': probes.transform.TransformProbe(antiquetest, probes.transform.ReverseNPSlots(), rel_range=(0, 1)),
    'TR-ANT-nrel-SwapNumNPSlots2': probes.transform.TransformProbe(antiquetest, probes.transform.SwapNumNPSlots2(), rel_range=(0, 1)),
    'TR-ANT-nrel-CaseFold': probes.transform.TransformProbe(antiquetest, probes.transform.CaseFold(), rel_range=(0, 1)),
    'TR-ANT-nrel-Lemma': probes.transform.TransformProbe(antiquetest, probes.transform.Lemmatize(), rel_range=(0, 1)),
    'TR-ANT-nrel-DelPunct': probes.transform.TransformProbe(antiquetest, probes.transform.DelPunct(), rel_range=(0, 1)),
    'TR-ANT-nrel-DelSent-start': probes.transform.TransformProbe(antiquetest, probes.transform.DelSent('start'), rel_range=(0, 1)),
    'TR-ANT-nrel-DelSent-end': probes.transform.TransformProbe(antiquetest, probes.transform.DelSent('end'), rel_range=(0, 1)),
    'TR-ANT-nrel-DelSent-rand': probes.transform.TransformProbe(antiquetest, probes.transform.DelSent('rand'), rel_range=(0, 1)),
    'TR-ANT-nrel-AddSent-start': probes.transform.TransformProbe(antiquetest, probes.transform.AddSent('start'), rel_range=(0, 1)),
    'TR-ANT-nrel-AddSent-end': probes.transform.TransformProbe(antiquetest, probes.transform.AddSent('end'), rel_range=(0, 1)),
    'TR-ANT-nrel-DocTTTTTQuery': probes.transform.TransformProbe(antiquetest, probes.transform.DocTTTTTQuery(), rel_range=(0, 1)),
    'TR-ANT-nrel-Query': probes.transform.TransformProbe(antiquetest, probes.transform.Query(), rel_range=(0, 1)),
    'TR-ANT-nrel-Typo': probes.transform.TransformProbe(antiquetest, probes.transform.Typo(), rel_range=(0, 1)),
    'TR-ANT-nrel-DelStops': probes.transform.TransformProbe(antiquetest, probes.transform.RmStops(), rel_range=(0, 1)),
    'TR-ANT-nrel-DelStops-DelPunct': probes.transform.TransformProbe(antiquetest, probes.transform.Multi([probes.transform.RmStops(), probes.transform.DelPunct()]), rel_range=(0, 1)),

    'TR-ANT-rel-ShufWords': probes.transform.TransformProbe(antiquetest, probes.transform.ShufWords(), rel_range=(2, 3)),
    'TR-ANT-rel-ShufSents': probes.transform.TransformProbe(antiquetest, probes.transform.ShufSents(), rel_range=(2, 3)),
    'TR-ANT-rel-ReverseSents': probes.transform.TransformProbe(antiquetest, probes.transform.ReverseSents(), rel_range=(2, 3)),
    'TR-ANT-rel-ReverseWords': probes.transform.TransformProbe(antiquetest, probes.transform.ReverseWords(), rel_range=(2, 3)),
    'TR-ANT-rel-ShufWordsKeepSents': probes.transform.TransformProbe(antiquetest, probes.transform.ShufWordsKeepSents(), rel_range=(2, 3)),
    'TR-ANT-rel-ShufWordsKeepSentsAndNPs': probes.transform.TransformProbe(antiquetest, probes.transform.ShufWordsKeepSentsAndNPs(), rel_range=(2, 3)),
    'TR-ANT-rel-ShufWordsKeepNPs': probes.transform.TransformProbe(antiquetest, probes.transform.ShufWordsKeepNPs(), rel_range=(2, 3)),
    'TR-ANT-rel-ShufNPSlots': probes.transform.TransformProbe(antiquetest, probes.transform.ShufNPSlots(), rel_range=(2, 3)),
    'TR-ANT-rel-ShufPrepositions': probes.transform.TransformProbe(antiquetest, probes.transform.ShufPrepositions(), rel_range=(2, 3)),
    'TR-ANT-rel-ReverseNPSlots': probes.transform.TransformProbe(antiquetest, probes.transform.ReverseNPSlots(), rel_range=(2, 3)),
    'TR-ANT-rel-SwapNumNPSlots2': probes.transform.TransformProbe(antiquetest, probes.transform.SwapNumNPSlots2(), rel_range=(2, 3)),
    'TR-ANT-rel-CaseFold': probes.transform.TransformProbe(antiquetest, probes.transform.CaseFold(), rel_range=(2, 3)),
    'TR-ANT-rel-Lemma': probes.transform.TransformProbe(antiquetest, probes.transform.Lemmatize(), rel_range=(2, 3)),
    'TR-ANT-rel-DelPunct': probes.transform.TransformProbe(antiquetest, probes.transform.DelPunct(), rel_range=(2, 3)),
    'TR-ANT-rel-DelSent-start': probes.transform.TransformProbe(antiquetest, probes.transform.DelSent('start'), rel_range=(2, 3)),
    'TR-ANT-rel-DelSent-end': probes.transform.TransformProbe(antiquetest, probes.transform.DelSent('end'), rel_range=(2, 3)),
    'TR-ANT-rel-DelSent-rand': probes.transform.TransformProbe(antiquetest, probes.transform.DelSent('rand'), rel_range=(2, 3)),
    'TR-ANT-rel-AddSent-start': probes.transform.TransformProbe(antiquetest, probes.transform.AddSent('start'), rel_range=(2, 3)),
    'TR-ANT-rel-AddSent-end': probes.transform.TransformProbe(antiquetest, probes.transform.AddSent('end'), rel_range=(2, 3)),
    'TR-ANT-rel-DocTTTTTQuery': probes.transform.TransformProbe(antiquetest, probes.transform.DocTTTTTQuery(), rel_range=(2, 3)),
    'TR-ANT-rel-Query': probes.transform.TransformProbe(antiquetest, probes.transform.Query(), rel_range=(2, 3)),
    'TR-ANT-rel-Typo': probes.transform.TransformProbe(antiquetest, probes.transform.Typo(), rel_range=(2, 3)),
    'TR-ANT-rel-DelStops': probes.transform.TransformProbe(antiquetest, probes.transform.RmStops(), rel_range=(2, 3)),
    'TR-ANT-rel-DelStops-DelPunct': probes.transform.TransformProbe(antiquetest, probes.transform.Multi([probes.transform.RmStops(), probes.transform.DelPunct()]), rel_range=(2, 3)),


    'DS-Jfleg': probes.JflegProbe(),
    'DS-Jfleg-sp': probes.JflegProbe('abnirml:jfleg/sp'),
    'DS-Gyafc': probes.GyafcProbe(),
    'DS-Gyafc-family': probes.GyafcProbe(genre_filter='Family_Relationships'),
    'DS-Gyafc-enter': probes.GyafcProbe(genre_filter='Entertainment_Music'),
    'DS-CnnDmProbe-cnn': probes.CnnDmProbe(source='cnn'),
    'DS-CnnDmProbe-dm': probes.CnnDmProbe(source='dm'),
    'DS-XSum': probes.XSumProbe(),
    'DS-Bias': probes.BiasProbe(),
    'DS-Paraphrase-mspc': probes.ParaphraseProbe('abnirml:mspc'),
    'DS-Factuality-nq-tertok-train-PERSON': probes.FactualityProbe('dpr-w100/natural-questions/train', valid_entities=('PERSON'), tokenizer=trecdl19_index),
    'DS-Factuality-nq-tertok-train-GPE': probes.FactualityProbe('dpr-w100/natural-questions/train', valid_entities=('GPE'), tokenizer=trecdl19_index),
    'DS-Factuality-nq-tertok-train-LOC': probes.FactualityProbe('dpr-w100/natural-questions/train', valid_entities=('LOC'), tokenizer=trecdl19_index),
    'DS-Factuality-nq-tertok-train-NORP': probes.FactualityProbe('dpr-w100/natural-questions/train', valid_entities=('NORP'), tokenizer=trecdl19_index),
    'DS-Factuality-nq-tertok-train-ORG': probes.FactualityProbe('dpr-w100/natural-questions/train', valid_entities=('ORG'), tokenizer=trecdl19_index),
    'DS-Simplification-wikiturk': probes.SimplificationProbe('abnirml:wikiturk'),
}

PROBES = {k: probes.CachedProbe(v, base_path/'cache'/'probes'/f'{k}.cache') for k, v in PROBES.items()}

def _monot5():
    import pyterrier_t5
    return pyterrier_t5.MonoT5ReRanker(batch_size=64, verbose=False)

def _monot5_large():
    import pyterrier_t5
    return pyterrier_t5.MonoT5ReRanker(model='castorini/monot5-large-msmarco', batch_size=32, verbose=False)

def _colbert():
    from pyterrier_colbert.ranking import ColBERTFactory
    pt_colbert = ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip", None, None)
    return pt_colbert.text_scorer()

def _wmd():
    from pyterrier_gensim import WmdScorer
    import logging
    logging.getLogger('gensim').level = 999
    return WmdScorer("glove-wiki-gigaword-100", verbose=False)

def _ance():
    import pyterrier_ance
    return pyterrier_ance.ANCEReRanker('ance_checkpoint')

def _sbert_bi(model_name):
    def wrapped():
        import pyterrier_sbert
        return pyterrier_sbert.SbertBiScorer(model_name)
    return wrapped

def _onir(f, gpu=True):
    def wrapped():
        import onir_pt
        return onir_pt.reranker.from_checkpoint(f, config={'verbose': False, 'batch_size': 64, 'gpu': gpu})
    return wrapped

# NOTE: below the delta values are assigned using the median values from abnirml.eval.topk_diffs
SCORERS = {
    'BM25-dl19': scorers.TerrierBM25(index=trecdl19.docs_ptindex()),
    'T5': scorers.PyTerrierScorer('T5-pt', _monot5, batch_size=64),
    'T5-l': scorers.PyTerrierScorer('T5-l-pt', _monot5_large, batch_size=32),
    'VBERT': scorers.VanillaBERT(weight_path="/home/sean/data/abnirml/vbert.p"),
    'EPIC': scorers.PyTerrierScorer('epic42', _onir('/home/sean/data/onir/model_checkpoints/epic.42.tar.gz'), batch_size=16),
    'ColBERT': scorers.PyTerrierScorer('ColBERT-pt', _colbert),
    'ANCE': scorers.PyTerrierScorer('ANCE-pt', _ance),
    'WMD': scorers.PyTerrierScorer('WMD-pt', _wmd),
    'SBERT': scorers.PyTerrierScorer('SBERT-bi-pt', _sbert_bi("sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking")),
    'KNRM': scorers.PyTerrierScorer('knrm', _onir('/home/sean/data/onir/model_checkpoints/knrm.msmarco.new.tar.gz', gpu=False), batch_size=64),
    'ConvKNRM': scorers.PyTerrierScorer('convknrm', _onir('/home/sean/data/onir/model_checkpoints/convknrm.msmarco.tar.gz', gpu=False), batch_size=64),
    'S2': scorers.S2(),
    'DocTTTTTQuery-BM25-dl19': scorers.DocTTTTTQuery(scorers.TerrierBM25(index=trecdl19.docs_ptindex())),
}

SCORERS = {k: scorers.CachedScorer(v, base_path/'cache'/'scorers'/f'{k}.cache') for k, v in SCORERS.items()}

# Tuned on top TREC DL'19
DELTAS = {
    'BM25-dl19': 0.35495386740345936,
    'T5': 0.0038296878337860107,
    'T5-l': 0.004211690276861191,
    'VBERT': 0.04632759094238281,
    'EPIC': 0.2732887268066406,
    'ColBERT': 0.15594482421875,
    'ANCE': 0.109130859375,
    'WMD': 0.00643495,
    'SBERT': 0.0011723,
    'KNRM': 0.15517234802246094,
    'ConvKNRM': 0.3034172058105469,
    'S2': 0.0,
    'DocTTTTTQuery-BM25-dl19': 0.3584156381449404,
}
