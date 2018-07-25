from operator import itemgetter

from tqdm import tqdm

from constants import LS_CORPUS_ROOT, RL_CORPUS_ROOT
from util.corpus_util import load_corpus
from util.vad_util import extract_voice

result = []
ls_corpus = load_corpus(LS_CORPUS_ROOT)
rl_corpus = load_corpus(RL_CORPUS_ROOT)
corpora = [rl_corpus, ls_corpus]

for corpus in corpora:
    for entry in tqdm(corpus):
        voice_segments = extract_voice(entry.audio, entry.rate)
        result.append((entry.id, entry.name, len(voice_segments)))

    ms = sorted(result, key=itemgetter(2), reverse=True)
    stats = [f'id={m[0]}, name={m[1]}, nos={m[2]}' for m in ms[:100]]
    with open(corpus.name + '_speech_segments.txt', 'w', encoding='utf-8') as f:
        f.writelines(stats)
