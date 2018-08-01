from operator import itemgetter

from tqdm import tqdm

from util.corpus_util import get_corpus
from util.vad_util import extract_voice

result = []
corpora = [get_corpus('rl'), get_corpus('ls')]

for corpus in corpora:
    for entry in tqdm(corpus):
        voice_segments = extract_voice(entry.audio, entry.rate)
        result.append((entry.id, entry.name, len(voice_segments)))

    ms = sorted(result, key=itemgetter(2), reverse=True)
    stats = [f'id={m[0]}, name={m[1]}, nos={m[2]}' for m in ms[:100]]
    with open(corpus.name + '_speech_segments.txt', 'w', encoding='utf-8') as f:
        f.writelines(stats)
