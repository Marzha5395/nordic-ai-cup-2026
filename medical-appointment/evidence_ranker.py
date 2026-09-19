import argparse
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from solver import Transcript, evidence_keywords
from utils import gold_evidence, group_questions_by_conversation, temporal_iou

ROOT = Path(__file__).resolve().parent
REPO = 'mixedbread-ai/mxbai-rerank-base-v1'
REVISION = '800f24c113213a187e65bde9db00c15a2bb12738'
MODEL_DIRECTORY = ROOT / 'models' / 'mxbai-rerank-base-v1'


def passages(transcript):
    words = transcript.words
    bounds = sorted({0, len(words)} | {a for a, _ in transcript.sentences} |
                    {b for _, b in transcript.sentences} |
                    {i + 1 for i, word in enumerate(words) if word.text.endswith((',', ';', ':'))})
    result = []
    for i, start in enumerate(bounds[:-1]):
        for end in bounds[i + 1:]:
            duration = words[end - 1].end - words[start].start
            if end - start > 55 or duration > 16:
                break
            if duration < 0.3:
                continue
            result.append((start, end, ' '.join(word.text for word in words[start:end])))
    return result


class EvidenceRanker:
    def __init__(self, directory=MODEL_DIRECTORY):
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.getenv('RERANK_THREADS', '2'))
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(directory / 'onnx' / 'model_quantized.onnx'),
                                           sess_options=options, providers=['CPUExecutionProvider'])
        self.tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=192)
        self.tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')

    def scores(self, pairs):
        scores = []
        input_names = {item.name for item in self.session.get_inputs()}
        for offset in range(0, len(pairs), 16):
            batch = self.tokenizer.encode_batch(pairs[offset:offset + 16])
            data = {
                'input_ids': np.array([item.ids for item in batch], dtype=np.int64),
                'attention_mask': np.array([item.attention_mask for item in batch], dtype=np.int64),
                'token_type_ids': np.array([item.type_ids for item in batch], dtype=np.int64),
            }
            scores.extend(self.session.run(None, {k: v for k, v in data.items() if k in input_names})[0].reshape(-1).tolist())
        return scores

    def rank(self, transcript, question):
        candidates = passages(transcript)
        query = evidence_keywords(question)
        shortlist = sorted(candidates, key=lambda item: len(query & evidence_keywords(item[2])) /
                           max(1, len(query)) - 0.002 * (item[1] - item[0]), reverse=True)[:40]
        scores = self.scores([(question, item[2]) for item in shortlist])
        return sorted(zip(scores, shortlist), reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--cache', default='transcripts/turbo_gpu')
    parser.add_argument('--output', default='benchmark_results/ranker.json')
    args = parser.parse_args()
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download(REPO, revision=REVISION, local_dir=MODEL_DIRECTORY,
                          allow_patterns=['onnx/model_quantized.onnx', 'tokenizer.json', 'config.json'])
        return
    ranker = EvidenceRanker()
    scores, results = [], []
    groups = group_questions_by_conversation()
    if args.ids:
        groups = [(name, rows) for name, rows in groups if name in {f'conversation_sample_{i}.mp3' for i in args.ids}]
    if args.limit:
        groups = groups[:args.limit]
    for name, rows in groups:
        transcript = Transcript.from_dict(json.loads((ROOT / args.cache / (Path(name).stem + '.json')).read_text()))
        for row in rows:
            gold = gold_evidence(row)
            if gold is None:
                continue
            ranking = ranker.rank(transcript, row['question'])
            top = ranking[0][1]
            span = (transcript.words[top[0]].start, transcript.words[top[1] - 1].end)
            iou = temporal_iou(gold, span)
            scores.append(iou)
            results.append(dict(filename=name, question=row['question'], gold=gold, span=span, iou=iou,
                                candidates=[dict(score=score, text=item[2], span=(transcript.words[item[0]].start,
                                            transcript.words[item[1] - 1].end)) for score, item in ranking[:5]]))
        print(f'{name}: mean positive IoU so far={np.mean(scores):.3f}', flush=True)
        path = ROOT / args.output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2))
    print(f'Evidence-only test with known positive labels: mean IoU={np.mean(scores):.3f}; n={len(scores)}')


if __name__ == '__main__':
    main()
