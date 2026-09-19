import argparse
import hashlib
import io
import json
import logging
import os
import time
from pathlib import Path

from llm import LocalLanguageModel, SYSTEM_PROMPT
from local_evaluator import Statistics
from solver import Transcript, parse_evidence, response_from_evidence
from speech import SpeechRecognizer, decode_audio, repair_word_times
from utils import gold_evidence, group_questions_by_conversation, load_sample_audio, validate_response

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Offline tests on supplied training data only; never submits an attempt.')
    parser.add_argument('--ids', nargs='*', help='Sample numbers, e.g. 4 5 6')
    parser.add_argument('--split', choices=['all', 'dev', 'holdout'], default='dev')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--transcribe-only', action='store_true')
    parser.add_argument('--reuse-transcripts', action='store_true')
    parser.add_argument('--repair-times', action='store_true')
    parser.add_argument('--cache', default='transcripts/small.en')
    parser.add_argument('--output', default='benchmark_results/current.json')
    parser.add_argument('--llm-budget', type=float, default=600)
    parser.add_argument('--predictions', help='Re-score saved raw model outputs without running the LLM')
    args = parser.parse_args()
    groups = group_questions_by_conversation()
    if args.ids:
        names = {f'conversation_sample_{i}.mp3' for i in args.ids}
        groups = [(name, rows) for name, rows in groups if name in names]
    elif args.split == 'dev':
        groups = groups[:-10]
    elif args.split == 'holdout':
        groups = groups[-10:]
    groups = groups[args.offset:]
    if args.limit:
        groups = groups[:args.limit]
    cache = ROOT / args.cache
    cache.mkdir(parents=True, exist_ok=True)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    asr = None
    transcripts = {}
    for filename, _ in groups:
        path = cache / (Path(filename).stem + '.json')
        if args.reuse_transcripts and path.exists():
            transcript = Transcript.from_dict(json.loads(path.read_text()))
            elapsed = None
        else:
            if asr is None:
                asr = SpeechRecognizer()
            started = time.monotonic()
            transcript = asr.transcribe(load_sample_audio(filename))
            elapsed = time.monotonic() - started
            path.write_text(json.dumps(transcript.to_dict(), indent=2))
        if args.repair_times:
            transcript.words = repair_word_times(transcript.words, decode_audio(io.BytesIO(load_sample_audio(filename))))
        transcripts[filename] = (transcript, elapsed)
        print(f'Transcribed {filename}: {len(transcript.words)} words; seconds={elapsed}', flush=True)
    del asr
    if args.transcribe_only:
        return
    saved = {row['filename']: row['raw'] for row in json.loads(Path(args.predictions).read_text())} if args.predictions else {}
    model = None if args.predictions else LocalLanguageModel()
    output.with_suffix('.metadata.json').write_text(json.dumps({
        'arguments': vars(args), 'system_prompt': SYSTEM_PROMPT,
        'prompt_sha256': hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        'environment': {key: os.getenv(key) for key in ('ASR_MODEL', 'ASR_DEVICE', 'ASR_BATCH_SIZE',
                                                       'LLM_MODEL', 'LLM_GPU_LAYERS')},
    }, indent=2))
    statistics = Statistics()
    results = []
    try:
        for filename, rows in groups:
            transcript, asr_seconds = transcripts[filename]
            questions = [row['question'] for row in rows]
            started = time.monotonic()
            raw = saved[filename] if args.predictions else model.complete(transcript, questions, started + args.llm_budget)
            llm_seconds = time.monotonic() - started
            evidence = parse_evidence(raw, len(questions))
            response = response_from_evidence(transcript, questions, evidence)
            validate_response(response, len(questions))
            statistics.record_request(len(questions), None, failed=False)
            details = []
            for i, row in enumerate(rows):
                span = None if response.evidence_start[i] is None else (
                    response.evidence_start[i], response.evidence_end[i])
                iou = statistics.record(row['question_type'], int(row['label']),
                                        int(response.answers[i]), gold_evidence(row), span)
                gold = gold_evidence(row)
                gold_text = ' '.join(w.text for w in transcript.words
                                     if gold and w.end > gold[0] + 0.02 and w.start < gold[1] - 0.02)
                details.append(dict(question=row['question'], label=int(row['label']),
                                    answer=response.answers[i], gold=gold, span=span, iou=iou,
                                    quote=evidence[i], gold_text=gold_text))
            results.append(dict(filename=filename, asr_seconds=asr_seconds, llm_seconds=llm_seconds,
                                raw=raw, details=details))
            output.write_text(json.dumps(results, indent=2))
            print(f'{filename}: correct={sum(d["answer"] == bool(d["label"]) for d in details)}/{len(rows)} '
                  f'tIoU={sum(d["iou"] for d in details) / max(1, sum(d["label"] for d in details)):.3f} '
                  f'LLM={llm_seconds:.1f}s; cumulative score={statistics.final_score:.3f}', flush=True)
    finally:
        if model:
            model.close()
    print(statistics.report())
    print('Offline training-data accuracy only. CPU timings do not establish GPU deadline compliance.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
