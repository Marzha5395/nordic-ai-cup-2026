import difflib
import json
import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from functools import lru_cache

from dtos import ASRQuestionResponseDto

logger = logging.getLogger(__name__)


@dataclass
class Word:
    start: float
    end: float
    text: str


class Transcript:
    def __init__(self, words):
        self.words = [word for word in words if word.text.strip() and
                      math.isfinite(word.start) and math.isfinite(word.end) and
                      0 <= word.start <= word.end]
        self.sentences = []
        start = 0
        for i, word in enumerate(self.words):
            if (re.search(r'[.!?][\"\u201d\u2019]*$', word.text) or
                    i == len(self.words) - 1 or
                    self.words[i + 1].start - word.end > 1.2):
                self.sentences.append((start, i + 1))
                start = i + 1
        self.tokens = []
        self.token_words = []
        for i, word in enumerate(self.words):
            tokens = normalize(word.text)
            self.tokens.extend(tokens)
            self.token_words.extend([i] * len(tokens))

    def render(self):
        return '\n'.join(f'[{i}] ' + ' '.join(w.text for w in self.words[start:end])
                         for i, (start, end) in enumerate(self.sentences))

    def to_dict(self):
        return {'words': [vars(word) for word in self.words], 'sentences': self.sentences}

    @classmethod
    def from_dict(cls, value):
        transcript = cls([Word(**word) for word in value['words']])
        if 'sentences' in value:
            transcript.sentences = [tuple(pair) for pair in value['sentences']]
        return transcript


def normalize(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower().replace('\u2019', "'"))


def align_quote(transcript, quote, sentence_id=None):
    query = normalize(quote)
    tokens = transcript.tokens
    if not query or not tokens:
        return None
    anchor = None
    if type(sentence_id) is int and 0 <= sentence_id < len(transcript.sentences):
        anchor = transcript.sentences[sentence_id][0]
    n = len(query)
    exact = [i for i in range(len(tokens) - n + 1) if tokens[i:i + n] == query]
    if exact:
        start = min(exact, key=lambda i: abs(transcript.token_words[i] - anchor)) if anchor is not None else exact[0]
        end = start + n
    else:
        best = (0.0, 0, 0)
        candidates = [i for i, token in enumerate(tokens) if token in query[:2]]
        for i in candidates:
            for length in range(max(1, n - 3), min(len(tokens) - i, n + 4) + 1):
                window = tokens[i:i + length]
                if [token for token in query if token.isdigit()] != [token for token in window if token.isdigit()]:
                    continue
                matcher = difflib.SequenceMatcher(None, query, window, autojunk=False)
                ratio = matcher.ratio()
                distance = abs(transcript.token_words[i] - anchor) if anchor is not None else 0
                score = ratio - min(0.08, distance * 0.001)
                if score > best[0]:
                    blocks = [block for block in matcher.get_matching_blocks() if block.size]
                    if blocks:
                        best = (score, i + blocks[0].b, i + blocks[-1].b + blocks[-1].size)
        score, start, end = best
        if score < 0.72 or end <= start:
            return None
    first = transcript.words[transcript.token_words[start]]
    last = transcript.words[transcript.token_words[end - 1]]
    return (round(first.start, 3), round(last.end, 3)) if last.end > first.start else None


def parse_json_entries(text, count):
    """The raw entries of the model's "evidence" array, in order."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
    match = re.search(r'"evidence"\s*:\s*\[', text)
    entries = []
    if match:
        position = match.end()
        decoder = json.JSONDecoder()
        while len(entries) < count:
            while position < len(text) and text[position] in ' \r\n\t,':
                position += 1
            if position >= len(text) or text[position] == ']':
                break
            try:
                value, position = decoder.raw_decode(text, position)
            except json.JSONDecodeError:
                break
            entries.append(value)
    return entries


def parse_evidence(text, count):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
    match = re.search(r'"evidence"\s*:\s*\[', text)
    values = []
    if match:
        position = match.end()
        decoder = json.JSONDecoder()
        while len(values) < count:
            while position < len(text) and text[position] in ' \r\n\t,':
                position += 1
            if position >= len(text) or text[position] == ']':
                break
            try:
                value, position = decoder.raw_decode(text, position)
            except json.JSONDecodeError:
                break
            if isinstance(value, list) and len(value) == 3 and type(value[0]) is bool:
                value = value[1:] if value[0] else None
            if value is None or value is False:
                values.append(None)
            elif (isinstance(value, list) and len(value) == 2 and
                  type(value[0]) is int and isinstance(value[1], str)):
                values.append(value)
            else:
                values.append('missing')
    return (values + ['missing'] * count)[:count]


def parse_ranges(text, count):
    """Like parse_evidence, but every entry is [yes, first_line, last_line]."""
    values = []
    for item in parse_json_entries(text, count):
        if isinstance(item, list) and len(item) in (3, 4) and type(item[0]) is bool:
            if not item[0]:
                values.append(None)
                continue
            # [yes, first, last] or [yes, anchor, first, last]
            anchor, first, last = (item[1], item[1], item[2]) if len(item) == 3 else (item[1], item[2], item[3])
            ok = all(type(v) is int for v in (anchor, first, last)) and 0 <= first <= last
            values.append([first, last, min(max(anchor, first), last)] if ok else 'missing')
        else:
            values.append('missing')
    return (values + ['missing'] * count)[:count]


def response_from_ranges(transcript, questions, evidence):
    """Turn [first_line, last_line] answers into spans over those transcript lines."""
    answers, starts, ends = [], [], []
    for i, question in enumerate(questions):
        item = evidence[i] if i < len(evidence) else 'missing'
        if item is None:
            answer, span = False, None
        elif item == 'missing':
            span = retrieve_fallback(transcript, question)
            answer = span is not None
        else:
            answer = True
            first, last = min(item[0], len(transcript.sentences) - 1), min(item[1], len(transcript.sentences) - 1)
            a = transcript.sentences[first][0]
            b = transcript.sentences[last][1]
            span = (transcript.words[a].start, transcript.words[b - 1].end) if b > a else retrieve_fallback(transcript, question)
        span = calibrate_start(transcript, span)
        if answer:
            span = add_context_to_short_span(transcript, span)
        answers.append(answer)
        starts.append(round(span[0], 3) if span else None)
        ends.append(round(span[1], 3) if span else None)
    return ASRQuestionResponseDto(answers=answers, evidence_start=starts, evidence_end=ends)


def answer_from_raw(transcript, questions, raw):
    """Parse one model reply and build the response, in whichever evidence mode is configured."""
    from llm import evidence_mode

    if evidence_mode() == 'range':
        evidence = parse_ranges(raw, len(questions))
        return response_from_ranges(transcript, questions, evidence), evidence
    evidence = parse_evidence(raw, len(questions))
    return response_from_evidence(transcript, questions, evidence), evidence


STOPWORDS = set('a an the is are was were be been being do does did has have had will would should '
                'can could patient doctor any there of for to in on at with by and or as that this '
                'it its their they them about from visit conversation mention discussed discussion '
                'right correct still also'.split())


def retrieve_fallback(transcript, question):
    query = set(normalize(question)) - STOPWORDS
    best = (0, None)
    for start, end in transcript.sentences:
        words = transcript.words[start:end]
        tokens = set(normalize(' '.join(word.text for word in words))) - STOPWORDS
        score = len(tokens & query) / max(1, len(query))
        if score > best[0]:
            best = (score, (words[0].start, words[-1].end))
    return best[1] if best[0] >= 0.35 else None


EVIDENCE_STOPWORDS = STOPWORDS | set('take taking taken given found reported report say says said '
                                   'considered described known mention include includes among total '
                                   'current regular regularly actually patient doctor examination findings'.split())


def evidence_keywords(text):
    result = set()
    for token in normalize(text):
        if token in EVIDENCE_STOPWORDS or len(token) < 4:
            continue
        if token.endswith('ing') and len(token) > 6:
            token = token[:-3]
        elif token.endswith('ed') and len(token) > 5:
            token = token[:-2]
        elif token.endswith('s') and not token.endswith('ss'):
            token = token[:-1]
        result.add(token)
    return result


def refine_span(transcript, question, span):
    if span is None:
        return None
    words = transcript.words
    selected = [i for i, word in enumerate(words) if word.end > span[0] + 0.001 and word.start < span[1] - 0.001]
    if not selected:
        return span
    start, end = selected[0], selected[-1] + 1
    target = evidence_keywords(question)
    present = evidence_keywords(' '.join(word.text for word in words[start:end]))
    generic = set('well unwell tolerat treatment mean since weekend fine feel felt feeling '
                  'help work what hope hear good better lately recently'.split())
    if present <= generic and not (target & present):
        alternatives = []
        for index, (a, b) in enumerate(transcript.sentences):
            if abs(words[a].start - span[0]) > 12:
                continue
            text = ' '.join(word.text for word in words[a:b])
            matches = len(target & evidence_keywords(text))
            if matches < 2:
                continue
            if text.endswith('?'):
                if index + 1 == len(transcript.sentences):
                    continue
                c, d = transcript.sentences[index + 1]
                reply = ' '.join(word.text for word in words[c:d]).lower()
                if d - c > 4 or not re.match(r'^(yes|no|none|correct|exactly|right)\b', reply):
                    continue
            alternatives.append((matches, -(b - a), -abs(words[a].start - span[0]), a, b))
        if alternatives:
            _, _, _, start, end = max(alternatives)
    for _ in range(2):
        present = evidence_keywords(' '.join(word.text for word in words[start:end]))
        missing = target - present
        candidates = []
        for a, b in transcript.sentences:
            adjacent = b == start or a == end or (a <= start and b >= end)
            if not adjacent or (a == start and b == end):
                continue
            if b <= start and words[start].start - words[b - 1].end > 2:
                continue
            if a >= end and words[a].start - words[end - 1].end > 2:
                continue
            text = ' '.join(word.text for word in words[a:b])
            gained = evidence_keywords(text) & missing
            if not gained:
                continue
            if b - a > 16 and (b <= start or a >= end):
                cuts = [a] + [i + 1 for i in range(a, b) if words[i].text.endswith((',', ';'))] + [b]
                clauses = [(c, d) for c, d in zip(cuts, cuts[1:]) if c < d and
                           gained <= evidence_keywords(' '.join(word.text for word in words[c:d]))]
                if clauses:
                    a, b = min(clauses, key=lambda pair: pair[1] - pair[0])
            new_start, new_end = min(start, a), max(end, b)
            if words[new_end - 1].end - words[new_start].start <= 16:
                candidates.append((len(gained), -(new_end - new_start), new_start, new_end))
        if not candidates:
            break
        _, _, start, end = max(candidates)
    if end < len(words) and words[end - 1].text.endswith('?'):
        following = next(((a, b) for a, b in transcript.sentences if a == end), None)
        if following:
            a, b = following
            text = ' '.join(word.text for word in words[a:b]).lower()
            if b - a <= 4 and re.match(r'^(yes|no|none|correct|exactly|right)\b', text):
                end = b
    return round(words[start].start, 3), round(words[end - 1].end, 3)


# Whisper's word start times include a little of the pause or breath before the word, while the annotated
# evidence starts at the speech itself: where a span covers exactly the annotated words, its start was a median
# 0.18 s early (0.18 s on the first 29 supplied conversations, 0.16 s on the last 10) and its end was on time.
SPAN_START_DELAY_SECONDS = 0.18


def calibrate_start(transcript, span):
    """Move a span's start later by SPAN_START_DELAY_SECONDS, never past the end of its first word."""
    if span is None:
        return None
    first = next((word for word in transcript.words if word.end > span[0] + 0.01), None)
    if first is None or first.end - 0.05 <= span[0]:
        return span
    return min(span[0] + SPAN_START_DELAY_SECONDS, first.end - 0.05), span[1]


# A reply shorter than this carries no evidence on its own ("None.", "A lot of wind."), so the sentence
# it answers is prepended. On the supplied data it gained 0.024 tIoU on the first 29 conversations
# (8 spans better, 3 worse) and lost 0.016 on the last 10 (2 spans touched), so it was settled on
# validation: 0.7515 -> 0.7617, about +0.017 tIoU. MEDICAL_SHORT_CONTEXT overrides; 0 disables it.
SHORT_SPAN_CONTEXT_SECONDS = float(os.getenv('MEDICAL_SHORT_CONTEXT', '1.0'))


def add_context_to_short_span(transcript, span, gap=2.0):
    """Prepend the preceding sentence to a very short span, if it is close enough in time."""
    if span is None or not SHORT_SPAN_CONTEXT_SECONDS or span[1] - span[0] >= SHORT_SPAN_CONTEXT_SECONDS:
        return span
    words = transcript.words
    selected = [i for i, word in enumerate(words) if word.end > span[0] + 0.01 and word.start < span[1] - 0.01]
    if not selected:
        return span
    preceding = [(a, b) for a, b in transcript.sentences if b <= selected[0]]
    if not preceding or words[selected[0]].start - words[preceding[-1][1] - 1].end > gap:
        return span
    return calibrate_start(transcript, (words[preceding[-1][0]].start, span[1]))


def response_from_evidence(transcript, questions, evidence):
    answers, starts, ends = [], [], []
    for i, question in enumerate(questions):
        item = evidence[i] if i < len(evidence) else 'missing'
        if item is None:
            answer, span = False, None
        elif item == 'missing':
            span = retrieve_fallback(transcript, question)
            answer = span is not None
        else:
            answer = True
            span = align_quote(transcript, item[1], item[0])
            if span is None:
                span = retrieve_fallback(transcript, question)
            if os.getenv('REFINE_EVIDENCE', '1') == '1':
                span = refine_span(transcript, question, span)
        span = calibrate_start(transcript, span)
        if answer:
            span = add_context_to_short_span(transcript, span)
        answers.append(answer)
        starts.append(round(span[0], 3) if span else None)
        ends.append(round(span[1], 3) if span else None)
    return ASRQuestionResponseDto(answers=answers, evidence_start=starts, evidence_end=ends)


def fallback_response(questions):
    return ASRQuestionResponseDto(answers=[False] * len(questions),
                                  evidence_start=[None] * len(questions),
                                  evidence_end=[None] * len(questions))


class MedicalSolver:
    def __init__(self):
        from llm import LocalLanguageModel
        from speech import SpeechRecognizer
        self.lock = threading.Lock()
        self.asr = SpeechRecognizer()
        self.llm = LocalLanguageModel()
        self.asr.warmup()
        self.llm.warmup()
        logger.info('Speech and language models loaded and warmed up')

    def answer(self, transcript, questions, deadline):
        text = self.llm.complete(transcript, questions, deadline)
        response, evidence = answer_from_raw(transcript, questions, text)
        logger.info('LLM: %d/%d complete answers', sum(item != 'missing' for item in evidence), len(questions))
        return response, text

    def predict(self, audio_bytes, questions):
        if not questions:
            return fallback_response(questions)
        started = time.monotonic()
        deadline = started + float(os.getenv('REQUEST_BUDGET_SECONDS', '55'))
        if not self.lock.acquire(timeout=max(0.01, deadline - time.monotonic())):
            return fallback_response(questions)
        try:
            transcript = self.asr.transcribe(audio_bytes)
            response, _ = self.answer(transcript, questions, deadline)
            logger.info('Request complete in %.2fs', time.monotonic() - started)
            return response
        finally:
            self.lock.release()


@lru_cache(maxsize=1)
def get_solver():
    return MedicalSolver()
