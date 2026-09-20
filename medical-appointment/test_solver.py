import base64
import json
from unittest.mock import patch

import pytest

from dtos import ASRQuestionRequestDto
from solver import Transcript, Word, align_quote, parse_evidence, response_from_evidence
from utils import validate_response


@pytest.fixture(autouse=True)
def exact_word_starts(monkeypatch):
    """These tests check which words the evidence covers; keep word starts uncalibrated here."""
    monkeypatch.setattr('solver.SPAN_START_DELAY_SECONDS', 0.0)



def transcript(text):
    return Transcript([Word(i * 0.4, (i + 1) * 0.4, word) for i, word in enumerate(text.split())])


def test_exact_quote_uses_word_boundaries():
    speech = transcript('Take fluconazole, fifty milligrams for seven days. Then stop.')
    assert align_quote(speech, 'fifty milligrams for seven days.', 0) == (0.8, 2.8)


def test_anchor_disambiguates_repeated_confirmation():
    speech = transcript('Is the chest clear? Yes. Is the heart normal? Yes.')
    assert align_quote(speech, 'Yes.', 3) == (3.6, 4.0)


def test_multi_sentence_evidence():
    speech = transcript('Your lungs sound clear. Your heart is normal. Continue treatment.')
    assert align_quote(speech, 'Your lungs sound clear. Your heart is normal.', 0) == (0.0, 3.2)


def test_unrelated_and_empty_quotes_do_not_get_timestamps():
    speech = transcript('Take fifty milligrams after food.')
    assert align_quote(speech, 'Attend a concert on Saturday.', 0) is None
    assert align_quote(speech, '', 0) is None


def test_fuzzy_alignment_rejects_a_changed_numeric_dose():
    speech = transcript('Take 100 milligrams every morning with food.')
    assert align_quote(speech, 'Take 200 milligrams every morning with food.', 0) is None


def test_minor_punctuation_and_whitespace_differences():
    speech = transcript('We renewed the anti-inflammatory medicine today.')
    assert align_quote(speech, 'We renewed the anti inflammatory medicine today', 0) == (0.0, 2.4)


def test_partial_generation_preserves_complete_answers():
    text = '{"evidence": [[0, "Continue treatment."], null, [2, "unfinished'
    assert parse_evidence(text, 3) == [[0, 'Continue treatment.'], None, 'missing']


def test_explicit_answers_are_respected_even_with_contradicting_quotes():
    text = json.dumps({'evidence': [[False, 0, 'Take 100 mg.'], [True, 0, 'Take 100 mg.']]})
    assert parse_evidence(text, 2) == [None, [0, 'Take 100 mg.']]


def test_untrusted_cloud_model_url_is_rejected(monkeypatch):
    from llm import LocalLanguageModel
    monkeypatch.setenv('LLM_URL', 'https://203.0.113.1')
    with pytest.raises(ValueError, match='loopback'):
        LocalLanguageModel()


def test_occupied_model_port_is_not_silently_reused():
    from llm import LocalLanguageModel
    model = LocalLanguageModel.__new__(LocalLanguageModel)
    with patch('llm.socket.create_connection'):
        with pytest.raises(RuntimeError, match='occupied'):
            model._start_server()


def test_failed_model_warmup_is_not_reported_ready():
    from llm import LocalLanguageModel
    model = LocalLanguageModel.__new__(LocalLanguageModel)
    with patch.object(model, 'complete', return_value=''):
        with pytest.raises(RuntimeError, match='warmup'):
            model.warmup()


def test_transcript_filters_invalid_timestamps():
    speech = Transcript([Word(float('nan'), 2, 'bad'), Word(3, 2, 'bad'), Word(-1, 2, 'bad'),
                         Word(0, 1, 'Normal.')])
    assert len(speech.words) == 1
    assert align_quote(speech, 'Normal.') == (0, 1)


def test_zero_questions():
    response = response_from_evidence(Transcript([]), [], [])
    validate_response(response, 0)
    assert response.answers == []


def test_parser_does_not_convert_false_strings_to_yes():
    evidence = parse_evidence('{"evidence": [false, "no", null]}', 3)
    assert evidence == [None, 'missing', None]


def test_response_lengths_nulls_and_finite_spans():
    speech = transcript('Continue treatment. Take it after food.')
    response = response_from_evidence(speech, ['Continue?', 'Before food?', 'After food?'],
                                      [[0, 'Continue treatment.'], None, [1, 'after food.']])
    validate_response(response, 3)
    assert response.answers == [True, False, True]
    assert response.evidence_start == [0.0, None, 1.6]
    assert response.evidence_end == [0.8, None, 2.4]


def test_evidence_includes_a_missing_condition_in_the_next_sentence():
    speech = transcript('This is your annual follow-up. It is for asthma. Everything else is fine.')
    response = response_from_evidence(speech, ['Is this an annual asthma follow-up?'],
                                      [[0, 'This is your annual follow-up.']])
    assert response.evidence_end == [3.6]


def test_evidence_includes_the_prescription_not_just_its_dose():
    speech = transcript('And fluconazole. Fifty milligrams for seven days. Goodbye.')
    response = response_from_evidence(speech, ['Will the patient take fluconazole fifty milligrams?'],
                                      [[1, 'Fifty milligrams for seven days.']])
    assert response.evidence_start == [0.0]
    assert response.evidence_end == [2.8]


def test_generic_quote_can_be_relocated_to_an_explicit_nearby_report():
    speech = transcript('I have been unwell. A stomach bug. All the usual gastroenteritis symptoms.')
    response = response_from_evidence(speech, ['Does the patient describe gastroenteritis symptoms?'],
                                      [[0, 'I have been unwell.']])
    assert response.evidence_start == [2.8]
    assert response.evidence_end == [4.8]


def test_side_effect_reply_is_preferred_to_generic_tolerability():
    speech = transcript('Any side effects? None. That means treatment is well tolerated.')
    response = response_from_evidence(speech, ['Has the patient been free of side effects?'],
                                      [[2, 'That means treatment is well tolerated.']])
    assert response.evidence_start == [0.0]
    assert response.evidence_end == [1.6]


def test_specific_diagnosis_is_not_replaced_by_a_nearby_body_part():
    speech = transcript('They look like seborrhea keratosis. The abdomen and lower leg are affected.')
    response = response_from_evidence(speech, ['Do the abdomen and lower leg resemble seborrheic keratoses?'],
                                      [[0, 'They look like seborrhea keratosis.']])
    assert response.evidence_start == [0.0]


def test_evidence_does_not_expand_into_an_unrelated_topic():
    speech = transcript('The heart is normal. Next we discuss a rash. The patient is going home.')
    response = response_from_evidence(speech, ['Is the heart normal?'], [[0, 'The heart is normal.']])
    assert response.evidence_end == [1.6]


def test_invalid_base64_still_returns_well_formed_response():
    from api import predict
    request = ASRQuestionRequestDto(audio_base64='a', audio_filename='x.mp3', questions=['Asthma?'])
    response = predict(request)
    validate_response(response, 1)
    assert response.evidence_start == [None]


def test_empty_audio_still_returns_well_formed_response():
    from api import predict
    request = ASRQuestionRequestDto(audio_base64=base64.b64encode(b'bad audio').decode(),
                                   audio_filename='../../untrusted.mp3', questions=['Asthma?'])
    with patch('api.get_solver') as factory:
        factory.return_value.predict.side_effect = ValueError('bad audio')
        response = predict(request)
    validate_response(response, 1)
    assert response.evidence_start == [None]


def test_http_contract_without_loading_models():
    from fastapi.testclient import TestClient
    from api import app
    from solver import fallback_response
    with patch('api.get_solver') as factory:
        factory.return_value.predict.return_value = fallback_response(['Anything?'])
        with TestClient(app) as client:
            response = client.post('/predict', json={
                'audio_base64': '', 'audio_filename': 'x.mp3', 'questions': ['Anything?'],
            })
            assert response.status_code == 200
            validate_response(type(fallback_response([])).model_validate(response.json()), 1)


def test_span_start_is_calibrated_but_never_past_the_first_word(monkeypatch):
    from solver import Transcript, Word, calibrate_start
    monkeypatch.setattr('solver.SPAN_START_DELAY_SECONDS', 0.18)
    speech = Transcript([Word(1.0, 1.5, 'The'), Word(1.5, 2.0, 'heart'), Word(2.0, 2.1, 'is'), Word(2.1, 2.6, 'normal.')])
    assert calibrate_start(speech, (1.0, 2.6)) == (1.18, 2.6)
    speech = Transcript([Word(1.0, 1.1, 'No.'), Word(1.2, 1.6, 'fever.')])
    assert calibrate_start(speech, (1.0, 1.6)) == (1.05, 1.6)
    assert calibrate_start(speech, None) is None
