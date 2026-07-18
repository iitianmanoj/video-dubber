"""
translation/translator.py
--------------------------
Translates transcript segments into natural, meaning-preserving English.

Backend is pluggable via `TranslationConfig.engine`:
  - "google"      (default): deep-translator's GoogleTranslate wrapper.
                   No API key required, good general-purpose quality,
                   works reliably out of the box — this is what makes the
                   pipeline actually runnable end-to-end for the assignment.
  - "nllb"        : Meta's No Language Left Behind model, run locally via
                   transformers. Higher quality for low-resource languages,
                   but requires a GPU and a multi-GB model download.
  - "indictrans2" : AI4Bharat's IndicTrans2, purpose-built for Indian
                   languages -> English. Best choice for Hindi, Tamil,
                   Telugu, Bengali, Marathi, Gujarati, Urdu, etc.

All backends implement the same `translate_segments` interface, so the
rest of the pipeline doesn't care which one is active.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import List

from config import CONFIG
from transcription.whisper_engine import Segment
from utils.logger import get_logger

logger = get_logger()

# Indic languages IndicTrans2 was specifically trained for.
_INDIC_LANGS = {"hi", "ta", "te", "bn", "mr", "gu", "ur", "pa", "kn", "ml", "or", "as"}


class TranslationError(RuntimeError):
    pass


class BaseTranslationBackend(ABC):
    @abstractmethod
    def translate_batch(self, texts: List[str], source_lang: str) -> List[str]:
        """Translate a batch of source-language strings into English."""


class GoogleTranslateBackend(BaseTranslationBackend):
    """Free, no-key translation via deep-translator's Google backend.

    Preserves meaning/tone reasonably well for short conversational
    segments (which is what Whisper segments typically are), and is the
    most dependable option to run without any credentials or GPU.
    """

    def translate_batch(self, texts: List[str], source_lang: str) -> List[str]:
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(source="auto", target="en")
        results: List[str] = []
        for text in texts:
            attempt = 0
            while True:
                attempt += 1
                try:
                    translated = translator.translate(text)
                    results.append(translated or text)
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt >= 3:
                        logger.warning(
                            "Translation failed for segment after 3 attempts "
                            "(%s); keeping original text as fallback.", exc,
                        )
                        results.append(text)
                        break
                    time.sleep(1.5 * attempt)
        return results


class NLLBBackend(BaseTranslationBackend):
    """Local translation via Meta's NLLB-200, for higher-quality offline use."""

    _FLORES_CODES = {
        "hi": "hin_Deva", "de": "deu_Latn", "fr": "fra_Latn", "es": "spa_Latn",
        "it": "ita_Latn", "pt": "por_Latn", "ru": "rus_Cyrl", "ja": "jpn_Jpan",
        "ko": "kor_Hang", "zh": "zho_Hans", "ar": "arb_Arab", "ta": "tam_Taml",
        "te": "tel_Telu", "bn": "ben_Beng", "mr": "mar_Deva", "gu": "guj_Gujr",
        "ur": "urd_Arab", "tr": "tur_Latn", "nl": "nld_Latn",
    }

    def __init__(self):
        self._pipe = None

    def _load(self, source_lang: str):
        from transformers import pipeline

        src_code = self._FLORES_CODES.get(source_lang)
        if src_code is None:
            raise TranslationError(f"NLLB backend has no mapping for language '{source_lang}'")

        if self._pipe is None:
            self._pipe = pipeline(
                "translation",
                model="facebook/nllb-200-distilled-600M",
                src_lang=src_code,
                tgt_lang="eng_Latn",
            )
        return self._pipe

    def translate_batch(self, texts: List[str], source_lang: str) -> List[str]:
        pipe = self._load(source_lang)
        outputs = pipe(texts, max_length=400)
        return [o["translation_text"] for o in outputs]


class IndicTrans2Backend(BaseTranslationBackend):
    """Local translation via AI4Bharat's IndicTrans2 (Indian languages -> English)."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from IndicTransToolkit.processor import IndicProcessor
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            model_name = "ai4bharat/indictrans2-indic-en-1B"
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
            self._processor = IndicProcessor(inference=True)
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
        except ImportError as exc:
            raise TranslationError(
                "IndicTrans2 backend requires 'IndicTransToolkit' and its "
                "dependencies. Install per the README, or set "
                "TRANSLATION_ENGINE=google to use the default backend."
            ) from exc

    def translate_batch(self, texts: List[str], source_lang: str) -> List[str]:
        import torch

        self._load()
        lang_map = {
            "hi": "hin_Deva", "ta": "tam_Taml", "te": "tel_Telu", "bn": "ben_Beng",
            "mr": "mar_Deva", "gu": "guj_Gujr", "ur": "urd_Arab", "pa": "pan_Guru",
            "kn": "kan_Knda", "ml": "mal_Mlym", "or": "ory_Orya",
        }
        src_tag = lang_map.get(source_lang)
        if src_tag is None:
            raise TranslationError(f"IndicTrans2 has no mapping for language '{source_lang}'")

        batch = self._processor.preprocess_batch(texts, src_lang=src_tag, tgt_lang="eng_Latn")
        inputs = self._tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            generated = self._model.generate(
                **inputs, max_length=256, num_beams=5, early_stopping=True,
            )
        decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
        return self._processor.postprocess_batch(decoded, lang="eng_Latn")


def _select_backend(source_lang: str) -> BaseTranslationBackend:
    engine = CONFIG.translation.engine
    if engine == "indictrans2" or (engine == "auto" and source_lang in _INDIC_LANGS):
        return IndicTrans2Backend()
    if engine == "nllb":
        return NLLBBackend()
    return GoogleTranslateBackend()


class Translator:
    """Translates a list of transcript Segments into natural English."""

    def translate_segments(self, segments: List[Segment], source_lang: str) -> List[Segment]:
        if source_lang == "en":
            logger.info("Source language is already English; skipping translation.")
            return segments

        backend = _select_backend(source_lang)
        texts = [seg.text for seg in segments]

        try:
            translated_texts = backend.translate_batch(texts, source_lang)
        except TranslationError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Primary translation backend failed (%s); falling back to "
                "Google Translate.", exc,
            )
            translated_texts = GoogleTranslateBackend().translate_batch(texts, source_lang)

        if len(translated_texts) != len(segments):
            raise TranslationError(
                "Translation backend returned a mismatched number of segments."
            )

        translated_segments = [
            Segment(id=seg.id, start=seg.start, end=seg.end, text=text.strip(), speaker=seg.speaker)
            for seg, text in zip(segments, translated_texts)
        ]
        return translated_segments
