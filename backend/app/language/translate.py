"""Phase 8: text translation via IndicTrans2 (ai4bharat), loaded directly
through HuggingFace `transformers` — not Ollama-servable, so this bypasses
app/router.py but still reads its two model_ids from the same
config/models.yaml registry via app.config.get_active_candidate(), same as
every other model in this project.

Four separate transformers-5.x compatibility shims live at the top of this
file, all for the same underlying reason: this project can't pin
transformers down (sentence-transformers, Phase 3's reranker, already
requires >=5), but IndicTrans2's own tooling — both the pip package and the
model's own `trust_remote_code=True` config/tokenizer/model code hosted on
HuggingFace — still assumes an older transformers layout. Each was found by
actually running a real translation with a real, authenticated HF_TOKEN and
reading the next traceback, not guessed at.

**STILL NOT WORKING END TO END**, honestly: with all four shims applied,
loading gets all the way through tokenizer + model construction + weight
tying, and `.generate()` starts running real inference — further than any
previous session got. It then fails inside the remote model's own custom
self-attention forward() at `torch.cat([past_key_value[0], key_states],
dim=2)`, a shape mismatch. Root cause: this remote code's hand-written
attention layers manually concatenate past/new key-value tensors assuming
the old tuple-cache's empty state was a real, correctly-shaped 4D tensor;
transformers 5.x's Cache-object equivalent reports `None` for an untouched
layer instead, and a same-shaped placeholder isn't obviously constructible
from outside the attention layer (its batch size/head count/head dim aren't
known at the point the cache is queried). This is no longer a simple
"missing import/method/kwarg" gap like the four shims below — it's the
remote model's own low-level attention math not matching the new Cache
object model at a structural level, and further blind patching risks
producing translations that run without crashing but are silently wrong
(already caught once: disabling caching entirely avoided this exact crash
but produced empty output for every input, with no error at all). Stopped
here rather than keep guessing at tensor shapes with no way to verify
correctness short of a fluent speaker checking the output. The four shims
below are real, verified, and worth keeping regardless — they're not wasted
effort, they got this from "won't import" to "runs real inference,
crashes deep inside custom attention code" — but actual translation output
is not yet achievable without either patching this specific model's
attention implementation properly (a bigger, riskier undertaking) or
running this one path against an isolated older transformers install.
"""

import sys
import threading
import types

# Shim 1/4 — IndicTransToolkit 1.1.1's collator.py does
# `from transformers.tokenization_utils import PreTrainedTokenizerBase`,
# which transformers 5.x no longer re-exports from that module (it moved to
# tokenization_utils_base) — breaking the package's own __init__.py import
# chain even though only IndicProcessor (not the collator) is used here.
# Verified live: this shim alone is sufficient for
# `from IndicTransToolkit import IndicProcessor` to succeed against the
# installed transformers 5.15.1.
import transformers.tokenization_utils as _tu
from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PreTrainedTokenizerBase

_tu.PreTrainedTokenizerBase = _PreTrainedTokenizerBase

# Shim 2/4 — once past the gated-repo wall (real HF_TOKEN + accepted terms),
# loading the model itself failed with `ModuleNotFoundError: No module named
# 'transformers.onnx'`: the model repo's own remote `configuration_indictrans.py`
# (fetched and exec'd by AutoConfig.from_pretrained(..., trust_remote_code=True))
# does `from transformers.onnx import OnnxConfig, OnnxSeq2SeqConfigWithPast` —
# a submodule transformers 5.x removed entirely (ONNX export moved to the
# separate `optimum` package). Confirmed by reading the cached remote file
# directly: those two classes are used *only* to define an unused
# `IndicTransOnnxConfig` (ONNX-export config) that this app never
# instantiates or calls — we only ever do AutoTokenizer/AutoModelForSeq2SeqLM
# loading and .generate(), never an ONNX export. A minimal stub (real class,
# just no working ONNX behavior) is enough to let that unused class
# definition import cleanly rather than crash the whole module.
if "transformers.onnx" not in sys.modules:
    _onnx_stub = types.ModuleType("transformers.onnx")

    class _OnnxConfig:
        default_fixed_batch = 2
        default_fixed_sequence = 8

    class _OnnxSeq2SeqConfigWithPast(_OnnxConfig):
        pass

    _onnx_stub.OnnxConfig = _OnnxConfig
    _onnx_stub.OnnxSeq2SeqConfigWithPast = _OnnxSeq2SeqConfigWithPast

    _onnx_utils_stub = types.ModuleType("transformers.onnx.utils")

    def _compute_effective_axis_dimension(dimension: int, fixed_dimension: int, num_token_to_add: int = 0) -> int:
        # Real ONNX dummy-input sizing logic, never actually exercised here
        # (only reachable via IndicTransOnnxConfig.generate_dummy_inputs,
        # which this app never calls) — good enough to be importable.
        return fixed_dimension if dimension == -1 else dimension

    _onnx_utils_stub.compute_effective_axis_dimension = _compute_effective_axis_dimension
    _onnx_stub.utils = _onnx_utils_stub

    sys.modules["transformers.onnx"] = _onnx_stub
    sys.modules["transformers.onnx.utils"] = _onnx_utils_stub

# Shim 3/4 — past the onnx wall, tokenizer loading itself then failed with
# `AttributeError: IndicTransTokenizer has no attribute _special_tokens_map`.
# Root cause, confirmed by reading the cached remote `tokenization_indictrans.py`
# directly: `IndicTransTokenizer.__init__` sets `self.unk_token = ...` (and
# pad/eos/bos) as its *first* lines, calling `super().__init__()` only at the
# very end. Under transformers 5.x, PreTrainedTokenizerBase.__setattr__
# intercepts special-token attribute names and routes them through
# `self._special_tokens_map[key] = value` — a dict that normally only exists
# once `super().__init__()` has run. Sets before that point hit
# `self._special_tokens_map`, find no instance *or* class attribute of that
# name anywhere, and raise via __getattr__. This ordering was fine under the
# older transformers version this remote code was written against (plain
# instance attributes, no interception); the model repo hasn't been updated
# for 5.x's stricter behavior. Fixed by lazily creating `_special_tokens_map`
# on first write instead of requiring `__init__` to have created it already —
# the exact same dict `PreTrainedTokenizerBase.__init__` itself creates
# (`dict.fromkeys(self.SPECIAL_TOKENS_ATTRIBUTES)`), just available a few
# lines earlier. When `super().__init__()` does eventually run, it
# recreates/overwrites the same dict, so this has no effect once
# initialization actually completes.
from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTB

_original_tokenizer_setattr = _PTB.__setattr__


def _lazy_special_tokens_map_setattr(self, key, value):
    if "_special_tokens_map" not in self.__dict__:
        object.__setattr__(self, "_special_tokens_map", dict.fromkeys(self.SPECIAL_TOKENS_ATTRIBUTES))
    _original_tokenizer_setattr(self, key, value)


_PTB.__setattr__ = _lazy_special_tokens_map_setattr

# Shim 4/4 — past the tokenizer wall, model loading itself then failed with
# `TypeError: IndicTransForConditionalGeneration.tie_weights() got an
# unexpected keyword argument 'recompute_mapping'`. transformers 5.x's own
# `PreTrainedModel.init_weights()` (and, separately, its own
# `from_pretrained()` weight-loading path) both now call
# `self.tie_weights(recompute_mapping=False)` / `(missing_keys=..., recompute_mapping=False)`
# — but the remote `modeling_indictrans.py`'s `IndicTransForConditionalGeneration`
# overrides `tie_weights` with the old zero-argument signature, predating
# both new parameters. Two call sites (not one) hit this with the exact same
# class, so rather than patching each call site individually (fragile if
# there turn out to be more, and the second one is buried deep inside
# `from_pretrained`'s own body, not a small method worth reimplementing),
# this hooks `PreTrainedModel.__init_subclass__` — a standard Python hook
# that fires for *every* subclass definition, including ones `exec`'d from a
# dynamically downloaded remote file — and wraps `tie_weights` to silently
# drop unsupported kwargs, but only for a subclass whose own override
# actually can't accept them (checked via real signature inspection, not
# assumed) — so it does nothing for any normal, up-to-date model class.
import inspect

from transformers.modeling_utils import PreTrainedModel as _PTM

_original_init_subclass = _PTM.__init_subclass__


def _tie_weights_compat_init_subclass(cls, **kwargs) -> None:
    _original_init_subclass(**kwargs)
    method = cls.__dict__.get("tie_weights")
    if method is None:
        return
    try:
        params = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return
    already_compatible = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params) or {"missing_keys", "recompute_mapping"} <= {
        p.name for p in params
    }
    if already_compatible:
        return

    def _tolerant_tie_weights(self, *args, **kwargs):
        return method(self)

    cls.tie_weights = _tolerant_tie_weights


_PTM.__init_subclass__ = classmethod(_tie_weights_compat_init_subclass)

# Shim 4b/4 — the remote tie_weights() above, now actually reached, calls
# `self._tie_or_clone_weights(output_embeddings, input_embeddings)` — a
# PreTrainedModel utility method that existed in the older transformers this
# remote code was written against but has been fully removed from
# transformers 5.x (superseded by the new tie_weights()'s own internal
# logic, which doesn't expose an equivalent public method). Restored here as
# the well-known historical implementation (tie by sharing the weight
# tensor directly, or clone it under torchscript; pad any bias to match; fix
# up out_features) — simple, self-contained, standard PyTorch/HF weight-tying,
# not something that depends on any other removed transformers internals.
import torch.nn as _nn


def _tie_or_clone_weights(self, output_embeddings, input_embeddings) -> None:
    if getattr(self.config, "torchscript", False):
        output_embeddings.weight = _nn.Parameter(input_embeddings.weight.clone())
    else:
        output_embeddings.weight = input_embeddings.weight
    if getattr(output_embeddings, "bias", None) is not None:
        output_embeddings.bias.data = _nn.functional.pad(
            output_embeddings.bias.data,
            (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0]),
            "constant",
            0,
        )
    if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
        output_embeddings.out_features = input_embeddings.num_embeddings


if not hasattr(_PTM, "_tie_or_clone_weights"):
    _PTM._tie_or_clone_weights = _tie_or_clone_weights

# Shim 4c/4 — with tie_weights fixed, model loading finishes, but the remote
# forward() itself does `past_key_values[0][0].shape[2]` — old tuple-style
# KV-cache access (`[layer][key_or_value]`). transformers 5.x replaced the
# plain-tuple cache with `Cache`/`EncoderDecoderCache` objects, which support
# iteration (yielding one tuple per layer — see EncoderDecoderCache.__iter__)
# but not integer indexing. First tried disabling caching outright
# (`use_cache=False`) to sidestep the whole codepath — that avoided the
# crash but silently produced degenerate output (immediate EOS, empty
# translations for every input), so caching is actually load-bearing for
# this model's generation quality, not just a performance nicety. Adding
# `__getitem__` (delegating to the existing `__iter__`) instead of disabling
# caching preserves real KV-cache behavior while satisfying the remote
# code's old-style access pattern.
import torch as _torch
from transformers.cache_utils import EncoderDecoderCache as _EDC

if not hasattr(_EDC, "__getitem__"):

    def _encoder_decoder_cache_getitem(self, layer_idx: int):
        layer = tuple(self)[layer_idx]
        # A layer that hasn't been written to yet (the very first forward
        # pass, before any decoding step) reports None for its
        # keys/values/sliding-window tensor — the remote code then does
        # `past_key_values[0][0].shape[2]` unconditionally to read how many
        # tokens are already cached, which crashes on None. Old-style
        # tuple caches never had this problem (an empty cache was just
        # `None` at the top level, checked with `is not None`); the new
        # object-based cache is non-None even when empty, so that check no
        # longer catches this case. A zero-length placeholder tensor makes
        # `.shape[2]` read as 0, matching what the old cache semantics
        # actually meant by "nothing cached yet".
        return tuple(t if t is not None else _torch.empty(0, 0, 0, 0) for t in layer)

    _EDC.__getitem__ = _encoder_decoder_cache_getitem

import torch
from IndicTransToolkit import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.config import get_active_candidate

# fastText lid.176 code -> IndicTrans2/FLORES-200 code, for the languages
# named across Phase 8's bullets (Hindi/Tamil/Telugu/Kannada/Malayalam).
LANG_CODE_TO_FLORES = {
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "en": "eng_Latn",
}

SUPPORTED_LANGUAGES = {"hi", "ta", "te", "kn", "ml"}

_lock = threading.Lock()
_loaded: dict[str, tuple] = {}  # task_type -> (tokenizer, model)
_processor: IndicProcessor | None = None


def _get_processor() -> IndicProcessor:
    global _processor
    if _processor is None:
        _processor = IndicProcessor(inference=True)
    return _processor


def _get_model(task_type: str):
    """task_type is "translation_en_indic" or "translation_indic_en" —
    resolved via the model registry (app.config) exactly like every other
    task type, so switching the active candidate works the same way here
    too, even though loading itself goes through transformers, not Ollama."""
    with _lock:
        if task_type not in _loaded:
            candidate = get_active_candidate(task_type)
            tokenizer = AutoTokenizer.from_pretrained(candidate.model_id, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(candidate.model_id, trust_remote_code=True)
            model.eval()
            _loaded[task_type] = (tokenizer, model)
        return _loaded[task_type]


def _translate(text: str, src_flores: str, tgt_flores: str, task_type: str) -> str:
    tokenizer, model = _get_model(task_type)
    ip = _get_processor()
    batch = ip.preprocess_batch([text], src_lang=src_flores, tgt_lang=tgt_flores)
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        # NOTE: use_cache=False was tried first here to sidestep Shim 4c's
        # problem entirely, and it did avoid the crash — but it silently
        # produced degenerate output (immediate EOS, empty translations for
        # every input): caching turned out to be load-bearing for this
        # model's generation quality, not just a performance nicety. Fixed
        # properly instead (Shim 4c, EncoderDecoderCache.__getitem__), so
        # caching stays on here, as it should by default.
        outputs = model.generate(**inputs, max_length=256, num_beams=5)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return ip.postprocess_batch(decoded, lang=tgt_flores)[0]


def translate_to_english(text: str, source_language: str) -> str:
    """`source_language` is a fastText lid.176 code (e.g. "hi")."""
    if source_language not in LANG_CODE_TO_FLORES:
        raise ValueError(f"Unsupported source language '{source_language}'. Expected one of: {sorted(SUPPORTED_LANGUAGES)}")
    return _translate(text, LANG_CODE_TO_FLORES[source_language], "eng_Latn", "translation_indic_en")


def translate_from_english(text: str, target_language: str) -> str:
    if target_language not in LANG_CODE_TO_FLORES:
        raise ValueError(f"Unsupported target language '{target_language}'. Expected one of: {sorted(SUPPORTED_LANGUAGES)}")
    return _translate(text, "eng_Latn", LANG_CODE_TO_FLORES[target_language], "translation_en_indic")
