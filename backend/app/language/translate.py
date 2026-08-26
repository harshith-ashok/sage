"""Phase 8: text translation via IndicTrans2 (ai4bharat), loaded directly
through HuggingFace `transformers` — not Ollama-servable, so this bypasses
app/router.py but still reads its two model_ids from the same
config/models.yaml registry via app.config.get_active_candidate(), same as
every other model in this project.

Six transformers-5.x compatibility shims live at the top of this file, all
for the same underlying reason: this project can't pin transformers down
(sentence-transformers, Phase 3's reranker, already requires >=5), but
IndicTrans2's own tooling — both the pip package and the model's own
`trust_remote_code=True` config/tokenizer/model code hosted on HuggingFace —
still assumes an older transformers layout. Each was found by actually
running a real translation with a real, authenticated HF_TOKEN and reading
the next traceback (or, in the end, inspecting real output), not guessed
at.

**NOW WORKING END TO END, verified with real output**: `translate_from_english`
and `translate_to_english` produce fluent, correct translations for
en<->hi/ta/te (spot-checked: "The pressure reading is normal." round-trips
through Hindi and back to the identical English sentence; Tamil/Telugu
outputs read as correct, fluent sentences, not garbage or empty strings).
Getting here took six distinct, independently-discovered incompatibilities
— import paths (shims 1-2), tokenizer init ordering (shim 3), weight tying
(shims 4a-4b), attention branch logic (shim 4d) and KV-cache lifecycle
(shims 4c/4e/4f), and finally a completely unrelated bug in position
embeddings (shim 4g) that turned out to be the last mile:

- Shims 4d (stale `elif past_key_value is not None:` branch — old code
  meant "is there real cached data", but the new Cache-object API always
  passes a non-None object even when empty) and 4e (this remote model
  builds its own legacy nested-tuple KV-cache internally and never calls
  `.update()` on the real Cache object it's handed — bridged by writing
  its computed tensors into that object's layers directly) got `.generate()`
  running without crashing, but production was still empty for a while
  after each fix — a real, demonstrated failure mode (silently wrong output,
  no exception) that reproduced independently three separate times across
  this investigation, which is why each fix here was verified against real
  decoded text, not just "did it stop crashing".
- Shim 4c's `EncoderDecoderCache.__getitem__` initially just reused the
  class's own (real, already-existing) `__iter__`, which yields a 6-tuple
  per layer (`self_k, self_v, self_sliding, cross_k, cross_v,
  cross_sliding`) — but the remote model's decoder layer code was written
  for the *old* 4-tuple legacy format and slices `past_key_value[-2:]` for
  cross-attention, which against a 6-tuple silently grabs
  `(cross_v, cross_sliding)` instead of `(cross_k, cross_v)`. Fixed by
  building the legacy 4-tuple directly from each sub-cache's own layers.
- Shim 4f bridges beam search's own `_reorder_cache` call (the model's
  legacy-tuple-shaped override crashes on a real Cache object, which
  already has its own correct `.reorder_cache()` — delegate to it instead).
- **Shim 4g, the actual last blocker**: with generation running cleanly and
  beam search reordering correctly, real output was *still* empty — traced
  to something with nothing to do with caching at all: the encoder's own
  hidden states were NaN from its very first layer. Root cause: this
  model's `IndicTransSinusoidalPositionalEmbedding` computes a real
  sinusoidal position table once in `__init__` and stores it as a
  *non-persistent* buffer (deliberately excluded from the checkpoint, since
  it's pure deterministic math, not learned). transformers 5.x's
  `from_pretrained` initializes the whole model on the meta device by
  default, then re-materializes any non-persistent buffer via
  `torch.empty_like(buffer, ...)` — genuinely uninitialized memory, not a
  recomputation, confirmed by reading `_move_missing_keys_from_meta_to_device`
  in `modeling_utils.py` and by the buffer's actual values (tiny
  denormalized garbage with scattered NaN — the signature of uninitialized
  memory). Fixed by forcing every such buffer to recompute for real
  immediately after loading, via the model's own `make_weights()` method.
"""

import functools as _functools
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

# .__func__: accessing a classmethod via PreTrainedModel.__init_subclass__
# binds it to PreTrainedModel itself, not to whichever subclass is actually
# being defined when this runs later — calling that bound version with only
# **kwargs would silently run every chained check against the wrong class.
# The raw function, called with cls passed through explicitly, doesn't have
# that problem. (Caught live: Shim 4e chaining through this exact capture
# the naive way caused tie_weights patching to silently stop working.)
_original_init_subclass = _PTM.__init_subclass__.__func__


def _tie_weights_compat_init_subclass(cls, **kwargs) -> None:
    _original_init_subclass(cls, **kwargs)
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
# plain-tuple cache with `Cache`/`EncoderDecoderCache` objects. First tried
# disabling caching outright (`use_cache=False`) to sidestep the whole
# codepath — that avoided the crash but silently produced degenerate output
# (immediate EOS, empty translations for every input), so caching is
# actually load-bearing for this model's generation quality, not just a
# performance nicety. Added `__getitem__` instead of disabling caching, to
# preserve real KV-cache behavior while satisfying the remote code's
# old-style `past_key_values[idx]` access pattern.
#
# IMPORTANT: this can't just delegate to `EncoderDecoderCache.__iter__` —
# that already exists on this transformers version, but yields a *6-tuple*
# per layer (`self_k, self_v, self_sliding, cross_k, cross_v, cross_sliding`
# — confirmed by reading its actual source). The remote model's decoder
# layer code was written for the old legacy format, a *4-tuple*
# `(self_k, self_v, cross_k, cross_v)`, and slices it accordingly:
# `past_key_value[:2]` for self-attention (still correct against a 6-tuple,
# happens to grab the same first two elements) but `past_key_value[-2:]`
# for cross-attention — against a 6-tuple this grabs `(cross_v,
# cross_sliding)` instead of `(cross_k, cross_v)`. That silently fed the
# wrong tensor in as cross-attention's *value* (an empty sliding-window
# placeholder), which is exactly what produced
# `RuntimeError: Expected size for first two dimensions of batch2 tensor to
# be: [8, 9] but got: [8, 0]` inside `torch.bmm` during real (non-beam)
# generation. Fixed by building the legacy 4-tuple directly from each
# sub-cache's own layers instead of reusing `__iter__`'s 6-tuple shape.
import torch as _torch
from transformers.cache_utils import EncoderDecoderCache as _EDC

if not hasattr(_EDC, "__getitem__"):

    def _sub_cache_layer_kv(cache, layer_idx: int):
        # A layer that hasn't been written to yet (the very first forward
        # pass, before any decoding step, or a sub-cache — e.g. cross-attn
        # on step 1 — with fewer populated layers than others) has no
        # entry at all, or reports None for keys/values. The remote code
        # then does `past_key_values[0][0].shape[2]` unconditionally to
        # read how many tokens are already cached, which crashes on None.
        # Old-style tuple caches never had this problem (an empty cache was
        # just `None` at the top level, checked with `is not None`); the
        # new object-based cache is non-None even when empty, so that check
        # no longer catches this case. A zero-length placeholder tensor
        # makes `.shape[2]` read as 0, matching what the old cache
        # semantics actually meant by "nothing cached yet".
        if layer_idx >= len(cache.layers):
            return (_torch.empty(0, 0, 0, 0), _torch.empty(0, 0, 0, 0))
        layer = cache.layers[layer_idx]
        keys = layer.keys if layer.keys is not None else _torch.empty(0, 0, 0, 0)
        values = layer.values if layer.values is not None else _torch.empty(0, 0, 0, 0)
        return (keys, values)

    def _encoder_decoder_cache_getitem(self, layer_idx: int):
        self_k, self_v = _sub_cache_layer_kv(self.self_attention_cache, layer_idx)
        cross_k, cross_v = _sub_cache_layer_kv(self.cross_attention_cache, layer_idx)
        return (self_k, self_v, cross_k, cross_v)

    _EDC.__getitem__ = _encoder_decoder_cache_getitem

# Shim 4d/4 — even with 4c's placeholder, generation still crashed:
# `RuntimeError: Sizes of tensors must match except in dimension 2. Expected
# size 0 but got size 5...` inside the remote attention forward()'s
# `torch.cat([past_key_value[0], key_states], dim=2)`. Read the cached
# `modeling_indictrans.py` directly to find the real cause — not a shape bug
# in the placeholder (its dim 2 being 0 was already correct), but stale
# *branch logic*: `elif past_key_value is not None:` (three occurrences —
# encoder self-attn, decoder self-attn, an alternate attention backend
# variant) used to mean "is there real cached data to concatenate with",
# back when the old tuple-cache was literally `None` until the first token
# was generated. Under the new Cache-object API a real (non-None) cache is
# passed in from the very first forward pass, before anything is cached, so
# that `is not None` check no longer distinguishes "empty cache" from
# "populated cache" — it always takes the "reuse and concatenate" branch,
# even when there's a zero-length placeholder (from 4c) with nothing real
# to concatenate. Confirmed live that the *cross*-attention equivalent check
# a few lines above doesn't have this bug: it already compares
# `past_key_value[0].shape[2] == key_value_states.shape[1]`, which
# correctly evaluates false for an empty (shape[2]==0) cache — only the
# self-attention branch is missing that same guard. Fixed by adding it,
# patched directly into the cached file's source (post-download, pre-import
# — get_class_in_module is what actually exec()s these files) rather than
# from outside: the bug is in the branch *condition* itself, which can't be
# changed by wrapping/patching methods after the fact the way shims 1-4c
# could. No-ops safely for every other cached remote-code file (the child
# text has to already be present to be replaced) and is naturally
# idempotent (replacing already-patched text finds nothing left to match).
import transformers.dynamic_module_utils as _dmu

_original_get_class_in_module = _dmu.get_class_in_module
_BROKEN_REUSE_CHECK = "elif past_key_value is not None:"
_FIXED_REUSE_CHECK = "elif past_key_value is not None and past_key_value[0].shape[2] > 0:"


def _patch_stale_kv_cache_check(module_file) -> None:
    try:
        text = module_file.read_text()
    except OSError:
        return
    if _BROKEN_REUSE_CHECK in text:
        module_file.write_text(text.replace(_BROKEN_REUSE_CHECK, _FIXED_REUSE_CHECK))


def _get_class_in_module_with_kv_cache_patch(class_name, module_path, **kwargs):
    from pathlib import Path

    _patch_stale_kv_cache_check(Path(_dmu.HF_MODULES_CACHE) / module_path)
    return _original_get_class_in_module(class_name, module_path, **kwargs)


_dmu.get_class_in_module = _get_class_in_module_with_kv_cache_patch

# Shim 4e/4 — with 4d's branch-logic fix, generation runs without crashing
# but returns instantly (`[2, 2]`, immediate EOS — empty output for every
# input, no error). Traced the real cause by reading the decoder's own
# forward() loop directly: this remote model builds its *own* hand-written
# legacy nested-tuple KV-cache internally (`next_decoder_cache += (...)`,
# one (self_k, self_v, cross_k, cross_v) tuple per layer) and returns it as
# `outputs.past_key_values` — but never calls `.update()` on the real
# EncoderDecoderCache object generate() handed it. So nothing this model
# computes during decoding is ever visible to the *next* decoding step:
# every step effectively starts over with an empty cache and no memory of
# prior tokens, which is exactly what produces immediate-EOS degenerate
# output rather than a crash. transformers itself used to ship
# `from_legacy_cache`/`to_legacy_cache` conversion helpers for exactly this
# legacy-format gap; both are fully removed in the installed version, so
# this reimplements the specific direction needed (legacy tuple -> write
# into a real Cache object) rather than a general bidirectional converter.
#
# Hooked onto the *same* `PreTrainedModel.__init_subclass__` chain as Shim
# 4a (chaining through its already-patched state, not undoing it) — wraps
# any dynamically-loaded model class's own `forward()` to, after each real
# call, check whether it returned a legacy plain-tuple cache while it was
# actually handed a real Cache object to keep updated; if so, write the
# legacy tuple's tensors into that Cache object's layers directly (replacing
# each layer's stored tensors wholesale, not appending — this model's own
# tensors are already the full accumulated sequence via its internal
# torch.cat, not an incremental delta) and substitute the real, now-updated
# Cache object back into the output in place of the legacy tuple. Self-limiting
# by construction: a model that already returns a proper Cache object (i.e.
# every other, unaffected model in this app) hits the `isinstance` guard and
# this becomes a no-op passthrough.
from transformers.cache_utils import Cache as _Cache


def _write_legacy_cache_into_encoder_decoder_cache(legacy_cache, target: _EDC) -> None:
    for layer_idx, layer in enumerate(legacy_cache):
        _replace_dynamic_cache_layer(target.self_attention_cache, layer_idx, layer[0], layer[1])
        if len(layer) > 2 and layer[2] is not None:
            _replace_dynamic_cache_layer(target.cross_attention_cache, layer_idx, layer[2], layer[3])


def _replace_dynamic_cache_layer(cache, layer_idx: int, keys, values) -> None:
    while len(cache.layers) <= layer_idx:
        cache.layers.append(cache.layer_class_to_replicate())
    cache.layers[layer_idx].keys = keys
    cache.layers[layer_idx].values = values


# .__func__ again — see Shim 4a's comment on why the bound-classmethod
# version can't be chained through correctly.
_previous_init_subclass = _PTM.__init_subclass__.__func__  # already Shim 4a's patched version — chained, not replaced


def _cache_writeback_compat_init_subclass(cls, **kwargs) -> None:
    _previous_init_subclass(cls, **kwargs)
    forward = cls.__dict__.get("forward")
    if forward is not None:

        @_functools.wraps(forward)  # generate()'s own kwarg validation inspects forward's real
        # signature (inspect.signature follows __wrapped__, which @wraps sets) — a bare
        # `(self, *args, **kwargs)` wrapper hides the real parameter names transformers
        # needs to see there, and generate() rejects every call before this even runs.
        def _forward_with_cache_writeback(self, *args, **fwd_kwargs):
            past_key_values = fwd_kwargs.get("past_key_values")
            outputs = forward(self, *args, **fwd_kwargs)
            returned_cache = getattr(outputs, "past_key_values", None)
            if isinstance(past_key_values, _EDC) and returned_cache is not None and not isinstance(returned_cache, _Cache):
                try:
                    _write_legacy_cache_into_encoder_decoder_cache(returned_cache, past_key_values)
                    outputs.past_key_values = past_key_values
                except (AttributeError, IndexError, TypeError):
                    pass  # leave outputs exactly as the model returned them if the legacy shape doesn't match what we expect
            return outputs

        cls.forward = _forward_with_cache_writeback

    # Same idea, one level further: beam search calls the model's own
    # `_reorder_cache(past_key_values, beam_idx)` to keep cached K/V aligned
    # with which beams survived each step — another legacy-tuple-shaped
    # method (`for past_state in layer_past: past_state.index_select(...)`)
    # that crashes on a real Cache object (some of its per-layer entries,
    # e.g. an unused sliding-window slot, are legitimately None, which the
    # legacy iteration has no concept of skipping). Real Cache objects
    # already carry their own correct, built-in `.reorder_cache()` — no
    # need to reimplement it here, just route to it instead of the model's
    # incompatible override when that's what we're holding.
    reorder_cache = cls.__dict__.get("_reorder_cache")
    if reorder_cache is not None:

        @_functools.wraps(reorder_cache)
        def _reorder_cache_with_bridge(self, past_key_values, beam_idx):
            if isinstance(past_key_values, _Cache):
                past_key_values.reorder_cache(beam_idx)
                return past_key_values
            return reorder_cache(self, past_key_values, beam_idx)

        cls._reorder_cache = _reorder_cache_with_bridge


_PTM.__init_subclass__ = classmethod(_cache_writeback_compat_init_subclass)

import torch
from IndicTransToolkit import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.config import get_active_candidate
from app.hf_cache import offline_kwargs

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


def _repopulate_sinusoidal_position_buffers(model) -> None:
    # Shim 4f/4 — even with generation running end to end (no crash), real
    # output was still degenerate: the encoder's own hidden states were NaN
    # from the very first layer, before any of the attention/cache machinery
    # above even runs. Traced directly: this remote model's positional
    # embedding class (`IndicTransSinusoidalPositionalEmbedding`) computes a
    # real sinusoidal table once in `__init__` and stores it as a
    # *non-persistent* buffer (`persistent=False` — deliberately excluded
    # from the checkpoint, since it's pure math, not learned). But
    # transformers 5.x's `from_pretrained` initializes the whole model on
    # the meta device by default (confirmed by reading
    # `_move_missing_keys_from_meta_to_device` in `modeling_utils.py`), and
    # explicitly re-materializes any non-persistent buffer afterwards via
    # `torch.empty_like(buffer, device=...)` — genuinely uninitialized
    # memory, not a recomputation — because it has no checkpoint data to
    # restore and no way to know the buffer was supposed to be
    # deterministically computed rather than loaded. Confirmed empirically:
    # the buffer's values were tiny denormalized garbage with scattered NaN,
    # exactly the signature of uninitialized memory, not a real computation
    # gone wrong. Older transformers versions didn't default to meta-device
    # init for a non-quantized/non-distributed load, so this custom model's
    # `__init__`-time computation used to just work. Fixed by forcing every
    # such buffer to recompute for real immediately after loading — using
    # the model's own `make_weights()` method, so it's the exact same math
    # the model already trusts, not a reimplementation.
    for module in model.modules():
        if type(module).__name__ == "IndicTransSinusoidalPositionalEmbedding":
            module.make_weights(module.weights.size(0), module.embedding_dim, module.padding_idx)


def _get_model(task_type: str):
    """task_type is "translation_en_indic" or "translation_indic_en" —
    resolved via the model registry (app.config) exactly like every other
    task type, so switching the active candidate works the same way here
    too, even though loading itself goes through transformers, not Ollama."""
    with _lock:
        if task_type not in _loaded:
            candidate = get_active_candidate(task_type)
            # Phase 12: offline once cached — see app/hf_cache.py.
            offline = offline_kwargs(candidate.model_id)
            tokenizer = AutoTokenizer.from_pretrained(candidate.model_id, trust_remote_code=True, **offline)
            model = AutoModelForSeq2SeqLM.from_pretrained(candidate.model_id, trust_remote_code=True, **offline)
            model.eval()
            _repopulate_sinusoidal_position_buffers(model)
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
