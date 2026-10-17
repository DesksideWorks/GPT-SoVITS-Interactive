"""GPT on DirectML, with optional semantic prefixes and GPU-resident KV."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dxgi_adapters import enumerate_adapters, match_adapter
from GPT_SoVITS.AR.models.utils import sample


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OrtGpt(torch.nn.Module):
    kind = "ort-dml"
    max_segment_phones = 180

    def __init__(self, checkpoint, adapter_name, torch_device):
        super().__init__()
        import onnxruntime as ort
        if "DmlExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("DirectML GPT requires onnxruntime-directml in this Python environment")
        self.ort = ort
        self.directory = Path(checkpoint).with_suffix(".ort-dml")
        manifest = json.loads((self.directory / "manifest.json").read_text("utf-8"))
        if manifest.get("schema") != 1 or manifest['gpt_sha256'] != sha256(checkpoint):
            raise RuntimeError("GPT ONNX assets do not match the selected checkpoint; run tools/export_gpt_dml.py")
        for name in ("prefill-p256.onnx", "step.onnx"):
            if sha256(self.directory / name) != manifest['files'][name]:
                raise RuntimeError(f"GPT ONNX asset checksum mismatch: {name}")
        self.config = manifest['config']
        self.capacity = int(manifest['capacity'])
        self.prefill_size = int(manifest['prefill_size'])
        if (self.capacity, self.prefill_size) != (512, 256):
            raise RuntimeError("Unsupported GPT ONNX asset dimensions")
        self.adapter = match_adapter(adapter_name, torch_device=torch_device)
        self.pre = self.step = None
        self.wake()

    @property
    def model(self):
        return self

    def rebind_inference_blocks(self):
        pass

    def wake(self):
        if self.pre is not None and self.step is not None:
            return
        matches = [a for a in enumerate_adapters() if a['luid'] == self.adapter['luid'] and a['d3d12']]
        if len(matches) != 1:
            raise RuntimeError("The selected GPT DirectML adapter is no longer available")
        self.adapter = matches[0]
        options = self.ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = self.ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 1
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        providers = [("DmlExecutionProvider", {'device_id': str(self.adapter['index'])})]
        try:
            self.pre = self.ort.InferenceSession(str(self.directory / "prefill-p256.onnx"), sess_options=options, providers=providers)
            self.step = self.ort.InferenceSession(str(self.directory / "step.onnx"), sess_options=options, providers=providers)
            for session in (self.pre, self.step):
                session.disable_fallback()
                if session.get_providers()[0] != "DmlExecutionProvider":
                    raise RuntimeError("GPT did not initialize on DirectML")
        except Exception:
            self.close()
            raise
        print(f"[gpt_backend] ORT DirectML adapter={self.adapter}")

    def hibernate(self):
        self.close()

    def close(self):
        # No bindings/OrtValues survive a generation, so dropping both sessions
        # releases their GPU allocations without retaining a second GPT model.
        self.pre = self.step = None

    def segment_phone_budget(self, prefix_phones=0, prefix_tokens=0):
        if not prefix_tokens:
            return self.max_segment_phones
        # Reserve three generated semantic tokens per target phoneme plus
        # 32 slots for pauses/EOS. Enforce the hard KV bound again at decoding.
        budget = min(self.max_segment_phones, self.prefill_size-prefix_phones,
                     (self.capacity-prefix_phones-prefix_tokens-32)//4)
        if budget < 1:
            raise ValueError('Semantic prefix leaves no usable GPT target capacity')
        return budget

    def _tokens(self, x, bert, top_k=15, top_p=1., temperature=1., repetition_penalty=1.35, early_stop_num=1500, prompts=None, **kwargs):
        if self.pre is None or self.step is None:
            raise RuntimeError("GPT sessions are hibernated")
        phones = x.detach().cpu().numpy().astype(np.int64, copy=False)
        n = phones.shape[1]
        if phones.shape[0] != 1 or not 0 < n <= self.prefill_size:
            raise ValueError(f"GPT expects one segment of 1–{self.prefill_size} phonemes, got {phones.shape}")
        features = bert.detach().cpu().float().numpy()
        if features.shape != (1, 1024, n):
            raise ValueError('GPT BERT and phoneme dimensions differ')
        history = []
        if prompts is not None:
            if (prompts.ndim != 2 or prompts.shape[0] != 1 or prompts.dtype != torch.long or
                    prompts.numel() == 0 or not bool(((prompts >= 0) & (prompts < 1024)).all())):
                raise ValueError('Invalid GPT semantic prefix')
            history = prompts[0].detach().cpu().tolist()
        prefix_length = len(history)
        if n + prefix_length + 12 >= self.capacity:
            raise ValueError('GPT prefix and text leave insufficient generation capacity')
        inputs = dict(phones=np.pad(phones, ((0, 0), (0, self.prefill_size-n))),
                      bert=np.pad(features, ((0, 0), (0, 0), (0, self.prefill_size-n))),
                      attention_bias=np.where(np.arange(self.prefill_size) >= n, -10000., 0.).astype('float32').reshape(1, 1, 1, -1),
                      last_position=np.asarray([n-1], dtype=np.int64))
        def run(session, values, caches=None):
            binding = session.io_binding()
            for name, array in values.items():
                binding.bind_cpu_input(name, array)
            if caches is not None:
                binding.bind_ortvalue_input('cache_k', caches[0])
                binding.bind_ortvalue_input('cache_v', caches[1])
            binding.bind_output('logits', 'cpu')
            # ORT's DML allocator uses logical device 0 even on physical adapter N.
            binding.bind_output('keys', 'dml', 0)
            binding.bind_output('values', 'dml', 0)
            session.run_with_iobinding(binding)
            outputs = binding.get_outputs()
            return outputs[0].numpy(), (outputs[1], outputs[2])
        logits, caches = run(self.pre, inputs)
        def decode(token, audio_position, caches):
            position = n + audio_position
            if position >= self.capacity:
                raise RuntimeError('GPT reached its generation limit before EOS; shorten the text segment')
            inputs = dict(token=np.asarray([[token]], dtype=np.int64), position=np.asarray([position], dtype=np.int64),
                          audio_position=np.asarray([audio_position], dtype=np.int64),
                          attention_bias=np.where(np.arange(self.capacity) > position, -10000., 0.).astype('float32').reshape(1, 1, 1, -1))
            return run(self.step, inputs, caches)

        # Known audio tokens are teacher-forced after the full text prefill.
        # They condition the target but are never emitted by this iterator.
        for audio_position, token in enumerate(history):
            logits, caches = decode(token, audio_position, caches)
        limit = min(1500, int(early_stop_num)+1) if early_stop_num > 0 else 1500
        for index in range(limit):
            candidate = torch.from_numpy(logits[:, :-1] if index < 11 else logits)
            prior = torch.tensor([history], dtype=torch.int) if history else None
            token = int(sample(candidate, prior, top_k=top_k, top_p=top_p, temperature=temperature,
                               repetition_penalty=repetition_penalty)[0][0, 0])
            if token == 1024 or int(candidate.argmax(-1)[0]) == 1024:
                if index == 0:
                    raise RuntimeError("GPT generated empty speech")
                return
            history.append(token)
            yield token
            if n+prefix_length+index >= self.capacity or index == limit-1:
                raise RuntimeError("GPT reached its generation limit before EOS; shorten the text segment")
            logits, caches = decode(token, prefix_length+index, caches)

    def infer_panel(self, x, x_lens, prompts, bert_feature, **kwargs):
        generated = torch.tensor([list(self._tokens(x, bert_feature, prompts=prompts, **kwargs))], dtype=torch.long)
        if prompts is not None:
            return torch.cat((prompts.detach().cpu(), generated), dim=1), generated.shape[-1]
        return generated, 0

    def infer_panel_naive(self, x, x_lens, prompts, bert_feature, streaming_mode=False, chunk_length=24, **kwargs):
        if not streaming_mode:
            yield self.infer_panel(x, x_lens, prompts, bert_feature, **kwargs)
            return
        pending = []
        for token in self._tokens(x, bert_feature, prompts=prompts, **kwargs):
            pending.append(token)
            if len(pending) >= chunk_length:
                yield torch.tensor([pending], dtype=torch.long), False
                pending = []
        yield torch.tensor([pending], dtype=torch.long) if pending else None, True

    def warmup(self):
        # Warm both graphs without requiring a complete sampled utterance.
        stream = self._tokens(torch.tensor([[1, 2, 3]]), torch.zeros(1, 1024, 3))
        try:
            next(stream)
            next(stream)
        finally:
            stream.close()
