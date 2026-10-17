"""Construction and lifecycle boundary for the semantic decoder."""
import torch
from GPT_SoVITS.AR.models.t2s_model import Text2SemanticDecoder


class TorchGpt(torch.nn.Module):
    kind = "pytorch"

    def __init__(self, checkpoint, backend):
        super().__init__()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.config = state['config']
        self.model = Text2SemanticDecoder(self.config, top_k=3, inference_only=True)
        self.load_state_dict(state['weight'])
        self.backend = backend
        if backend.is_half:
            self.half()
        self.wake()
        self.eval()

    def wake(self):
        self.to(self.backend.torch_device)
        self.model.rebind_inference_blocks()

    def hibernate(self):
        self.cpu()
        self.model.rebind_inference_blocks()

    def close(self):
        self.hibernate()


def load_gpt(checkpoint, backend):
    if backend.is_directml:
        from ort_gpt import OrtGpt
        return OrtGpt(checkpoint, backend.label, backend.torch_device)
    return TorchGpt(checkpoint, backend)
