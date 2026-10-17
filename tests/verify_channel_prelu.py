"""Check real device PReLU output against PyTorch CPU, including trained negative slopes."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from GPT_SoVITS.module.modules import ChannelPReLU

parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['cpu', 'cuda', 'dml'], required=True)
args = parser.parse_args()
if args.device == 'dml':
    import torch_directml
    device = torch_directml.device()
else:
    device = torch.device(args.device)

torch.manual_seed(42)
cases = 0
for channels in (1, 4, 1024):
    reference = torch.nn.PReLU(channels)
    reference.weight.data.copy_(torch.linspace(-0.2, 0.9, channels))
    layer = ChannelPReLU(channels).to(device)
    layer.load_state_dict(reference.state_dict(), strict=True)
    for shape in ((2, channels, 1), (2, channels, 7), (2, channels)):
        x = torch.randn(shape) * 100
        x.flatten()[0] = 0
        with torch.no_grad():
            actual = layer(x.to(device)).cpu()
            expected = reference(x)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        cases += 1
print(f'PASSED: {args.device}, {cases} cases, checkpoint-compatible weights and per-channel output')
