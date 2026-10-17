"""Compare the trimmed BERT with upstream features, or time first streamed speech."""
import argparse
import json
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--parity', action='store_true')
parser.add_argument('--language', choices=['zh','ja','en'], default='zh')
parser.add_argument('--device', choices=['cpu','cuda','dml'], default='cuda',
                    help='Device for the real-model streaming check.')
parser.add_argument('--warmup', action='store_true', help='Warm up Japanese synthesis before measuring.')
args = parser.parse_args()
args.root = args.root.resolve()
sys.path.insert(0, str(args.root))
import torch
torch.set_num_threads(4)
import chinese_frontend as cf

if args.parity:
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    baseline_tokenizer = AutoTokenizer.from_pretrained(cf.MODEL_DIR, local_files_only=True)
    baseline = AutoModelForMaskedLM.from_pretrained(cf.MODEL_DIR, local_files_only=True, torch_dtype=torch.float32, attn_implementation='eager').eval()
    report = {'rows': [], 'torch': torch.__version__}
    for text in ['早饭想吃什么？', '你好，今天也请多关照。', '重庆银行重新开门了。', '现在是下午三点二十五分。']:
        inputs = baseline_tokenizer(text, return_tensors='pt')
        with torch.inference_mode():
            hidden = baseline(**inputs, output_hidden_states=True).hidden_states[-3][0,1:-1]
            word2ph = [1+i%3 for i in range(len(text))]
            expected = torch.cat([hidden[i].repeat(n,1) for i,n in enumerate(word2ph)]).T.contiguous()
            actual = cf.get_bert_feature(text, word2ph)
        assert baseline_tokenizer(text) == cf._tokenizer(text)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        report['rows'].append({'text':text,'max_abs':float((actual-expected).abs().max())})
    report['parameters_before'] = sum(p.numel() for p in baseline.parameters())
    report['parameters_after'] = sum(p.numel() for p in cf._model.parameters())
else:
    from run_inference import GPTSoVITSInference
    from streaming_latents import LatentStream
    config = json.loads((args.root/'resources/voices.json').read_text('utf-8'))['tyc']
    engine = GPTSoVITSInference(str(args.root/'resources'/config['gpt_path']),str(args.root/'resources'/config['sovits_path']),str(args.root/'resources'/config['references']['default']['cache_path']),device=args.device,cpu_threads=4,pipeline=False)
    report = {'root':str(args.root), 'language':args.language, 'rows':[]}
    try:
        prompts = {'zh':['你好，今天也请多关照。','早饭想吃什么？我打算煮一碗面，再加一个鸡蛋。'], 'ja':['こんにちは。今日もよろしくお願いします。','今日はいい天気ですね。一緒に散歩しませんか。'], 'en':['Hello, it is nice to see you today.','What would you like for breakfast? I can make some noodles.']}
        # Explicitly choose warmup; directory location does not change the test.
        if args.warmup:
            engine.synthesize('こんにちは。',text_lang='ja')
            texts = prompts[args.language][1:]
        else:
            texts = prompts[args.language]
        for text in texts:
            torch.manual_seed(42)
            started = time.perf_counter(); first = None; count = 0
            for event in LatentStream(engine,seed=42).stream(text,language=args.language):
                if event['type']=='chunk':
                    count += len(event['pcm'])
                    if first is None: first=time.perf_counter()-started
            report['rows'].append(dict(text=text,total=time.perf_counter()-started,first=first,samples=count))
    finally:
        engine.close()
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
