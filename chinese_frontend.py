"""CPU Chinese BERT features, adapted from minimal-inference get_bert_feature.

Shared across character engine switches. Reference-side SSL/SV remain absent;
the text model is loaded only when Chinese is actually synthesized.
"""
from functools import lru_cache
from pathlib import Path
import threading

import torch

MODEL_DIR=Path(__file__).resolve().parent/'resources/models/chinese-roberta-wwm-ext-large'
_lock=threading.RLock()
_tokenizer=None
_model=None


@lru_cache(maxsize=64)
def _features(text, word2ph):
    global _tokenizer,_model
    from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
    from transformers.models.bert.configuration_bert import BertConfig
    from transformers.models.bert.modeling_bert import BertModel
    with _lock:
        if _model is None:
            tokenizer=BertTokenizerFast.from_pretrained(MODEL_DIR,local_files_only=True)
            config=BertConfig.from_pretrained(MODEL_DIR,local_files_only=True)
            # hidden_states[-3] is the output before the final two encoder layers.
            # Later layers, the pooler and the masked-LM head cannot affect it.
            config.num_hidden_layers -= 2
            model=BertModel.from_pretrained(
                MODEL_DIR,config=config,add_pooling_layer=False,local_files_only=True,
                torch_dtype=torch.float32,attn_implementation='eager').cpu().eval()
            _tokenizer,_model=tokenizer,model
        if len(text)!=len(word2ph) or not text:
            raise ValueError('Chinese character/phone alignment is empty or inconsistent.')
        inputs=_tokenizer(text,return_tensors='pt')
        if inputs['input_ids'].shape[1]>_model.config.max_position_embeddings:
            raise ValueError('Chinese TTS segment exceeds the BERT context; split the text into shorter sentences.')
        with torch.inference_mode():
            hidden=_model(**inputs).last_hidden_state[0,1:-1]
            if hidden.shape[0]!=len(word2ph):
                raise ValueError('Chinese tokenizer must produce one token per normalized character.')
            return torch.cat([hidden[i].repeat(n,1) for i,n in enumerate(word2ph)]).T.contiguous()


def get_bert_feature(text,word2ph):
    return _features(text,tuple(word2ph))
