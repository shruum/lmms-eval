#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/anna2/shruthi/lmms-eval/my_analysis')
from transformers import LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained(
    'llava-hf/llava-1.5-7b-hf', torch_dtype='auto', device_map='cpu'
)
lm = model.language_model
# LLaVA uses LlamaForCausalLM which has .layers directly on the model
if hasattr(lm, 'model'):
    layer = lm.model.layers[0]
else:
    layer = lm.layers[0]
print(f'Attention type: {type(layer.self_attn)}')
print(f'Module name: {layer.self_attn.__class__.__name__}')
print(f'Has num_heads attr: {hasattr(layer.self_attn, "num_heads")}')
print(f'Has q_proj attr: {hasattr(layer.self_attn, "q_proj")}')
print(f'Forward signature: {layer.self_attn.forward.__code__.co_varnames[:10]}')
