# VAF (Visual Amplification Fusion) - ClearSight Implementation

**Repository:** https://github.com/ustc-hyin/ClearSight
**Paper:** CVPR 2025
**Local Path:** `/home/anna2/shruthi/ClearSight/`

---

## 🎯 **VAF Method Overview**

### **How VAF Works**
1. **Training-free** - no model weights modification
2. **Attention amplification** - multiplicative scaling on attention logits
3. **Target layers** - middle fusion layers (9-14) where modality fusion occurs
4. **Enhancement** - boost attention to visual tokens
5. **Suppression** - reduce attention to system tokens

---

## 🔬 **Key Implementation Details**

### **VAF Parameters (from ClearSight code)**

```python
# From visaug/inference/infer_pope.py (lines 53-58)
for i, layer in enumerate(model.model.layers):
    if i > 8 and i < 15:  # Layers 9-14
        attn_adap = AttnAdapter(layer.self_attn.config, enh_para=1.15, sup_para=0.95)
        attn_adap.load_state_dict(layer.self_attn.state_dict())
        attn_adap = attn_adap.half().cuda()
        layer.self_attn = attn_adap
```

**Parameters:**
- `enh_para=1.15` - Enhancement parameter (boost visual tokens by 15%)
- `sup_para=0.95` - Suppression parameter (reduce system tokens by 5%)
- Layers: 9-14 (middle fusion layers)
- System tokens: 35 (SYS_LEN in AttnAdapter.py line 71)
- Image tokens: 576 (IMG_LEN in AttnAdapter.py line 72)

### **Attention Scaling Formula** (from AttnAdapter.py lines 70-79)

```python
# During generation phase (q_len > SYS_LEN + IMG_LEN)
attn_weights[:, :, SYS_LEN+IMG_LEN:, SYS_LEN:SYS_LEN+IMG_LEN] = self.enh_para * attn_weights[:, :, SYS_LEN+IMG_LEN:, SYS_LEN:SYS_LEN+IMG_LEN]
attn_weights[:, :, SYS_LEN+IMG_LEN:, :SYS_LEN] = self.sup_para * attn_weights[:, :, SYS_LEN+IMG_LEN:, :SYS_LEN]

# During prefill phase (q_len <= SYS_LEN + IMG_LEN)
attn_weights[:, :, :, SYS_LEN:SYS_LEN+IMG_LEN] = self.enh_para * attn_weights[:, :, :, SYS_LEN:SYS_LEN+IMG_LEN]
attn_weights[:, :, :, :SYS_LEN] = self.sup_para * attn_weights[:, :, :, :SYS_LEN]
```

---

## 📊 **Paper Results (POPE)**

| Method   | LLaVA-7B | LLaVA-13B | Qwen-7B |
|----------|----------|-----------|---------|
| Regular  | 87.8%    | 87.6%     | 88.2%   |
| VCD      | 88.4%    | 88.9%     | 89.1%   |
| VAF      | **89.7%** | **90.1%** | **90.0%** |

**VAF Improvement:** +1.9% over Regular, +1.3% over VCD

---

## 🔧 **Implementation Differences**

### **What We Thought vs. Actual ClearSight**

| Parameter | Our Understanding | ClearSight Actual |
|-----------|-------------------|-------------------|
| Enhancement α | 0.15 | 0.15 (✓ correct) |
| Suppression β | 0.1 | 0.05 (✗ incorrect) |
| enh_para | 1.15 | 1.15 (✓ correct) |
| sup_para | 0.9 | 0.95 (✗ incorrect) |
| Layer start | 10 | 9 (✗ incorrect) |
| Layer end | 15 | 14 (✗ incorrect) |

---

## 🚀 **Running VAF on POPE**

### **Official ClearSight Method**

```bash
cd /home/anna2/shruthi/ClearSight/LLaVA

# Setup environment (first time only)
conda create -yn clearsight python=3.10
conda activate clearsight
pip install -e .

# Run on one split
python ./visaug/inference/infer_pope.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./data/pope/coco/coco_pope_random.json \
    --image-folder ./data/pope/coco/val2014 \
    --answers-file ./outputs/inference/res_coco_random.jsonl \
    --use-visaug \
    --enh-para 1.15 \
    --sup-para 0.95

# Evaluate
python ./visaug/inference/eval_pope.py \
    --annotation-file ./data/pope/coco/coco_pope_random.json \
    --result-file ./outputs/inference/res_coco_random.jsonl
```

---

## 📁 **Key Files**

### **ClearSight Implementation**
- `visaug/inference/AttnAdapter.py` - VAF attention adapter
- `visaug/inference/infer_pope.py` - POPE inference script
- `visaug/inference/eval_pope.py` - POPE evaluation script

### **Data Requirements**
- POPE data: `./data/pope/{coco,aokvqa,gqa}/{dataset}_pope_{random,popular,adversarial}.json`
- Images:
  - COCO/A-OKVQA: `./data/pope/coco/val2014`
  - GQA: `./data/pope/gqa/images`

---

## 💡 **Key Insights**

1. **VAF is simpler than SRF** - no CLIP guidance needed
2. **VAF is more efficient than VCD** - no contrastive samples needed
3. **VAF targets middle layers** - where modality fusion occurs
4. **VAF preserves content quality** - maintains coherence while reducing hallucinations
5. **Our previous understanding was slightly off** - sup_para is 0.95, not 0.9

---

## 🎯 **Next Steps**

1. ✅ **Clone ClearSight repository** - done
2. ⏳ **Setup ClearSight environment**
3. ⏳ **Run VAF on all 9 POPE configurations**
4. ⏳ **Compare with our baselines**
5. ⏳ **Understand why VAF works better than SRF**

---

*For detailed code analysis: see AttnAdapter.py lines 70-79*
*For inference details: see infer_pope.py lines 53-58*
