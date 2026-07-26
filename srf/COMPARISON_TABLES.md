# Baseline Numbers from VAF and AIR Papers

## For Replication: Add SRF rows to these tables

---

### Table 1: POPE Results (Averaged across MSCOCO, A-OKVQA, GQA)

**From VAF Paper (Table 2)** - LLaVA-v1.5 models

| Method | LLaVA-7B Random | LLaVA-7B Popular | LLaVA-7B Adv | LLaVA-13B Random | LLaVA-13B Popular | LLaVA-13B Adv |
|--------|-----------------|------------------|--------------|------------------|------------------|--------------|
| Regular | 87.8 / 87.5 | 82.5 / 83.2 | 77.6 / 79.4 | 87.6 / 87.4 | 82.7 / 84.1 | 77.8 / 79.5 |
| VCD | 88.4 / 87.7 | 83.1 / 84.1 | 78.1 / 79.6 | 88.9 / 87.8 | 83.7 / 85.1 | 78.2 / 79.7 |
| ICD | 88.1 / 87.6 | 82.1 / 82.9 | 78.5 / 79.9 | 88.1 / 87.6 | 82.9 / 84.3 | 79.1 / 80.1 |
| **VAF** | **89.6 / 89.3** | **84.5 / 84.9** | **80.1 / 81.0** | **90.1 / 89.9** | **85.2 / 86.4** | **80.7 / 81.7** |
| **SRF (ours)** | ? | ? | ? | ? | ? | ? |

**From VAF Paper** - Qwen-VL-7B

| Method | Qwen-7B Random | Qwen-7B Popular | Qwen-7B Adv |
|--------|----------------|-----------------|-------------|
| Regular | 88.2 / 87.9 | 82.4 / 83.1 | 77.2 / 78.9 |
| VCD | 89.1 / 88.4 | 83.0 / 84.1 | 78.8 / 80.1 |
| ICD | 88.9 / 88.1 | 83.2 / 84.5 | 78.1 / 79.2 |
| **VAF** | **90.0 / 89.7** | **84.9 / 85.1** | **80.4 / 81.2** |
| **SRF (ours)** | ? | ? | ? |

**From AIR Paper (Table 2)** - LLaVA-1.5-7B on MSCOCO only

| Method | Random Acc/F1 | Popular Acc/F1 | Adv Acc/F1 |
|--------|---------------|----------------|------------|
| Vanilla | 83.7 / 83.0 | 78.2 / 78.4 | 75.0 / 76.0 |
| VCD | 85.4 / 83.7 | 84.3 / 83.0 | 81.8 / 80.9 |
| MemVR | 87.6 / 86.2 | 86.0 / 84.7 | 83.5 / 82.5 |
| VAF | 87.6 / 86.2 | 86.2 / 85.0 | 83.9 / 82.8 |
| **AIR** | **89.0 / 88.2** | **87.1 / 86.4** | **83.9 / 83.6** |
| **SRF (ours)** | ? | ? | ? |

---

### Table 2: MME Results

**From VAF Paper (Table 3)**

| Model | Method | Existence | Count | Position | Color | Total |
|-------|--------|-----------|-------|----------|-------|-------|
| LLaVA-7B | Regular | 185.0 | 146.7 | 128.3 | 150.0 | 610.0 |
| LLaVA-7B | VCD | 185.0 | 141.3 | 128.3 | 153.0 | 607.7 |
| LLaVA-7B | ICD | 185.0 | 148.3 | 126.7 | 148.3 | 608.3 |
| LLaVA-7B | **VAF** | **195.0** | **158.3** | 128.3 | 155.0 | **636.7** |
| LLaVA-7B | **SRF** | ? | ? | ? | ? | ? |

| Model | Method | Existence | Count | Position | Color | Total |
|-------|--------|-----------|-------|----------|-------|-------|
| LLaVA-13B | Regular | 185.0 | 155.0 | 133.3 | 165.0 | 638.3 |
| LLaVA-13B | VCD | 185.0 | 155.0 | 130.0 | 168.3 | 638.3 |
| LLaVA-13B | ICD | 183.3 | 153.3 | 131.7 | 165.0 | 633.3 |
| LLaVA-13B | **VAF** | **195.0** | **160.0** | 136.7 | 170.0 | **661.7** |
| LLaVA-13B | **SRF** | ? | ? | ? | ? | ? |

| Model | Method | Existence | Count | Position | Color | Total |
|-------|--------|-----------|-------|----------|-------|-------|
| Qwen-7B | Regular | 158.3 | 150.0 | 128.3 | 170.0 | 606.7 |
| Qwen-7B | VCD | 158.3 | 150.0 | 133.3 | 175.0 | 616.7 |
| Qwen-7B | ICD | 128.3 | 151.7 | 128.3 | 170.0 | 578.3 |
| Qwen-7B | **VAF** | **165.0** | **155.0** | 133.3 | 175.0 | **628.3** |
| Qwen-7B | **SRF** | ? | ? | ? | ? | ? |

---

### Table 3: CHAIR Results (Lower is better)

**From AIR Paper (Table 1)**

| Model | Method | CHAIR S ↓ | CHAIR I ↓ | BLEU ↑ |
|-------|--------|-----------|-----------|--------|
| LLaVA-1.5-7B | Vanilla | 22.0 | 6.7 | 14.5 |
| LLaVA-1.5-7B | VCD | 24.6 | 7.3 | 13.9 |
| LLaVA-1.5-7B | MemVR | 21.6 | 6.4 | 14.4 |
| LLaVA-1.5-7B | VAF | 20.4 | 6.5 | 14.6 |
| LLaVA-1.5-7B | **AIR** | **18.4** | **5.7** | 14.4 |
| LLaVA-1.5-7B | **SRF** | ? | ? | ? |

| Model | Method | CHAIR S ↓ | CHAIR I ↓ | BLEU ↑ |
|-------|--------|-----------|-----------|--------|
| Qwen-VL-Chat | Vanilla | 20.0 | 6.2 | 13.5 |
| Qwen-VL-Chat | VCD | 19.2 | 5.7 | 13.4 |
| Qwen-VL-Chat | MemVR | 20.0 | 6.1 | 13.3 |
| Qwen-VL-Chat | VAF | 20.6 | 6.6 | 13.4 |
| Qwen-VL-Chat | **AIR** | **18.6** | **5.9** | 13.6 |
| Qwen-VL-Chat | **SRF** | ? | ? | ? |

---

## Experiments to Run

### Priority 1: Direct Comparison (Same datasets, same models)

1. **POPE on LLaVA-1.5-7B** (MSCOCO, A-OKVQA, GQA)
   - Run: `python srf/eval.py --method srf --model llava-hf/llava-1.5-7b-hf --datasets pope --output results/llava_pope_comparison/`
   - Compare to VAF/AIR rows above

2. **POPE on Qwen-VL-Chat** (MSCOCO, A-OKVQA, GQA)
   - Run: `python srf/eval.py --method srf --model Qwen/Qwen-VL-Chat --datasets pope --output results/qwen_pope_comparison/`
   - Compare to VAF/AIR rows above

3. **MME on LLaVA-1.5-7B**
   - Run: `python srf/eval.py --method srf --model llava-hf/llava-1.5-7b-hf --datasets mme --output results/llava_mme_comparison/`

4. **MME on Qwen-VL-Chat**
   - Run: `python srf/eval.py --method srf --model Qwen/Qwen-VL-Chat --datasets mme --output results/qwen_mme_comparison/`

5. **CHAIR on LLaVA-1.5-7B** (NEW - need to implement)
6. **CHAIR on Qwen-VL-Chat** (NEW - need to implement)

### Priority 2: Our unique contributions

7. **Qwen2.5-VL-3B** on POPE, MME, CHAIR (not in VAF/AIR papers)
8. **MMVP** on all models (not in VAF/AIR papers)
9. **VLM Bias** on all models (not in VAF/AIR papers)

---

## Implementation Tasks

- [ ] Add CHAIR dataset loader to `srf/eval_datasets.py`
- [ ] Add A-OKVQA and GQA to POPE splits (currently only MSCOCO?)
- [ ] Verify our POPE evaluation matches paper's setup (averaging across 3 datasets)
- [ ] Verify our MME evaluation matches paper's setup (existence/count/position/color subsets)
