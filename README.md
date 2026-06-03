# LoRA-Boost: AI Builders 2026

Generative augmentation สำหรับ long-tail plant species classification

เทรน per-species LoRA บน FLUX.2-klein generate รูป synthetic ให้ rare species แล้วใช้ augment
classifier บน Pl@ntNet-300K โดยไม่แตะ architecture

## Pipeline

| Notebook | ทำอะไร | เปิดดู |
|---|---|---|
| NB01 | Data prep: clean Pl@ntNet, DINOv2 outlier removal, เลือก 15 rare species, split | [GitHub](notebooks/01_data.ipynb) · [Colab](https://drive.google.com/file/d/1SvcbaNDMRKn1rv8K2tUGsBogR0LPmbY6/view?usp=sharing) |
| NB02 | เทรน 15 per-species LoRA บน FLUX.2-klein + eval คุณภาพด้วย DINOv2 | [GitHub](notebooks/02_lora.ipynb) · [Colab](https://drive.google.com/file/d/1FtgZIel3sS8-XoUugkx1HGIrI5Uwk_YP/view?usp=sharing) |
| NB03a | Generate synthetic ด้วย LoRA (condition D) | [GitHub](notebooks/03a_synth_lora.ipynb) · [Colab](https://drive.google.com/file/d/1qxRR1oMZku80MLJlxCnOlsPgzXcTCm7r/view?usp=sharing) |
| NB03b | Generate synthetic ด้วย FLUX zero-shot ไม่ใช้ LoRA (condition C) | [GitHub](notebooks/03b_synth_zeroshot.ipynb) · [Colab](https://drive.google.com/file/d/1pOfJukn47LOgcyDrk8KYdY6gbxDS50_F/view?usp=sharing) |
| NB04 | เทรน ResNet-50 classifier 4 conditions (A/B/C/D) เทียบ rare macro F1 | [GitHub](notebooks/04_classifier.ipynb) · [Colab](https://drive.google.com/file/d/1BFpYCHckE3HAxCnMS59kLaVqQnv6mtkg/view?usp=sharing) |
| NB05 | Error analysis: วิเคราะห์ว่าทำไม augmentation ทำให้ rare แย่ลง | [GitHub](notebooks/05_error_analysis.ipynb) · [Colab](https://drive.google.com/file/d/1qFKV8HCBzSQpGkWVFaMLKQlQmO5TqSMH/view?usp=sharing) |

## 1. Problem statement

Plant classification dataset มี long-tail rare tropical species มีรูปน้อย (20-71 รูป) ทำให้ classifier
จำได้แย่ คำถามคือ generative augmentation ด้วย per-species LoRA ช่วย rare species ได้ไหม
ปัญหานี้ตรงกับงานจริงด้าน biodiversity monitoring ที่ species หายากมักมีข้อมูลน้อยเสมอ

## 2. Metrics & baselines

มี metric 2 ระดับ

**ระดับ LoRA (NB02)** วัดคุณภาพรูป synthetic ก่อนเอาไปใช้ ด้วย DINOv2 embedding
- consistency: รูป gen เหมือน species จริงแค่ไหน
- organ_acc: gen ออกมาตรง organ ที่สั่งไหม (ดอก/ใบ)
- memorization: overfit copy รูป train ไหม

ผล: LoRA 13/15 ตัวผ่านเกณฑ์ (ok) flag 2 ตัว คือ Kigelia africana (consistency 0.49 ต่ำ)
และ Acalypha indica (organ_acc 0.567 ต่ำ) memorization ได้ ~0 ทุกตัว ไม่มี overfit
consistency รวมอยู่ราว 0.49 ถึง 0.80 metric พวกนี้ feed ต่อไป error analysis (NB05)
ซึ่งพบว่า consistency correlate กับ classifier error จริง (r=0.61)

**ระดับ classifier (NB04)** metric หลักของโปรเจกต์: **macro F1 บน 15 rare species** (test split)
เลือก macro เพราะ micro จะถูกกลบด้วย common species ที่รูปเยอะ เทียบ 4 conditions ที่ต่างกัน
เฉพาะวิธี augment rare species

| Cond | Augmentation | rare F1 |
|---|---|---|
| A | none (baseline) | **0.644** |
| B | traditional rotate/flip | 0.594 |
| C | FLUX zero-shot (no LoRA) | 0.502 |
| D | LoRA-Boost (ours) | 0.546 |

Baseline A (ไม่ augment เลย) ได้ดีที่สุด augmentation ทุกแบบทำให้แย่ลง เป็น negative result
ต่อสมมติฐานหลัก แต่ D > C (+0.044) แสดงว่า LoRA ช่วยจริงเทียบกับ zero-shot

## 3. Data collection & cleaning

Pl@ntNet-300K v1.1 (240k รูป 399 species) ทำความสะอาดใน NB01: ลบรูปเสีย, DINOv2 outlier
detection ราย species และ organ, กรอง species ที่มีครบทั้งใบและดอก, เลือก 15 rare tropical species
จาก lower tail (n_train 20-71) ด้วย GBIF occurrence + manual review แล้ว split per-species 70/15/15

## 4. Exploratory data analysis

NB01 วิเคราะห์การกระจาย long-tail, count ต่อ species, สัดส่วน organ (ใบ/ดอก) ของ rare species
และ outlier ที่ DINOv2 จับได้ ใช้ตัดสินใจ T (จำนวนภาพต่อ rare หลัง augment) และ rare species selection

## 5. Modeling, validation & error analysis

**Model:** ResNet-50 (ImageNet-pretrained, fine-tune all, 399 classes) เหมือนกันทุก condition
**LoRA:** FLUX.2-klein-base-4B, per-species, rank 8, DreamBooth-style (NB02)
**Validation:** train/val/test split คงที่ seed 42 reproducible ทุก run

**Error analysis (NB05):** วิเคราะห์ว่าทำไม D แพ้ A
- D ทำ rare แย่ลง 10/15 species กระจุกที่ Petiveria + Pereskia pair
- ไม่ใช่ hard-pair confusion เพราะ rare แทบไม่สับกันเอง
- กลไก: D ดัน rare ไปถูกทายเป็น common มากขึ้น (leak 41.5% → 49.6%)
- LoRA consistency correlate กับ error (r=0.61, p=0.015)
- รูป synthetic บางตัวผิด species (เช่น Pereskia aculeata ได้ดอกเหลือง) หรือเป็นใบเขียวเปล่า
  ที่ไม่มี discriminative feature

## 7. Non-trivial data sources

ไม่ใช่ดาวน์โหลด dataset สำเร็จมาเทรนตรงๆ แต่สร้างและคัดข้อมูลใหม่เองในหลายจุด

- **Data ที่เทรน LoRA** คัดรูปจริงของ rare species ราย (species, organ) ผ่าน DINOv2 filtering
  แล้วออกแบบ caption เอง (natural language ไม่ใส่ genus name กัน prior-fighting) ไม่ใช่โยนรูปดิบเข้าเทรน
- **Synthetic images ที่เทรน classifier** generate ใหม่เองด้วย LoRA 15 ตัวบน FLUX.2-klein-base-4B
  เป็น training data ที่ไม่เคยมีในข้อมูลจริง สร้างเองทั้งหมด (NB02-03)
- **GBIF occurrence API** ใช้ filter tropical species จาก native range ไม่ใช่แค่ count ใน dataset
- **Multi-stage cleaning ด้วย DINOv2** outlier detection ราย (species, organ) ก่อนเข้า pipeline

## Model weights & data

Weights และรูปไม่ได้อยู่ใน repo เพราะขนาดใหญ่:
- **LoRA weights** (15 per-species models) — จะอัปขึ้น HuggingFace Hub
- **Classifier checkpoints** (A/B/C/D) — regenerate ได้จาก NB04 (reproducible, seed 42)
- **รูป raw + synthetic** — raw จาก Pl@ntNet-300K v1.1, synthetic regenerate ได้จาก NB02-03

รัน notebook ตามลำดับ NB01 → NB05 จะได้ผลลัพธ์เดิมทั้งหมด

## License

CC-BY-SA 4.0
