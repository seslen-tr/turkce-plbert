---
language: tr
license: cc-by-sa-4.0
tags: [turkish, pl-bert, phoneme, prosody, tts, styletts2]
base_model: papercup-ai/multilingual-pl-bert
datasets: [wikimedia/wikipedia]
---

# Türkçe PL-BERT

> **English:** Phoneme-level BERT for Turkish (ALBERT, 12 shared layers, hidden 768), initialized from
> papercup-ai/multilingual-pl-bert and trained 150k steps on ~800k Turkish Wikipedia paragraphs transcribed to
> stress-marked IPA with turkce-vurgu. Drop-in for StyleTTS2's `Utils/PLBERT`.

Okunuştan (ses dizisinden) bağlam çıkaran dil modeli: StyleTTS2 ve Kokoro gibi modellerde vurgu, tonlama ve ritmi besler.

| | |
|---|---|
| Mimari | ALBERT; 12 paylaşımlı katman, gizli boyut 768 (çok dilli PL-BERT ile aynı) |
| Başlangıç | papercup-ai/multilingual-pl-bert, step_1100000 (MIT) |
| Veri | Türkçe Vikipedi (20231101.tr), ~800 bin paragraf, ~300 milyon ses |
| Okunuş | turkce-vurgu G2P'si, vurgu işaretli (ˈ); ses modeli de aynı okunuşu görmeli |
| Sözcük parçalayıcı | dbmdz/bert-base-turkish-cased (sözcüğün ilk parçası) |
| Görevler | maskelenmiş ses tahmini + her sesin ait olduğu sözcüğü tahmin |
| Eğitim | 150.000 adım, batch 32, fp16, tek RTX 4090 (~9 saat); öğrenme oranı 1e-4, ısınma + kosinüs |

## Dosyalar

| Dosya | İçerik |
|---|---|
| `step_150000.t7` (247 MB) | eğitim kaydı (`net`: "module." önekli ağırlıklar); StyleTTS2'nin `load_plbert`'i kodlayıcıyı alır |
| `config.yml` | eğitim ayarları; StyleTTS2 yalnız `model_params` bölümünü okur (içindeki `/workspace` yolları eğitim ortamındandır) |

## Ölçüler (ayrılmış %1 doğrulama)

| Ölçü | 150.000. adım |
|---|---|
| Sözcük isabeti | **%90,6** |
| Maske (ses) isabeti | **%48,2** |
| Sözcük kaybı / ses kaybı | 0,556 / 1,635 |

## Kullanım (StyleTTS2)

`step_150000.t7` ve `config.yml` dosyalarını `StyleTTS2/Utils/PLBERT_tr/` altına koyun, eğitim ayarında
`PLBERT_dir: Utils/PLBERT_tr/`. Metni `turkce_vurgu.g2p(metin, vurgu=True)` ile okunuşa çevirin. StyleTTS2'nin
`Utils/PLBERT/util.py` dosyasındaki `del new_state_dict["embeddings.position_ids"]` satırı
`new_state_dict.pop("embeddings.position_ids", None)` olmalı.

Eğitim kodu ve yeniden üretme adımları: [github.com/seslen-tr/turkce-plbert](https://github.com/seslen-tr/turkce-plbert).
Bu model [seslen-tts](https://huggingface.co/seslen-tr/seslen-tts) Türkçe ses modelinde kullanılır.

## Lisans

Ağırlıklar Türkçe Vikipedi metniyle eğitildi: **CC BY-SA 4.0**. Başlangıç modeli MIT. Kod (turkce-plbert): MIT.

Kaynak: Li, Han, Jiang, Mesgarani (2023), Phoneme-Level BERT for Enhanced Prosody of Text-to-Speech with Grapheme
Predictions, ICASSP.
