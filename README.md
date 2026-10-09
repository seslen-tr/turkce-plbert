# turkce-plbert

> **English:** Turkish PL-BERT: a phoneme-level language model that feeds prosody (stress, intonation, rhythm) into TTS models such as StyleTTS2 and Kokoro. ALBERT architecture, initialized from multilingual PL-BERT and trained on ~800k Turkish Wikipedia paragraphs transcribed with stress-marked IPA (turkce-vurgu).

**Türkçe PL-BERT:** okunuştan (ses dizisinden) bağlam çıkaran dil modeli. StyleTTS2 ve Kokoro
gibi ses sentezi modellerinde vurgu, tonlama ve ritmi besler. Hazır çok dilli PL-BERT
(papercup-ai) 15 dil biliyor ama Türkçe bilmiyor. Bu proje o boşluğu doldurur.

## Nasıl eğitildi

| | |
|---|---|
| Mimari | ALBERT; 12 paylaşımlı katman, gizli boyut 768. Çok dilli PL-BERT ile aynı, StyleTTS2'ye doğrudan takılır |
| Başlangıç | [papercup-ai/multilingual-pl-bert](https://huggingface.co/papercup-ai/multilingual-pl-bert), step_1100000 (MIT). Yalnız kodlayıcı alındı, iki başlık sıfırdan |
| Veri | Türkçe Vikipedi (20231101.tr), yaklaşık 800 bin paragraf, 300 milyon ses. Yabancı sözcük içeren paragraflar ayıklandı |
| Okunuş | [turkce-vurgu](../turkce-vurgu) G2P'si, **vurgu işaretli** (ˈ). Ses modeli de aynı okunuşu görür |
| Sözcük parçalayıcı | [dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased) (MIT); sözcüğün ilk parçası (eklemeli dilde ≈ kök) |
| Görevler | Maskelenmiş ses tahmini + her sesin ait olduğu sözcüğü tahmin |
| Eğitim | 150.000 adım, batch 32, fp16, tek RTX 4090 (~9 saat). Öğrenme oranı 1e-4, ısınma + kosinüs |

## StyleTTS2'de kullanım

```
StyleTTS2/Utils/PLBERT_tr/
    config.yml
    step_150000.t7
```

Yapılandırmada `PLBERT_dir: Utils/PLBERT_tr/` olarak ayarlanır. Metin, StyleTTS2'nin 178
sembollük listesiyle uyumlu `turkce_vurgu.g2p(metin, vurgu=True)` ile okunuşa çevrilir.
StyleTTS2'nin eski `Utils/PLBERT/util.py` dosyası `embeddings.position_ids` anahtarını
silmeye çalışır ve hata verir. O satır `new_state_dict.pop("embeddings.position_ids", None)`
olarak değiştirilmelidir.

## Yeniden üretmek

```
pip install turkce-vurgu datasets transformers accelerate
git clone https://github.com/yl4579/PL-BERT          # dataloader.py, model.py, utils.py, text_utils.py
python veri.py --karakter 300e6                      # Vikipedi → veri/tr + veri/token_maps.pkl
# başlangıç ağırlığı: papercup-ai/multilingual-pl-bert step_1100000.t7 → veri/ml/
python egit.py --veri veri --depo PL-BERT --baslangic veri/ml/step_1100000.t7 --cikti model --adim 150000 --batch 32
```

`plbert.sh` kiralık GPU'da (RunPod) kurulum ve çalıştırma betiğidir. Batch 64, 24 GB bellekte
taşar: sözcük başlığı yaklaşık 19 bin sınıflıdır.

## Ölçüler

Doğrulama, ayrılmış %1 veride yapıldı. İlk sürüm (eski vurgu kurallarıyla) 80 bininci adımda
şu değerlere ulaştı; ikinci sürümün sonuçları eğitim bitince eklenecek.

| Ölçü | Değer |
|---|---|
| Sözcük isabeti | %89,8 |
| Maske (ses) isabeti | %45 |

## Lisans

- **Kod:** MIT.
- **Model ağırlıkları:** Türkçe Vikipedi metniyle eğitildi (CC BY-SA 4.0); başlangıç modeli MIT.

## Kaynaklar

- Y. A. Li, C. Han, X. Jiang, N. Mesgarani (2023). Phoneme-Level BERT for Enhanced Prosody of
  Text-to-Speech with Grapheme Predictions. ICASSP. Kod: https://github.com/yl4579/PL-BERT
- Papercup, multilingual PL-BERT checkpoint.
