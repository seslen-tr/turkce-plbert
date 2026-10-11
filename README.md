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
| Okunuş | [turkce-vurgu](https://github.com/seslen-tr/turkce-vurgu) G2P'si, **vurgu işaretli** (ˈ). Ses modeli de aynı okunuşu görür |
| Sözcük parçalayıcı | [dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased) (MIT); sözcüğün ilk parçası (eklemeli dilde ≈ kök) |
| Görevler | Maskelenmiş ses tahmini + her sesin ait olduğu sözcüğü tahmin |
| Eğitim | 150.000 adım, batch 32, fp16, tek RTX 4090 (~9 saat). Öğrenme oranı 1e-4, 2.000 adım ısınma + kosinüsle 1e-5'e iniş (AdamW) |

Model ağırlıkları (`step_150000.t7`, `config.yml`) Hugging Face'te
[seslen-tr/turkce-plbert](https://huggingface.co/seslen-tr/turkce-plbert) deposuna konur (model kartı: `hf/README.md`).

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

`veri.py --sinir 2000` küçük bir deneme verisi üretir. `egit.py` verinin %1'ini doğrulamaya ayırır, her
10.000 adımda kayıt alır ve doğrulama ölçülerini yazar (`--lr`, `--isinma`, `--kayit` ile değişir).

`plbert.sh` kiralık GPU'da (RunPod) kurulum ve çalıştırma betiğidir; adım ve batch `ADIM` ve `BATCH`
ortam değişkenlerinden okunur (varsayılan 100.000 / 64). Yayımlanan sürüm `ADIM=150000 BATCH=32` ile
eğitildi: batch 64, 24 GB bellekte taşar, çünkü sözcük başlığı yaklaşık 19 bin sınıflıdır. Betik bitince
makineyi `GRACE` dakika sonra durdurur.

## Düzen

| Dosya | İçerik |
|---|---|
| `veri.py` | Türkçe Vikipedi → vurgulu okunuş + sözcük kimlikleri (`veri/tr`, `veri/token_maps.pkl`) |
| `egit.py` | eğitim (yl4579/PL-BERT train.ipynb'nin betik hâli) |
| `config_tr.yml` | model ve veri ayarları; `egit.py` yol, ayırıcı, batch ve adımı doldurur |
| `plbert.sh` | kiralık GPU'da kurulum ve eğitim |
| `hf/README.md`, `hf/yukle.py` | Hugging Face model kartı ve yükleme betiği (`python hf/yukle.py [--acik]`, varsayılan gizli depo) |

`veri/`, `model/` ve `PL-BERT/` (yl4579/PL-BERT klonu) git dışındadır. GitHub Actions yalnız betiklerin
derlendiğini, `plbert.sh` sözdizimini ve `config_tr.yml`'nin okunduğunu sınar; eğitim GPU ister.

## Ölçüler

Doğrulama, ayrılmış %1 veride yapıldı (yayımlanan sürüm, vurgu işaretli okunuş, 150.000 adım; 03.10.2026).

| Ölçü | 80.000. adım | 150.000. adım (yayımlanan) |
|---|---|---|
| Sözcük isabeti (her sesin ait olduğu sözcük) | %89,9 | **%90,6** |
| Maske (ses) isabeti | %45,4 | **%48,2** |
| Sözcük kaybı / ses kaybı | 0,614 / 1,735 | 0,556 / 1,635 |

Ses modelindeki katkısı: Seslen'in 3. aşaması (bu model + vurgulu okunuş) 2. aşamayla (İngilizce
çok dilli PL-BERT, vurgusuz) aynı anlaşılırlıkta (%100) ve doğallıkta (UTMOS 3,60 / 3,58).

## Kardeş projeler

Okunuş [turkce-vurgu](https://github.com/seslen-tr/turkce-vurgu) ile çıkarılır; model
[seslen-tts](https://github.com/seslen-tr/seslen-tts) ses modelinin 3. eğitim aşamasında kullanılır.
Diğerleri: [turkce-normalize](https://github.com/seslen-tr/turkce-normalize) ·
[turkce-duygu](https://github.com/seslen-tr/turkce-duygu) · [turkce-uslup](https://github.com/seslen-tr/turkce-uslup) ·
[turkce-tts-sinav](https://github.com/seslen-tr/turkce-tts-sinav)

## Lisans

- **Kod:** MIT.
- **Model ağırlıkları:** Türkçe Vikipedi metniyle eğitildi (CC BY-SA 4.0); başlangıç modeli MIT.

## Kaynaklar

- Y. A. Li, C. Han, X. Jiang, N. Mesgarani (2023). Phoneme-Level BERT for Enhanced Prosody of
  Text-to-Speech with Grapheme Predictions. ICASSP. Kod: https://github.com/yl4579/PL-BERT
- Papercup, multilingual PL-BERT checkpoint.
