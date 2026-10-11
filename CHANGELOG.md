# Değişiklikler

## 0.1.0 — ilk açık sürüm

- İlk yayın. Ayrıntılar ve ölçüler README'de.
- Türkçe PL-BERT: papercup-ai çok dilli PL-BERT'ten (step_1100000) Türkçe Vikipedi'nin vurgu işaretli
  okunuşuyla 150.000 adım ince ayar. Doğrulama: sözcük isabeti %90,6, maske (ses) isabeti %48,2.
- Veri hazırlama (`veri.py`), eğitim (`egit.py`, `config_tr.yml`) ve kiralık GPU betiği (`plbert.sh`).
- Hugging Face model kartı ve yükleme betiği (`hf/`; varsayılan gizli depo).
- GitHub Actions: betiklerin derlenmesi, `plbert.sh` sözdizimi ve yapılandırmanın okunuşu.
