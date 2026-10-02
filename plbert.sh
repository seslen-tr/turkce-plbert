#!/bin/bash
# Türkçe PL-BERT ince ayarı (plbert/egit.py) — kiralık GPU'da (RunPod) çalışır.
# Bitince EGITIM gibi işaret bırakır ve GRACE dk sonra makineyi durdurur (izleyici yoksa para yanmasın).
cd /workspace; exec >> /workspace/plbert.log 2>&1
GRACE=${GRACE:-10}
bitir() { echo "== $(date) $1; $GRACE dk sonra durdurulacak"; echo "$1" > /workspace/PLBERT_BITTI; sleep $((GRACE * 60)); runpodctl stop pod "$(cat /workspace/.pod_id)"; exit; }
rm -f /workspace/PLBERT_BITTI
pip install -q "transformers==4.46.3" "accelerate<1.3" "datasets>=3.6" pyyaml || { GRACE=30; bitir "HATA: paketler"; }
cd /workspace/plbert
python egit.py --veri /workspace/plbert --depo /workspace/plbert/PL-BERT --baslangic /workspace/plbert/ml/step_1100000.t7 \
  --cikti /workspace/plbert/Checkpoint_tr --adim "${ADIM:-100000}" --batch "${BATCH:-64}"
kod=$?; [ $kod -ne 0 ] && GRACE=30; bitir "eğitim bitti (çıkış $kod)"
