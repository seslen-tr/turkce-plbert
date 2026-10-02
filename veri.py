# -*- coding: utf-8 -*-
"""
TÜRKÇE PL-BERT VERİSİ — Türkçe Vikipedi → (sözcük kimlikleri, sözcük okunuşları).

PL-BERT (yl4579/PL-BERT) ses dizisinden hem maskelenmiş sesleri hem de her sesin
ait olduğu SÖZCÜĞÜ tahmin etmeyi öğrenir; böylece okunuştan anlam ve bağlam çıkarır,
StyleTTS2 de vurgu/tonlamayı ondan alır. Özgün sürüm İngilizce + espeak; burada:
  - okunuş: turkce_vurgu G2P'si, VURGULU — ses modeli de aynısını görür
  - sözcük: dbmdz/bert-base-turkish-cased (MIT) parçalayıcısı, sözcüğün ilk parçası
    (Türkçe eklemeli: "kitaplarımızdan" → "kitap…" ≈ kök)
Çıktı: veri/tr/ (datasets, save_to_disk) + veri/token_maps.pkl.

    python veri.py --sinir 2000        # deneme
    python veri.py --karakter 300e6    # ~300 milyon ses

Vikipedi metni CC BY-SA 4.0: eğitilen modelin kartında kaynak belirtilir.
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

KOK = Path(__file__).resolve().parent
from turkce_vurgu.g2p import NOKTALAMA, normallestir, ozel_kokler, sozcuk_okunusu  # noqa: E402  (pip install turkce-vurgu)

PARCALAYICI = "dbmdz/bert-base-turkish-cased"
HEDEF = KOK / "veri" / "tr"
TR = "a-zA-ZçğıöşüÇĞİÖŞÜâîûÂÎÛ"
_SIMGE = re.compile(rf"[{TR}]+(?:['’][{TR}]+)?|\d+(?:[.,]\d+)*|[{re.escape(NOKTALAMA)}]|\S")
_YABANCI = re.compile(rf"[^{TR}\d\s{re.escape(NOKTALAMA)}'’\"“”()\[\]«»%\-–—/]")
_INGILIZCE = re.compile(r"[qwxQWX]|\b(the|of|and|with|for|from|The|Of)\b")
_tok = None


def _parcalayici():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(PARCALAYICI)
    return _tok


def paragraflar(metin: str):
    for p in metin.split("\n"):
        p = p.strip()
        # başlık, liste, tablo artığı, çok kısa satır: atla
        if len(p) < 80 or p.count(" ") < 8 or p[-1] not in ".!?…:":
            continue
        if len(_YABANCI.findall(p)) > len(p) * 0.01:
            continue
        # Türkçe kurallarla okunan yabancı sözcükler (The Station → ˈthe staˈtion) veriyi kirletiyor (02.10)
        if _INGILIZCE.search(p):
            continue
        yield p


def isle(p: str) -> dict | None:
    tok = _parcalayici()
    kokler = ozel_kokler(p)
    kimlik, okunus = [], []
    for s in _SIMGE.findall(p):
        if s in NOKTALAMA:
            kimlik.append(tok.convert_tokens_to_ids(s))
            okunus.append(s)
            continue
        if not re.match(rf"[{TR}\d]", s):
            continue                                      # tırnak, ayraç, tire: okunmaz
        ilk = tok.tokenize(s.replace("’", "'"))[:1]
        if not ilk:
            continue
        k = tok.convert_tokens_to_ids(ilk[0])
        for w in re.findall(r"[a-zçğıöşüâîûwxq]+", normallestir(s)):
            okunus.append(sozcuk_okunusu(w, kokler, vurgu=True))
            kimlik.append(k)
    return {"input_ids": kimlik, "phonemes": okunus} if len(okunus) >= 8 else None


def _yigin(metinler: list[str]) -> list[dict]:
    return [r for m in metinler for p in paragraflar(m) if (r := isle(p))]


def token_maps(ds, ayirici: int) -> dict:
    """Notebook'taki gibi: geçen kimlikler → küçük harfli sözcüğün sıkıştırılmış sırası."""
    tok = _parcalayici()
    gecen = {ayirici}
    for ids in ds["input_ids"]:
        gecen.update(ids)
    harita, sira = {}, {}
    for t in sorted(gecen):
        sozcuk = tok.convert_ids_to_tokens(t)
        kucuk = sozcuk.replace("I", "ı").replace("İ", "i").lower() if t != ayirici else sozcuk
        yeni = tok.convert_tokens_to_ids(kucuk)
        yeni = t if yeni == tok.unk_token_id else yeni
        harita[t] = {"word": kucuk, "token": sira.setdefault(yeni, len(sira))}
    return harita


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sinir", type=int, default=0, help="en çok bu kadar makale (deneme)")
    ap.add_argument("--karakter", type=float, default=400e6, help="işlenecek paragraf karakteri üst sınırı")
    ap.add_argument("--isci", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    a = ap.parse_args()
    from datasets import Dataset, load_dataset
    wiki = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train")
    if a.sinir:
        wiki = wiki.select(range(min(a.sinir, len(wiki))))
    wiki = wiki.shuffle(seed=7)
    t0, toplam, kayit = time.time(), 0, []
    yiginlar = (wiki[i:i + 200]["text"] for i in range(0, len(wiki), 200))
    with Pool(a.isci) as havuz:
        for parca in havuz.imap(_yigin, yiginlar):
            kayit += parca
            toplam += sum(len("".join(r["phonemes"])) for r in parca)
            if len(kayit) % 20000 < len(parca):
                print(f"{len(kayit)} paragraf, {toplam / 1e6:.0f} M ses, {time.time() - t0:.0f} sn", flush=True)
            if toplam >= a.karakter:
                havuz.terminate()
                break
    ds = Dataset.from_list(kayit)
    HEDEF.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(HEDEF))
    ayirici = _parcalayici().sep_token_id
    tm = token_maps(ds, ayirici)
    with open(HEDEF.parent / "token_maps.pkl", "wb") as f:
        pickle.dump(tm, f)
    print(f"bitti: {len(ds)} paragraf, {toplam / 1e6:.0f} M ses, {len(set(m['token'] for m in tm.values()))} "
          f"sözcük sınıfı, ayırıcı {ayirici}, {time.time() - t0:.0f} sn")
    print("örnek:", list(zip(ds[0]["phonemes"][:12], _parcalayici().convert_ids_to_tokens(ds[0]["input_ids"][:12]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
