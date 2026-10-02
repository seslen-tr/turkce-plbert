# -*- coding: utf-8 -*-
"""
TÜRKÇE PL-BERT EĞİTİMİ — çok dilli PL-BERT'ten (papercup-ai, MIT) Türkçeye ince ayar.

yl4579/PL-BERT train.ipynb'nin betik hâli; farklar:
  - Başlangıç: papercup-ai/multilingual-pl-bert step_1100000 (15 dil, Türkçe yok). Yayımlanan
    kayıtta yalnız kodlayıcı var (StyleTTS2'nin kullandığı da o); iki başlık SIFIRDAN.
  - Isınma + kosinüs öğrenme oranı; %1 ayrılmış doğrulama (maske ve sözcük isabeti).
  - Kayıt StyleTTS2'nin Utils/PLBERT/util.py'sinin beklediği biçimde ("module." önekli 'net').
Ortam (kiralık GPU): /workspace/plbert ← veri/tr, veri/token_maps.pkl, bu klasörün betikleri; yl4579/PL-BERT deposu yanında.

    python egit.py --veri /workspace/plbert --adim 100000 --batch 64
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import time
from pathlib import Path

import torch
import yaml
from torch import nn

BURASI = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--veri", default="/workspace/plbert")
    ap.add_argument("--depo", default="/workspace/PL-BERT", help="yl4579/PL-BERT (dataloader, model)")
    ap.add_argument("--baslangic", default="/workspace/plbert/ml/step_1100000.t7")
    ap.add_argument("--cikti", default="/workspace/plbert/Checkpoint_tr")
    ap.add_argument("--adim", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--isinma", type=int, default=2000)
    ap.add_argument("--kayit", type=int, default=10_000)
    a = ap.parse_args()
    sys.path.insert(0, a.depo)
    from accelerate import Accelerator
    from datasets import load_from_disk
    from transformers import AlbertConfig, AlbertModel
    from dataloader import build_dataloader
    from model import MultiTaskModel
    from utils import length_to_mask

    veri = Path(a.veri)
    cikti = Path(a.cikti)
    cikti.mkdir(parents=True, exist_ok=True)
    tm_yol = veri / "token_maps.pkl"
    token_maps = pickle.load(open(tm_yol, "rb"))
    from transformers import AutoTokenizer
    ayirici = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased").sep_token_id
    cfg = yaml.safe_load(open(BURASI / "config_tr.yml"))
    cfg["dataset_params"].update({"token_maps": str(tm_yol), "word_separator": ayirici})
    cfg.update({"batch_size": a.batch, "num_steps": a.adim, "save_interval": a.kayit, "log_dir": str(cikti)})
    yaml.safe_dump(cfg, open(cikti / "config.yml", "w"), allow_unicode=True, sort_keys=False)

    ds = load_from_disk(str(veri / "tr")).train_test_split(test_size=0.01, seed=1)
    yuk = build_dataloader(ds["train"], batch_size=a.batch, num_workers=6, dataset_config=cfg["dataset_params"])
    dog = build_dataloader(ds["test"], validation=True, batch_size=a.batch, num_workers=2,
                           dataset_config=cfg["dataset_params"])

    sinif = 1 + max(m["token"] for m in token_maps.values())
    bert = MultiTaskModel(AlbertModel(AlbertConfig(**cfg["model_params"])), num_vocab=sinif,
                          num_tokens=cfg["model_params"]["vocab_size"], hidden_size=cfg["model_params"]["hidden_size"])
    opt = torch.optim.AdamW(bert.parameters(), lr=a.lr, weight_decay=0.01)
    adim = 0
    onceki = sorted(cikti.glob("step_*.t7"), key=lambda p: int(p.stem.split("_")[1]))
    if onceki:                                   # kaldığı yerden
        d = torch.load(onceki[-1], map_location="cpu")
        bert.load_state_dict({k.removeprefix("module."): v for k, v in d["net"].items()})
        opt.load_state_dict(d["optimizer"])
        adim = d["step"]
        print(f"sürdürülüyor: {onceki[-1].name}", flush=True)
    else:
        d = torch.load(a.baslangic, map_location="cpu")["net"]
        al = {k.removeprefix("module."): v for k, v in d.items() if not k.startswith("module.word_predictor")}
        eksik = bert.load_state_dict(al, strict=False)
        print(f"çok dilli başlangıç yüklendi; yeni: {eksik.missing_keys}", flush=True)

    hiz = Accelerator(mixed_precision="fp16")
    bert, opt, yuk, dog = hiz.prepare(bert, opt, yuk, dog)
    oran = lambda s: min(1.0, (s + 1) / a.isinma) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.adim))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, oran, last_epoch=adim - 1 if adim else -1)
    kayip_f = nn.CrossEntropyLoss()

    def kayiplar(batch):
        words, labels, phonemes, input_lengths, masked_indices = batch
        maske = length_to_mask(torch.Tensor(input_lengths)).to(phonemes.device)
        t_pred, w_pred = bert(phonemes, attention_mask=(~maske).int())
        lv = sum(kayip_f(p[:n], w[:n]) for p, w, n in zip(w_pred, words, input_lengths)) / words.size(0)
        lt, say, isabet_t, top_t = 0, 1, 0, 0
        for p, l, n, mi in zip(t_pred, labels, input_lengths, masked_indices):
            if len(mi) > 0:
                lt += kayip_f(p[:n][mi], l[:n][mi])
                say += 1
                isabet_t += (p[:n][mi].argmax(-1) == l[:n][mi]).sum().item()
                top_t += len(mi)
        isabet_w = sum((p[:n].argmax(-1) == w[:n]).sum().item() for p, w, n in zip(w_pred, words, input_lengths))
        return lv, lt / say, isabet_t / max(top_t, 1), isabet_w / sum(input_lengths)

    t0, toplam = time.time(), 0.0
    bert.train()
    while adim < a.adim:
        for batch in yuk:
            lv, lt, _, _ = kayiplar(batch)
            opt.zero_grad()
            hiz.backward(lv + lt)
            hiz.clip_grad_norm_(bert.parameters(), 1.0)
            opt.step()
            sched.step()
            adim += 1
            toplam += (lv + lt).item()
            if adim % 100 == 0:
                print(f"Step [{adim}/{a.adim}] kayıp {toplam / 100:.4f} (sözcük {lv.item():.3f}, ses {float(lt):.3f}) "
                      f"lr {sched.get_last_lr()[0]:.2e} {100 / (time.time() - t0):.2f} adım/sn", flush=True)
                t0, toplam = time.time(), 0.0
            if adim % a.kayit == 0 or adim >= a.adim:
                bert.eval()
                with torch.no_grad():
                    olc = [kayiplar(b) for _, b in zip(range(50), dog)]
                bert.train()
                print(f"DOĞRULAMA adım {adim}: sözcük kaybı {sum(float(x[0]) for x in olc) / len(olc):.3f}, "
                      f"ses kaybı {sum(float(x[1]) for x in olc) / len(olc):.3f}, "
                      f"maske isabeti %{100 * sum(x[2] for x in olc) / len(olc):.1f}, "
                      f"sözcük isabeti %{100 * sum(x[3] for x in olc) / len(olc):.1f}", flush=True)
                net = {f"module.{k}": v for k, v in hiz.unwrap_model(bert).state_dict().items()}
                torch.save({"net": net, "step": adim, "optimizer": opt.state_dict()}, cikti / f"step_{adim}.t7")
                for eski in sorted(cikti.glob("step_*.t7"), key=lambda p: int(p.stem.split("_")[1]))[:-2]:
                    eski.unlink()                    # son iki kayıt yeter (disk)
            if adim >= a.adim:
                break
    print("BITTI", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
