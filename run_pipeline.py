#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run full IFRS9 Sandbox pipeline:
1. Veri üretimi (panel_generator.py)
2. Modelleme (modeling.py)
3. Validasyon (validation.py)
4. ECL hesaplamaları (ecl.py)
5. Rapor üretimi (report.py)
"""

import argparse
import os
from pathlib import Path

# Modüller import 
import panel_generator as pg
import modeling as mdl
import validation as val
import ecl as ecl
import report as rpt


def main():
    parser = argparse.ArgumentParser(description="Run IFRS9 Sandbox Pipeline")
    parser.add_argument("--n", type=int, default=100000, help="Müşteri sayısı")
    parser.add_argument("--months", type=int, default=36, help="Panel uzunluğu (ay)")
    parser.add_argument("--scenario", type=str, default="baseline",
                        choices=["baseline", "mild_downturn", "severe_downturn"],
                        help="Makro senaryo")
    parser.add_argument("--outdir", type=str, default="artifacts", help="Çıktı klasörü")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("🔹 1. Veri üretimi başlıyor...")
    snapshot, panel = pg.generate_portfolio(
        n=args.n, months=args.months, scenario=args.scenario
    )
    snapshot_path = outdir / "ifrs9_snapshot.parquet"
    panel_path = outdir / "ifrs9_panel.parquet"
    snapshot.to_parquet(snapshot_path, index=False)
    panel.to_parquet(panel_path, index=False)
    print(f"✅ Snapshot kaydedildi: {snapshot_path}")
    print(f"✅ Panel kaydedildi: {panel_path}")

    print("🔹 2. Modelleme başlıyor (PD/LGD/EAD tahminleri)...")
    scored_df, models = mdl.run_models(snapshot, outdir=outdir)
    scored_path = outdir / "snapshot_scored.parquet"
    scored_df.to_parquet(scored_path, index=False)
    print(f"✅ Skorlu veri kaydedildi: {scored_path}")

    print("🔹 3. Validasyon çalışıyor...")
    metrics = val.run_validation(scored_df, outdir=outdir)
    print("✅ Validasyon metrikleri kaydedildi.")

    print("🔹 4. ECL hesaplamaları...")
    ecl_results = ecl.run_ecl(scored_df, panel, outdir=outdir/"ecl")
    print("✅ ECL sonuçları kaydedildi.")

    print("🔹 5. Rapor üretimi...")
    rpt.build_report(outdir=outdir, metrics=metrics, ecl_results=ecl_results)
    print("✅ HTML/PDF rapor tamamlandı.")

    print("\n🎉 Pipeline başarıyla tamamlandı!")


if __name__ == "__main__":
    main()
