# IFRS 9 Risk Modeling Sandbox 🚀

Açık kaynaklı **IFRS 9 risk modelleme sandbox** ortamı.  
Kredi riski modellemesi, ECL hesaplama ve makro senaryo stres testleri için kullanılabilir.  

## 🔑 Özellikler
- **Tail Risk için T-Copula** → Uç risk bağımlılıklarını simüle eder  
- **Stage 3 Accrual Stop** → IFRS 9 faiz politikası entegrasyonu  
- **Stokastik Recovery Time-to-Sale** → Tahsilat belirsizliği modellenir  
- **Hyperparameter Tuning (LightGBM + Optuna)** → Monotonic kısıtlı modeller  
- **Otomatik Raporlama** → HTML/PDF çıktılar, Parquet/CSV artifact’ler  

## 🖥️ Kurulum

```bash
# Depoyu klonla
git clone https://github.com/<username# IFRS9 Sandbox

IFRS9 Sandbox, bankacılık ve finans kurumlarının **kredi riski modelleme süreçlerini** (PD, LGD, EAD) deneysel olarak çalıştırabileceği, **açık kaynaklı bir prototip** ortamıdır.  
Proje; makine öğrenmesi, simülasyonlar, senaryo analizi ve otomatik raporlama bileşenlerini bir araya getirir.

## 🎯 Özellikler
- **PD (Probability of Default)** modelleme (Logistic Regression, LightGBM, Monotonic Constraints).
- **LGD (Loss Given Default)** için stokastik recovery modelleri (Time-to-Sale simülasyonu).
- **EAD (Exposure at Default)** hesaplamaları.
- **Tail Risk Analysis** (t-Copula tabanlı bağımlılık).
- **Stage 3 accrual stop policy** simülasyonu.
- **Hyperparameter tuning** (Optuna + LightGBM).
- **Otomatik HTML/PDF rapor üretimi** (Jinja2 + ReportLab).
- **Export**: Parquet / CSV.GZ formatında çıktı.

## 📂 Proje Yapısı
```
ifrs9-sandbox/
│
├── sandbox/              # Python kodları (pipeline, modeller vs.)
├── artifacts/            # Çıktı dosyaları (parquet, pdf, html)
├── docs/                 # Görseller, rapor örnekleri
├── requirements.txt      # Bağımlılıklar
├── README.md             # Proje tanımı
├── LICENSE               # Lisans (MIT)
└── setup.py              # (opsiyonel) pip install için
```

## 🚀 Kurulum
```bash
git clone https://github.com/CelkMehmett/ifrs9-sandbox.git
cd ifrs9-sandbox
pip install -r requirements.txt
```

## ▶️ Çalıştırma
```bash
python sandbox/pipeline.py
```

## 📊 Örnek Çıktılar
- **Parquet/CSV.GZ** dosyaları `artifacts/` klasöründe.
- **HTML/PDF raporları** `docs/` klasöründe.

## 📝 Lisans
Bu proje **MIT Lisansı** altında sunulmaktadır.  
Detaylar için [LICENSE](LICENSE) dosyasına göz atın.>/ifrs9-sandbox.git
cd ifrs9-sandbox

# Sanal ortam oluştur ve bağımlılıkları yükle
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

🚀 Kullanım

from sandbox import run_pipeline_full, load_sample_portfolio, load_macro_scenarios, SandboxCfg

cfg = SandboxCfg()
portfolio = load_sample_portfolio()
macro = load_macro_scenarios()

out = run_pipeline_full(cfg, portfolio, macro)
print(out['summary'].head())

📂 Çıktılar artifacts/ klasöründe:

    ead_path.parquet

    ead_path.csv.gz

    report.html

    report.pdf

📊 Örnek Rapor
<p align="center"> <img src="docs/example_report.png" alt="ECL Report" width="600"/> </p>
📚 Yol Haritası

V1: Sandbox + Pipeline + Raporlama

V2: Streamlit arayüzü

V3: ECB/IMF makro senaryo entegrasyonu

    V4: Çok dillilik (EN + TR)

🤝 Katkı

Pull request gönderebilirsiniz. Büyük değişiklikler için önce bir “issue” açmanız önerilir.