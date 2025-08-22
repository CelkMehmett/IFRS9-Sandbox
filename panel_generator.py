# -*- coding: utf-8 -*-
"""
Bu dosya, gerçek dünya IFRS9 panel ve snapshot verisi üretmek için kullanılır.
- Makroekonomik faktörler, ürün tipleri, müşteri skorları ve temerrüt geçişleri içerir.
- Panel ve snapshot üretimi, banka seviyesinde analiz ve modelleme için uygundur.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Log dosyası ayarı
logging.basicConfig(filename='artifacts/panel_generator.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

macro_scenarios = {
    'baseline': {'gdp_growth': 0.03, 'unemployment': 0.08, 'stress_month': 12, 'stress_factor': 1.5},
    'mild_downturn': {'gdp_growth': 0.01, 'unemployment': 0.12, 'stress_month': 6, 'stress_factor': 1.8},
    'severe_downturn': {'gdp_growth': -0.02, 'unemployment': 0.18, 'stress_month': 3, 'stress_factor': 2.2}
}

def generate_ifrs9_panel(n_accounts=10000, n_months=24, scenario='baseline'):
    """
    Gerçekçi IFRS9 panel verisi üretir.
    Parametreler:
        n_accounts (int): Hesap sayısı
        n_months (int): Panel uzunluğu (ay)
    Dönüş:
        pd.DataFrame: Panel veri tablosu
    """
    try:
        """
        Gerçek dünya IFRS9 panel verisi üretir
        - Daha uzun gözlem periyodu (24 ay)
        - Daha fazla hesap sayısı (10,000)
        - Gerçekçi temerrüt oranları (2-5%)
        - Makroekonomik faktörler
        - Sektörel farklılıklar
        """
        np.random.seed(42)
        
        # Kredi türleri ve özellikleri
        products = {
            'Consumer Loan': {
                'weight': 0.4, 'base_pd': 0.03, 'ead_range': (5000, 50000),
                'eir_range': (0.18, 0.32), 'secured_prob': 0.1, 'ltv_range': (0.7, 1.0), 'ccf': 0.20
            },
            'Mortgage': {
                'weight': 0.3, 'base_pd': 0.015, 'ead_range': (100000, 500000),
                'eir_range': (0.10, 0.18), 'secured_prob': 0.98, 'ltv_range': (0.5, 0.9), 'ccf': 0.05
            },
            'Credit Card': {
                'weight': 0.2, 'base_pd': 0.05, 'ead_range': (1000, 25000),
                'eir_range': (0.30, 0.45), 'secured_prob': 0.0, 'ltv_range': (0.9, 1.2), 'ccf': 0.50
            },
            'Business Loan': {
                'weight': 0.1, 'base_pd': 0.04, 'ead_range': (50000, 1000000),
                'eir_range': (0.15, 0.28), 'secured_prob': 0.7, 'ltv_range': (0.4, 0.8), 'ccf': 0.30
            }
        }
        
        # Hesap bilgileri oluştur
        accounts = []
        account_id = 0
        
        # Müşteri segmentleri
        segments = ['Bireysel', 'Ticari', 'KOBİ']

        for product, props in products.items():
            n_prod = int(n_accounts * props['weight'])
            for _ in range(n_prod):
                # Segment seçimi
                segment = np.random.choice(segments, p=[0.6, 0.25, 0.15])
                # Başlangıç riski (segment ve ürün etkisi)
                seg_risk_adj = {'Bireysel': 1.0, 'Ticari': 1.2, 'KOBİ': 1.4}
                base_risk = np.random.normal(props['base_pd'] * seg_risk_adj[segment], props['base_pd'] * 0.3)
                base_risk = np.clip(base_risk, 0.005, 0.15)
                # EAD başlangıç değeri
                ead_start = np.random.uniform(*props['ead_range'])
                # Müşteri özellikleri
                customer_score = np.random.normal(650, 100)
                customer_score = np.clip(customer_score, 400, 850)
                age = np.random.normal(40, 12)
                age = np.clip(age, 18, 75)
                income_factor = np.random.lognormal(10.5, 0.5)
                # Finansal alanlar
                # EIR (yıllık efektif faiz)
                eir = np.random.uniform(*props.get('eir_range', (0.12, 0.30)))
                # Limit (revolving için daha yüksek)
                if product == 'Credit Card':
                    limit = ead_start * np.random.uniform(1.5, 3.0)
                else:
                    limit = ead_start * np.random.uniform(1.0, 1.2)
                # Teminat / LTV
                secured = int(np.random.random() < props.get('secured_prob', 0.0))
                ltv0 = 0.0
                collateral_value = 0.0
                if secured:
                    ltv0 = np.random.uniform(*props.get('ltv_range', (0.6, 0.9)))
                    collateral_value = max(ead_start / max(ltv0, 1e-6), ead_start)
                # Risk skoru (0-100): düşük skor = yüksek risk
                risk_score = float(np.clip(100 - (customer_score - 400) / 450 * 100, 0, 100))
                # Ürün bazlı CCF
                ccf_product = props.get('ccf', 0.20)
                accounts.append({
                    'account_id': account_id,
                    'product': product,
                    'segment': segment,
                    'base_pd': base_risk,
                    'ead_start': ead_start,
                    'customer_score': customer_score,
                    'age': age,
                    'income_factor': income_factor,
                    'eir': eir,
                    'limit': limit,
                    'secured': secured,
                    'collateral_value': collateral_value,
                    'ltv0': ltv0,
                    'risk_score': risk_score,
                    'ccf_product': ccf_product
                })
                account_id += 1
        
        # Panel veri oluştur
        data = []
        
        macro_base = macro_scenarios.get(scenario, macro_scenarios['baseline'])
        if hasattr(generate_ifrs9_panel, 'scenario'):
            scenario = generate_ifrs9_panel.scenario
        macro_base = macro_scenarios.get(scenario, macro_scenarios['baseline'])
        
        for account in accounts:
            current_state = "current"
            days_past_due = 0
            ead = account['ead_start']
            
            for month in range(1, n_months + 1):
                # Makroekonomik döngü (senaryoya göre)
                if month > macro_base['stress_month']:
                    macro_stress = macro_base['stress_factor']
                    gdp_growth = macro_base['gdp_growth'] - 0.02
                    unemployment = macro_base['unemployment'] + 0.03
                else:
                    macro_stress = 1.0
                    gdp_growth = macro_base['gdp_growth']
                    unemployment = macro_base['unemployment']
                
                # Risk faktörleri
                pd_monthly = account['base_pd'] / 12 * macro_stress
                
                # Customer score etkisi
                score_factor = (account['customer_score'] - 400) / 450
                pd_monthly *= (1.5 - score_factor)  # Düşük skor = yüksek risk
                
                # Yaş faktörü
                age_factor = 1.0
                if account['age'] < 25 or account['age'] > 65:
                    age_factor = 1.3  # Genç ve yaşlı daha riskli
                pd_monthly *= age_factor
                
                # EAD değişimi (kredi kartı için daha volatil)
                if account['product'] == 'Credit Card':
                    ead_change = np.random.normal(0, 0.15)
                    ead *= (1 + ead_change)
                    ead = max(ead, account['ead_start'] * 0.1)  # Minimum %10
                elif account['product'] == 'Mortgage':
                    ead *= 0.999  # Mortgage azalır
                else:
                    ead_change = np.random.normal(0, 0.05)
                    ead *= (1 + ead_change)
                    ead = max(ead, account['ead_start'] * 0.5)
                # LTV güncelle (varsa teminat)
                ltv = (ead / max(account.get('collateral_value', 0.0), 1e-6)) if account.get('secured', 0) == 1 else np.nan
                
                # Delinquency transition (gerçekçi geçiş matrisi)
                transition_prob = pd_monthly * 2.5  # Aylık temerrüt olasılığı
                
                if current_state == "current":
                    if np.random.random() < transition_prob:
                        current_state = "30dpd"
                        days_past_due = 35
                elif current_state == "30dpd":
                    if np.random.random() < 0.4:  # %40 şans ile 60dpd'ye geçer
                        current_state = "60dpd"
                        days_past_due = 65
                    elif np.random.random() < 0.3:  # %30 şans ile current'a döner
                        current_state = "current"
                        days_past_due = 0
                elif current_state == "60dpd":
                    if np.random.random() < 0.5:  # %50 şans ile 90dpd'ye geçer
                        current_state = "90dpd"
                        days_past_due = 95
                    elif np.random.random() < 0.2:  # %20 şans ile 30dpd'ye döner
                        current_state = "30dpd"
                        days_past_due = 35
                elif current_state == "90dpd":
                    if np.random.random() < 0.6:  # %60 şans ile default
                        current_state = "default"
                        days_past_due = 120
                    elif np.random.random() < 0.1:  # %10 şans ile 60dpd'ye döner
                        current_state = "60dpd"
                        days_past_due = 65
                elif current_state == "default":
                    # Default'tan çıkış çok nadir
                    if np.random.random() < 0.05:
                        current_state = "current"
                        days_past_due = 0
                
                # Seasoning etkisi (hesap yaşı arttıkça daha stabil)
                seasoning_factor = min(1.0, month / 24.0)
                pd12 = pd_monthly * 12 * (1.5 - seasoning_factor * 0.5)
                
                # LGD simülasyonu (ürün, segment, skor, makro stres etkisi)
                seg_lgd_adj = {'Bireysel': 1.0, 'Ticari': 0.8, 'KOBİ': 1.2}
                if account['product'] == 'Mortgage':
                    base_lgd = 0.2
                elif account['product'] == 'Business Loan':
                    base_lgd = 0.45
                elif account['product'] == 'Credit Card':
                    base_lgd = 0.65
                else:
                    base_lgd = 0.35
                lgd = base_lgd * seg_lgd_adj[account['segment']] * (1.1 - (account['customer_score']-400)/900) * macro_stress
                lgd = np.clip(lgd, 0.1, 0.95)

                # Davranışsal özellikler
                payment_ratio = np.random.beta(8, 2) if current_state == 'current' else np.random.beta(3, 5)
                limit_usage = np.random.uniform(0.2, 1.0) if account['product'] == 'Credit Card' else np.random.uniform(0.05, 0.8)
                new_application = int(np.random.random() < 0.02)
                
                # Erken uyarı sinyalleri (EWS)
                ews_late_payment = int(payment_ratio < 0.7 and days_past_due > 0)
                ews_limit_exceed = int(limit_usage > 0.9)
                ews_new_app = new_application
                ews_income_drop = int(np.random.random() < 0.01 if month > 1 and account['income_factor'] < account['income_factor'] else 0)
                data.append({
                    'account_id': account['account_id'],
                    'month': month,
                    'date': pd.to_datetime('2022-01-01') + pd.DateOffset(months=month-1),
                    'product': account['product'],
                    'segment': account['segment'],
                    'state': current_state,
                    'days_past_due': days_past_due,
                    'ead': ead,
                    'pd12': pd12,
                    'lgd': lgd,
                    'customer_score': account['customer_score'],
                    'age': account['age'],
                    'income_factor': account['income_factor'],
                    'eir': account['eir'],
                    'limit': account['limit'],
                    'secured': account['secured'],
                    'collateral_value': account['collateral_value'],
                    'ltv': ltv,
                    'risk_score': account['risk_score'],
                    'ccf_product': account['ccf_product'],
                    'gdp_growth': gdp_growth,
                    'unemployment': unemployment,
                    'macro_stress': macro_stress,
                    'payment_ratio': payment_ratio,
                    'limit_usage': limit_usage,
                    'new_application': new_application,
                    'ews_late_payment': ews_late_payment,
                    'ews_limit_exceed': ews_limit_exceed,
                    'ews_new_app': ews_new_app,
                    'ews_income_drop': ews_income_drop,
                })
        
        df = pd.DataFrame(data)
        print(f"✅ Gerçekçi panel veri üretildi: {len(df):,} satır, {n_accounts:,} hesap, {n_months} ay")
        logging.info(f"Panel üretildi: {len(df):,} satır, {n_accounts:,} hesap, {n_months} ay")
        print(f"📊 Ürün dağılımı: {df.groupby('product')['account_id'].nunique().to_dict()}")
        logging.info(f"Ürün dağılımı: {df.groupby('product')['account_id'].nunique().to_dict()}")
        print(f"⚠️  State dağılımı: {df['state'].value_counts().to_dict()}")
        logging.info(f"State dağılımı: {df['state'].value_counts().to_dict()}")
        return df
    except Exception as e:
        logging.error(f"Panel üretiminde hata: {str(e)}")
        print(f"❌ Panel üretiminde hata: {str(e)}")
        raise

def create_snapshot_from_panel(panel_df):
    """
    Panel verisinden point-in-time snapshot oluşturur.
    Parametreler:
        panel_df (pd.DataFrame): Panel veri tablosu
    Dönüş:
        pd.DataFrame: Snapshot veri tablosu
    """
    try:
        """
        Panel veriden point-in-time snapshot oluşturur (gerçek dünya yaklaşımı)
        """
        # Son gözlem zamanını al
        latest_month = panel_df['month'].max()
        
        # Her hesap için son 6 ayın performans metriklerini hesapla
        snapshot_data = []
        
        for account_id in panel_df['account_id'].unique():
            acc_data = panel_df[panel_df['account_id'] == account_id].sort_values('month')
            
            # Son 6 aylık veri (varsa)
            recent_data = acc_data[acc_data['month'] > latest_month - 6]
            last_row = acc_data[acc_data['month'] == latest_month].iloc[0]
            
            # Behavioral skorlama için özellikler
            months_observed = len(acc_data)
            
            # EAD istatistikleri
            ead_stats = {
                'ead_last': last_row['ead'],
                'ead_mean': recent_data['ead'].mean(),
                'ead_std': recent_data['ead'].std(),
                'ead_min': recent_data['ead'].min(),
                'ead_max': recent_data['ead'].max(),
                'ead_first': acc_data.iloc[0]['ead'],
                'ead_pct_change': ((last_row['ead'] - acc_data.iloc[0]['ead']) / 
                                  acc_data.iloc[0]['ead']) if acc_data.iloc[0]['ead'] > 0 else 0
            }
            # LGD istatistikleri
            lgd_stats = {
                'lgd_last': last_row['lgd'],
                'lgd_mean': recent_data['lgd'].mean(),
                'lgd_std': recent_data['lgd'].std(),
                'lgd_min': recent_data['lgd'].min(),
                'lgd_max': recent_data['lgd'].max(),
                'lgd_first': acc_data.iloc[0]['lgd'],
                'lgd_pct_change': ((last_row['lgd'] - acc_data.iloc[0]['lgd']) / acc_data.iloc[0]['lgd']) if acc_data.iloc[0]['lgd'] > 0 else 0
            }
            
            # PD trend analizi
            pd_stats = {
                'pd12_last': last_row['pd12'],
                'pd12_mean': recent_data['pd12'].mean(),
                'pd12_trend': recent_data['pd12'].pct_change().mean(),  # PD trend
                'pd12_volatility': recent_data['pd12'].std()
            }
            
            # Lifetime PD (forward-looking)
            # Seasoning ve ekonomik döngü etkisi
            seasoning_factor = min(1.0, months_observed / 24.0)
            macro_adjustment = last_row.get('macro_stress', 1.0)
            
            lifetime_pd = pd_stats['pd12_last'] * (1.5 - seasoning_factor * 0.3) * macro_adjustment
            lifetime_pd = min(lifetime_pd, 0.95)  # Maximum %95
            
            pd_stats.update({
                'lifetime_pd_last': lifetime_pd,
                'lifetime_pd_mean': lifetime_pd * 0.9  # Ortalama biraz daha düşük
            })
            
            # Delinquency geçmişi analizi
            states = acc_data['state'].values
            delinq_stats = {
                'sev_max': max([0 if s == 'current' else 1 if s == '30dpd' else 
                               2 if s == '60dpd' else 3 if s == '90dpd' else 
                               4 if s == 'default' else 0 for s in states]),
                'sev_mean': np.mean([0 if s == 'current' else 1 if s == '30dpd' else 
                                    2 if s == '60dpd' else 3 if s == '90dpd' else 
                                    4 if s == 'default' else 0 for s in states]),
                'delinquent_months': sum([1 for s in states if s != 'current']),
                'any30': any([s in ['30dpd', '60dpd', '90dpd', 'default'] for s in states]),
                'any60': any([s in ['60dpd', '90dpd', 'default'] for s in states]),
                'any90': any([s in ['90dpd', 'default'] for s in states]),
                'any_default': any([s == 'default' for s in states]),
                'any_closed': any([s == 'closed' for s in states])
            }
            
            # Target değişkeni (forward-looking 12 ay)
            target_default = 1 if last_row['state'] in ['90dpd', 'default'] else 0
            
            # IFRS9 Stage belirleme (gerçekçi staging)
            if target_default == 1:
                stage_last = 3  # Default
            elif delinq_stats['any30']:
                # SICR (Significant Increase in Credit Risk) kontrolü
                if (pd_stats['pd12_last'] > pd_stats['pd12_mean'] * 1.5 or  # Relatif artış
                    pd_stats['pd12_last'] > 0.05 or  # Absolut seviye
                    delinq_stats['any60']):  # 60+ DPD backstop
                    stage_last = 2  # SICR
                else:
                    stage_last = 1  # Normal
            else:
                stage_last = 1  # Normal
            
            # Müşteri demografik bilgileri
            customer_features = {
                'customer_score': last_row.get('customer_score', 650),
                'age': last_row.get('age', 40),
                'income_factor': last_row.get('income_factor', 1.0),
                'product': last_row['product']
            }
            
            # Makroekonomik faktörler
            macro_features = {
                'macro_gdp_growth': last_row.get('gdp_growth', 0.03),
                'macro_unemployment': last_row.get('unemployment', 0.08),
                'macro_stress': last_row.get('macro_stress', 1.0)
            }
            
            # Tüm özellikleri birleştir
            snapshot_row = {
                'account_id': account_id,
                'months_observed': months_observed,
                **ead_stats,
                **lgd_stats,
                **pd_stats,
                **delinq_stats,
                'target_default': target_default,
                'stage_last': stage_last,
                **customer_features,
                **macro_features,
                # Finansal özetler
                'limit': last_row.get('limit', np.nan),
                'eir': last_row.get('eir', np.nan),
                'secured': last_row.get('secured', 0),
                'collateral_value': last_row.get('collateral_value', 0.0),
                'ltv_last': last_row.get('ltv', np.nan),
                'risk_score': last_row.get('risk_score', np.nan),
                'ccf_product': last_row.get('ccf_product', np.nan),
                # Tahmini LGD (LTV etkisi ile düzeltme)
                'lgd_est': float(np.clip(
                    (last_row['lgd'] * (0.6 + 0.8 * float(last_row.get('ltv', np.nan)))) if last_row.get('secured', 0)==1 and pd.notnull(last_row.get('ltv', np.nan)) else last_row['lgd'],
                    0.05, 0.95
                )),
                # Seasoning metrik
                'seasoning_m': months_observed
            }
            
            snapshot_data.append(snapshot_row)
        
        snapshot_df = pd.DataFrame(snapshot_data)
        
        # Eksik değerleri doldur
        numeric_cols = snapshot_df.select_dtypes(include=[np.number]).columns
        snapshot_df[numeric_cols] = snapshot_df[numeric_cols].fillna(0)
        
        print(f"✅ Snapshot oluşturuldu: {len(snapshot_df):,} hesap")
        logging.info(f"Snapshot oluşturuldu: {len(snapshot_df):,} hesap")
        print(f"📊 Target distribution: {snapshot_df['target_default'].value_counts().to_dict()}")
        logging.info(f"Target distribution: {snapshot_df['target_default'].value_counts().to_dict()}")
        print(f"🎯 Default rate: {snapshot_df['target_default'].mean():.2%}")
        logging.info(f"Default rate: {snapshot_df['target_default'].mean():.2%}")
        print(f"📈 Stage distribution: {snapshot_df['stage_last'].value_counts().sort_index().to_dict()}")
        logging.info(f"Stage distribution: {snapshot_df['stage_last'].value_counts().sort_index().to_dict()}")
        return snapshot_df
    except Exception as e:
        logging.error(f"Snapshot üretiminde hata: {str(e)}")
        print(f"❌ Snapshot üretiminde hata: {str(e)}")
        raise

def generate_portfolio(n=10000, months=24, scenario="baseline"):
    """
    Pipeline için uyumlu panel ve snapshot üretir.
    Parametreler:
        n (int): Hesap sayısı
        months (int): Panel uzunluğu
        scenario (str): Makro senaryo
    Dönüş:
        tuple: (snapshot, panel)
    """
    panel = generate_ifrs9_panel(n, months, scenario)
    snapshot = create_snapshot_from_panel(panel)
    return snapshot, panel

if __name__ == "__main__":
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Gerçek dünya simülasyonu
    panel = generate_ifrs9_panel(n_accounts=10000, n_months=24, scenario="baseline")
    snapshot = create_snapshot_from_panel(panel)
    
    panel.to_parquet(out_dir/"ifrs9_panel.parquet", index=False)
    snapshot.to_parquet(out_dir/"ifrs9_snapshot.parquet", index=False)
    
    print("\n🎯 Gerçek dünya IFRS9 verisi başarıyla üretildi!")
