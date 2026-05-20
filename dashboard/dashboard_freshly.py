import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.stats import gaussian_kde

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(page_title="Freshly Dashboard", layout="wide")
sns.set_style("whitegrid")

# =========================================
# PALETTE WARNA (sama dengan notebook)
# =========================================
PALETTE = {
    'ripe': '#f59e0b',    # orange
    'unripe': '#22c55e',  # green
    'rotten': '#7f1d1d'   # dark red
}

# =========================================
# LOAD SEMUA FILE CSV
# =========================================
@st.cache_data
def load_all_csv():
    data = {}
    
    # BQ1
    data['distribusi'] = pd.read_csv("distribusi_kematangan.csv")
    data['class_composition'] = pd.read_csv("class_composition_percent.csv")
    data['imbalance_ratio'] = pd.read_csv("imbalance_ratio.csv")
    
    # BQ2
    data['color_analysis'] = pd.read_csv("color_analysis.csv")
    
    # BQ3
    data['overlap_scores'] = pd.read_csv("overlap_score_pairs.csv")
    data['overlap_visual'] = pd.read_csv("overlap_visualization_dataset.csv")
    
    # BQ4
    data['fisher_scores'] = pd.read_csv("fisher_separability_score.csv")
    data['scatter_features'] = pd.read_csv("scatter_visual_features.csv")
    
    # BQ5
    data['cleaning_summary'] = pd.read_csv("dataset_cleaning_summary.csv")
    data['elimination_per_class'] = pd.read_csv("data_elimination_per_class.csv")
    data['duplicate_per_class'] = pd.read_csv("duplicate_per_class.csv")
    data['distribution_before_after'] = pd.read_csv("dataset_distribution_before_after_cleaning.csv")
    
    return data

data = load_all_csv()

# =========================================
# METRIKS UTAMA (dari cleaning_summary)
# =========================================
cleaning = data['cleaning_summary']
total_raw = cleaning.loc[cleaning['category'] == 'Data Awal', 'count'].values[0]
total_clean = cleaning.loc[cleaning['category'] == 'Dataset Bersih', 'count'].values[0]
total_elim = total_raw - total_clean
elim_pct = (total_elim / total_raw) * 100

# =========================================
# TITLE
# =========================================
st.title("Freshly Dashboard")
st.caption("Fruit & Vegetable Ripeness Analysis")

col1, col2, col3, col4 = st.columns(4)
col1.metric("📸 Total Gambar Awal", f"{total_raw:,}")
col2.metric("✨ Dataset Bersih", f"{total_clean:,}", delta=f"-{total_elim:,}")
col3.metric("🗑️ Total Dihapus", f"{total_elim:,}")
col4.metric("📉 Eliminasi", f"{elim_pct:.2f}%")

# =========================================
# BQ1: DISTRIBUSI KONDISI KEMATANGAN
# =========================================
st.header("BQ-1: Distribusi Kondisi Kematangan per Item")
tab1, tab2, tab3 = st.tabs(["Bar Plot Distribusi", "Stacked Persentase", "Imbalance Ratio"])

with tab1:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=data['distribusi'], x='item', y='count', hue='condition', palette=PALETTE, ax=ax)
    ax.set_xlabel(""); ax.set_ylabel("Jumlah Gambar")
    ax.tick_params(axis='x', rotation=30)
    st.pyplot(fig)

with tab2:
    # Stacked bar horizontal dari class_composition_percent.csv
    # File ini berisi pivot table (item, ripe, unripe, rotten) dalam persen
    comp = data['class_composition']
    fig, ax = plt.subplots(figsize=(10, 4))
    bottom = np.zeros(len(comp))
    for cond in ['ripe', 'unripe', 'rotten']:
        if cond in comp.columns:
            vals = comp[cond].values
            ax.barh(comp.index, vals, left=bottom, color=PALETTE[cond], label=cond, height=0.7)
            bottom += vals
    ax.set_xlim(0, 100)
    ax.set_xlabel("Persentase (%)")
    ax.set_ylabel("Item")
    ax.legend()
    st.pyplot(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(10, 5))
    ir = data['imbalance_ratio'].sort_values('imbalance_ratio', ascending=False)
    bars = ax.barh(ir['item'], ir['imbalance_ratio'], color='steelblue')
    ax.axvline(1, color='green', linestyle='--')
    ax.axvline(2, color='orange', linestyle='--')
    ax.axvline(3, color='red', linestyle='--')
    ax.set_xlabel("Imbalance Ratio")
    st.pyplot(fig)

# =========================================
# BQ2: POLA WARNA RGB & GR RATIO
# =========================================
st.header("BQ-2: Pola Warna (RGB & Green/Red Ratio)")
tab4, tab5, tab6 = st.tabs(["Boxplot R, G, GR", "Mean RGB per Kondisi", "GR Ratio Heatmap & Violin"])

with tab4:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, (col, title) in enumerate([('red', 'Red Channel'), ('green', 'Green Channel'), ('gr_ratio', 'GR Ratio')]):
        sns.boxplot(data=data['color_analysis'], x='item', y=col, hue='condition', palette=PALETTE, ax=axes[idx])
        axes[idx].set_title(title)
        axes[idx].set_xlabel("")
        axes[idx].tick_params(axis='x', rotation=45)
        if col == 'gr_ratio':
            axes[idx].axhline(1.0, color='gray', linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)

with tab5:
    # Mean RGB per kondisi (global)
    rgb_mean = data['color_analysis'].groupby('condition')[['red','green','blue']].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(3)
    width = 0.25
    for i, cond in enumerate(['ripe', 'unripe', 'rotten']):
        if cond in rgb_mean.index:
            vals = rgb_mean.loc[cond]
            ax.bar(x + i*width, [vals['red'], vals['green'], vals['blue']], width, label=cond, color=PALETTE[cond])
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Red', 'Green', 'Blue'])
    ax.set_ylabel("Mean Pixel Value")
    ax.legend()
    st.pyplot(fig)

with tab6:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Violin plot GR ratio
    sns.violinplot(data=data['color_analysis'], x='condition', y='gr_ratio', palette=PALETTE, ax=axes[0])
    axes[0].axhline(1.0, color='gray', linestyle='--')
    axes[0].set_title("Distribusi GR Ratio per Kondisi")
    # Heatmap GR ratio per item
    gr_heat = data['color_analysis'].groupby(['item','condition'])['gr_ratio'].mean().unstack()
    sns.heatmap(gr_heat, annot=True, fmt=".2f", cmap="RdYlGn", center=1.0, ax=axes[1])
    axes[1].set_title("Rata-rata GR Ratio per Item")
    plt.tight_layout()
    st.pyplot(fig)

# =========================================
# BQ3: OVERLAP FITUR WARNA ANTAR KELAS MIRIP
# =========================================
st.header("BQ-3: Overlap Fitur Warna (Kelas yang Mirip)")
tab7, tab8 = st.tabs(["Scatter Plot & KDE", "Overlap Score"])

with tab7:
    # Ada 3 pasangan: mango_ripe vs orange_ripe, mango_unripe vs tomato_unripe, paprika_ripe vs banana_unripe
    # Data overlap_visualization_dataset.csv berisi kolom: pair, group, red, green, gr_ratio, blue, condition
    ov = data['overlap_visual']
    pairs = ov['pair'].unique()
    fig, axes = plt.subplots(len(pairs), 3, figsize=(15, 4*len(pairs)))
    if len(pairs) == 1:
        axes = [axes]
    for i, pair in enumerate(pairs):
        sub = ov[ov['pair'] == pair]
        # Scatter R vs G
        for grp, color in zip(['A','B'], ['#f59e0b', '#22c55e']):
            d = sub[sub['group'] == grp]
            axes[i][0].scatter(d['red'], d['green'], alpha=0.5, color=color, label=grp)
        axes[i][0].set_title(f"{pair} - R vs G")
        axes[i][0].legend()
        # Scatter GR vs Blue
        for grp, color in zip(['A','B'], ['#f59e0b', '#22c55e']):
            d = sub[sub['group'] == grp]
            axes[i][1].scatter(d['gr_ratio'], d['blue'], alpha=0.5, color=color)
        axes[i][1].axvline(1.0, color='gray', linestyle='--')
        axes[i][1].set_title("GR Ratio vs Blue")
        # KDE GR ratio
        for grp, color in zip(['A','B'], ['#f59e0b', '#22c55e']):
            vals = sub[sub['group'] == grp]['gr_ratio'].dropna()
            if len(vals) > 1:
                kde = gaussian_kde(vals)
                x = np.linspace(vals.min(), vals.max(), 200)
                axes[i][2].plot(x, kde(x), color=color)
                axes[i][2].fill_between(x, kde(x), alpha=0.2, color=color)
        axes[i][2].axvline(1.0, color='gray', linestyle='--')
        axes[i][2].set_title("KDE GR Ratio")
    plt.tight_layout()
    st.pyplot(fig)

with tab8:
    fig, ax = plt.subplots(figsize=(10, 5))
    ov_score = data['overlap_scores'].sort_values('overlap', ascending=False)
    colors = [PALETTE['rotten'] if lvl=='High' else PALETTE['ripe'] if lvl=='Medium' else PALETTE['unripe'] 
              for lvl in ov_score['level']]
    ax.barh(ov_score['pair'], ov_score['overlap'], color=colors)
    ax.axvline(0.3, color='orange', linestyle='--', label='Medium (0.3)')
    ax.axvline(0.6, color='red', linestyle='--', label='High (0.6)')
    ax.set_xlim(0,1)
    ax.set_xlabel("Overlap Score")
    ax.legend()
    st.pyplot(fig)

# =========================================
# BQ4: SEPARABILITAS FITUR (HSV vs LBP)
# =========================================
st.header("BQ-4: Separabilitas Fitur HSV dan LBP")
tab9, tab10 = st.tabs(["Fisher Separability Score", "Scatter Plot Fitur"])

with tab9:
    fig, ax = plt.subplots(figsize=(8, 6))
    fisher = data['fisher_scores']
    if 'Feature' in fisher.columns:
        fisher = fisher.set_index('Feature')
    sns.heatmap(fisher, annot=True, fmt=".3f", cmap="YlGn", ax=ax)
    st.pyplot(fig)

with tab10:
    # Gunakan scatter_visual_features.csv
    sc = data['scatter_features']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # GR ratio vs Hue mean
    for cond, color in PALETTE.items():
        sub = sc[sc['condition'] == cond]
        axes[0].scatter(sub['gr_ratio'], sub['hue_mean'], alpha=0.4, color=color, label=cond)
    axes[0].axvline(1.0, color='gray', linestyle='--')
    axes[0].set_xlabel("GR Ratio")
    axes[0].set_ylabel("Hue Mean")
    axes[0].legend()
    # Saturation vs LBP std
    for cond, color in PALETTE.items():
        sub = sc[sc['condition'] == cond]
        axes[1].scatter(sub['sat_mean'], sub['lbp_std'], alpha=0.4, color=color, label=cond)
    axes[1].set_xlabel("Saturation Mean")
    axes[1].set_ylabel("LBP Std (Texture Complexity)")
    axes[1].legend()
    plt.tight_layout()
    st.pyplot(fig)

# =========================================
# BQ5: DAMPAK CLEANING DATA
# =========================================
st.header("🧹 BQ5: Dampak Pembersihan Data")
tab11, tab12, tab13 = st.tabs(["Ringkasan Cleaning", "Duplikat per Kelas", "Distribusi Sebelum/Sesudah"])

with tab11:
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(cleaning['category'], cleaning['count'], color=['#94a3b8', PALETTE['ripe'], PALETTE['rotten'], PALETTE['unripe']])
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{int(bar.get_height()):,}", ha='center', va='bottom')
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        elim = data['elimination_per_class'].sort_values('elim_pct', ascending=False)
        colors_elim = [PALETTE['rotten'] if p>10 else PALETTE['ripe'] if p>5 else PALETTE['unripe'] for p in elim['elim_pct']]
        ax.barh(elim['label'], elim['elim_pct'], color=colors_elim)
        ax.axvline(5, color='orange', linestyle='--')
        ax.axvline(10, color='red', linestyle='--')
        ax.set_xlabel("Elimination (%)")
        st.pyplot(fig)

with tab12:
    # Duplikat per kelas (dari duplicate_per_class.csv)
    dup = data['duplicate_per_class']
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=dup, y='label', x='duplicate_count', hue='category_type', palette='Set2', ax=ax)
    ax.set_xlabel("Jumlah Duplikat")
    st.pyplot(fig)

with tab13:
    df_before_after = data['distribution_before_after'].copy()
    # Normalisasi nama kolom
    rename_map = {}
    for col in df_before_after.columns:
        if col.lower() in ['item', 'items']:
            rename_map[col] = 'item'
        elif col.lower() in ['condition', 'conditions', 'status']:
            rename_map[col] = 'condition'
        elif col.lower() in ['count', 'jumlah', 'total']:
            rename_map[col] = 'count'
        elif col.lower() in ['source', 'jenis', 'category']:
            rename_map[col] = 'source'
        elif col.lower() in ['stage', 'tahap']:
            rename_map[col] = 'stage'
    df_before_after = df_before_after.rename(columns=rename_map)
    
    # Pastikan kolom yang diperlukan ada
    required = ['item', 'condition', 'count', 'source', 'stage']
    missing = [c for c in required if c not in df_before_after.columns]
    if missing:
        st.error(f"Kolom yang hilang: {missing}. Periksa file CSV.")
    else:
        fruits = df_before_after[df_before_after['source'] == 'fruit']
        vegs = df_before_after[df_before_after['source'] == 'vegetable']
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        configs = [
            (axes[0,0], "Fruits - Awal", fruits[fruits['stage'] == 'before_cleaning']),
            (axes[0,1], "Vegetables - Awal", vegs[vegs['stage'] == 'before_cleaning']),
            (axes[1,0], "Fruits - Setelah Cleaning", fruits[fruits['stage'] == 'after_cleaning']),
            (axes[1,1], "Vegetables - Setelah Cleaning", vegs[vegs['stage'] == 'after_cleaning'])
        ]
        for ax, title, df_sub in configs:
            if df_sub.empty:
                ax.text(0.5, 0.5, f"Data {title} kosong", ha='center', va='center')
                ax.set_title(title)
            else:
                sns.barplot(data=df_sub, x='item', y='count', hue='condition', palette=PALETTE, ax=ax)
                ax.set_title(title)
                ax.set_xlabel("")
                ax.tick_params(axis='x', rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

# =========================================
# FOOTER
# =========================================
st.caption("Freshly Dashboard © 2026 — Berdasarkan Capstone Project CC26-PSU059")