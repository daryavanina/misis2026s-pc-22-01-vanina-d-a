"""
plot_lab2.py — графики для лабораторной работы №2
=========================================================
Запуск:   python plot_lab2.py
Вход:     lab2_proc1.json  (и опционально lab2_proc2.json, lab2_proc3.json)
Выход:    папка plots_lab2/

Когда получишь данные с других процессоров — просто добавь файл
lab2_proc2.json / lab2_proc3.json в ту же папку и запусти снова.
Структура файла описана в конце скрипта.
"""

import json, os, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = 'plots_lab2'
DPI     = 150
N_FIXED = 512
os.makedirs(OUT_DIR, exist_ok=True)

# ── загрузка данных ───────────────────────────────────────────────────────────
proc_files = sorted(glob.glob('lab2_proc*.json'))
if not proc_files:
    raise FileNotFoundError("Не найден ни один файл lab2_proc*.json")

procs = []
for fname in proc_files:
    with open(fname, 'r', encoding='utf-8') as f:
        procs.append(json.load(f))

print(f"Загружено процессоров: {len(procs)}")
for p in procs:
    print(f"  {p['proc_name']}")

# ── палитра ───────────────────────────────────────────────────────────────────
COLORS  = ['#4472C4', '#ED7D31', '#70AD47', '#C00000', '#7030A0']
MARKERS = ['o', 's', '^', 'D', 'v']
ALGO_COLORS = {
    'classic':   '#4472C4',
    'transpose': '#ED7D31',
    'buffered':  '#70AD47',
    'block':     '#C00000',
}
ALGO_LABELS = {
    'classic':   'Классическое',
    'transpose': 'Транспонирование (без T)',
    'buffered':  'Буферизация столбца',
    'block':     'Блочное',
}

def gflops(n, ms):
    return 2.0 * n**3 / (ms * 1e-3) / 1e9

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  сохранён: {path}")

def best_idx(values):
    return values.index(max(values))

# ── Рис. 1 — Debug vs Release ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(6, len(procs) * 2.5), 4.5))
names    = [p['proc_name'] for p in procs]
dbg_vals = [p['debug_classic_ms'] for p in procs]
rel_vals = [p['release']['classic_ms'] for p in procs]
x = np.arange(len(procs))
w = 0.35

b1 = ax.bar(x - w/2, dbg_vals, w, label='Debug (-O0)',   color='#4472C4', alpha=0.85)
b2 = ax.bar(x + w/2, rel_vals,  w, label='Release (-O2)', color='#70AD47', alpha=0.85)

for bar, v in list(zip(b1, dbg_vals)) + list(zip(b2, rel_vals)):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + max(dbg_vals) * 0.01,
            f'{v:.1f}', ha='center', va='bottom', fontsize=9)

for i, (d_, r_) in enumerate(zip(dbg_vals, rel_vals)):
    ax.text(x[i], max(d_, r_) + max(dbg_vals) * 0.06,
            f'×{d_/r_:.2f}', ha='center', fontsize=11,
            color='#C00000', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel('Время, мс', fontsize=11)
ax.set_title(f'Рис. 1 — Классическое умножение: Debug vs Release (N={N_FIXED})',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.4)
save(fig, 'fig1_debug_vs_release.png')

# ── Рис. 2 — Preal всех алгоритмов ───────────────────────────────────────────
algo_keys   = ['classic_ms','transpose_ms','transpose_noT_ms','buffered_ms','block_ms']
algo_labels = ['Классическое','Транспонир.\n(с T)','Транспонир.\n(без T)',
               'Буферизация\nстолбца','Блочное']
algo_clrs   = ['#4472C4','#9DC3E6','#ED7D31','#70AD47','#C00000']

fig, ax = plt.subplots(figsize=(max(8, len(procs) * 3), 5))
x   = np.arange(len(procs))
n_a = len(algo_keys)
w   = min(0.15, 0.7 / n_a)

for idx, (key, label, color) in enumerate(zip(algo_keys, algo_labels, algo_clrs)):
    vals = [gflops(N_FIXED, p['release'][key]) for p in procs]
    offset = (idx - n_a / 2 + 0.5) * w
    bars = ax.bar(x + offset, vals, w, label=label, color=color, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7.5, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel('Preal, GFLOP/s', fontsize=11)
ax.set_title(f'Рис. 2 — Производительность алгоритмов (N={N_FIXED}, S=64, M=4)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.4)
save(fig, 'fig2_algos_preal.png')

# ── Рис. 3 — Буферизация: Preal(M) ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, p in enumerate(procs):
    M_vals = p['release']['buffered_M_values']
    times  = p['release']['buffered_M_times_ms']
    preal  = [gflops(N_FIXED, t) for t in times]
    ax.plot(M_vals, preal, marker=MARKERS[i], color=COLORS[i],
            linewidth=2, markersize=8, label=p['proc_name'])
    bi = best_idx(preal)
    ax.annotate(f"M*={M_vals[bi]}\n{preal[bi]:.3f} GFLOP/s",
                xy=(M_vals[bi], preal[bi]), xytext=(6, -22),
                textcoords='offset points', fontsize=8.5,
                color=COLORS[i], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS[i], lw=1.2))

ax.set_xscale('log', base=2)
ax.set_xticks(M_vals); ax.set_xticklabels(M_vals)
ax.set_xlabel('Степень раскрутки M', fontsize=11)
ax.set_ylabel('Preal, GFLOP/s', fontsize=11)
ax.set_title(f'Рис. 3 — Буферизация столбца: Preal(M), N={N_FIXED}',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(linestyle='--', alpha=0.4)
save(fig, 'fig3_buffered_M.png')

# ── Рис. 4 — Блочное: Preal(S) ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, p in enumerate(procs):
    S_vals = p['release']['block_S_values']
    times  = p['release']['block_S_times_ms']
    preal  = [gflops(N_FIXED, t) for t in times]
    ax.plot(S_vals, preal, marker=MARKERS[i], color=COLORS[i],
            linewidth=2, markersize=8, label=p['proc_name'])
    bi = best_idx(preal)
    ax.annotate(f"S*={S_vals[bi]}\n{preal[bi]:.3f} GFLOP/s",
                xy=(S_vals[bi], preal[bi]), xytext=(6, 6),
                textcoords='offset points', fontsize=8.5,
                color=COLORS[i], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS[i], lw=1.2))

ax.set_xscale('log', base=2)
ax.set_xticks(S_vals); ax.set_xticklabels(S_vals)
ax.set_xlabel('Размер блока S', fontsize=11)
ax.set_ylabel('Preal, GFLOP/s', fontsize=11)
ax.set_title(f'Рис. 4 — Блочное умножение: Preal(S), N={N_FIXED}, M=4',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(linestyle='--', alpha=0.4)
save(fig, 'fig4_block_S.png')

# ── Рис. 5 — Блочное: Preal(M) ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, p in enumerate(procs):
    M_vals = p['release']['block_M_values']
    times  = p['release']['block_M_times_ms']
    preal  = [gflops(N_FIXED, t) for t in times]
    ax.plot(M_vals, preal, marker=MARKERS[i], color=COLORS[i],
            linewidth=2, markersize=8, label=p['proc_name'])
    bi = best_idx(preal)
    ax.annotate(f"M*={M_vals[bi]}\n{preal[bi]:.3f} GFLOP/s",
                xy=(M_vals[bi], preal[bi]), xytext=(6, 6),
                textcoords='offset points', fontsize=8.5,
                color=COLORS[i], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS[i], lw=1.2))

ax.set_xscale('log', base=2)
ax.set_xticks(M_vals); ax.set_xticklabels(M_vals)
ax.set_xlabel('Степень раскрутки M', fontsize=11)
ax.set_ylabel('Preal, GFLOP/s', fontsize=11)
ax.set_title(f'Рис. 5 — Блочное умножение: Preal(M), N={N_FIXED}, S=64',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(linestyle='--', alpha=0.4)
save(fig, 'fig5_block_M.png')

# ── Рис. 6+ — Preal(N) для каждого процессора ────────────────────────────────
for fig_num, p in enumerate(procs, start=6):
    rel  = p['release']
    Nv   = rel['sweep_N_values']
    fig2, ax2 = plt.subplots(figsize=(8, 4.8))

    for algo in ['classic', 'transpose', 'buffered', 'block']:
        times = rel[f'sweep_{algo}_ms']
        preal = [gflops(n, t) for n, t in zip(Nv, times)]
        ax2.plot(Nv, preal, marker='o', color=ALGO_COLORS[algo],
                 linewidth=2, markersize=6, label=ALGO_LABELS[algo])
        ax2.annotate(f'{preal[-1]:.3f}',
                     xy=(Nv[-1], preal[-1]), xytext=(4, 0),
                     textcoords='offset points', fontsize=8,
                     color=ALGO_COLORS[algo])

    ax2.set_xscale('log', base=2)
    ax2.set_xticks(Nv); ax2.set_xticklabels(Nv, rotation=30)
    ax2.set_xlabel('Размер матрицы N', fontsize=11)
    ax2.set_ylabel('Preal, GFLOP/s', fontsize=11)
    ax2.set_title(f'Рис. {fig_num} — Preal(N), {p["proc_name"]} (S=64, M=8)',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10); ax2.grid(linestyle='--', alpha=0.4)
    safe = (p['proc_name']
            .replace(' ', '_').replace('(','').replace(')','')
            .replace('/','_').replace('\\','_'))
    save(fig2, f'fig{fig_num}_preal_N_{safe}.png')

total = len(os.listdir(OUT_DIR))
print(f'\nВсего сохранено графиков: {total}')
print(f'Папка: {os.path.abspath(OUT_DIR)}')
