from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

APP = Path(__file__).resolve().parent
DATA = APP/'data'
ASSETS = APP/'assets'

st.set_page_config(page_title='LoRA-Boost', layout='wide')
st.markdown('<style>.block-container{max-width:1050px;padding-top:2rem}</style>', unsafe_allow_html=True)

COLORS = {'A': '#888888', 'B': '#5e7259', 'C': '#9a7a35', 'D': '#b1502b'}
COND_NAME = {'A': 'A · baseline', 'B': 'B · traditional aug',
             'C': 'C · FLUX zero-shot', 'D': 'D · LoRA-Boost (ours)'}

@st.cache_data
def load(name):
    df = pd.read_csv(DATA/name)
    if 'species_id' in df.columns:
        df['species_id'] = df['species_id'].astype(str)
    return df

summary  = load('results_summary.csv')
per_sp   = load('per_species.csv')
longtail = load('longtail.csv')
rare     = load('rare_species.csv')
samples  = load('samples_index.csv')
gallery  = load('gallery_index.csv')

def gamma_label(r):
    if str(r['loss']) == 'ce':
        return 'CE (γ=0)'
    return f"focal γ={int(r['gamma'])}" if pd.notna(r['gamma']) else 'focal'

S = summary.copy()
S['glabel'] = S.apply(gamma_label, axis=1)
S['gnum'] = S.apply(lambda r: 0 if str(r['loss']) == 'ce' else r['gamma'], axis=1)
main = S[S.es_monitor == 'overall_f1']
DEFAULT_EXP = main[(main.k == 1) & (main.gnum == 2) & (main.loss == 'focal')].exp_id.iloc[0]

def cond_table(exp):
    return main[main.exp_id == exp].set_index('condition')

def style_fig(fig, h=380):
    fig.update_layout(height=h, template='plotly_white', margin=dict(t=40, l=10, r=10, b=10),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

dft = cond_table(DEFAULT_EXP)
d_rec, a_rec = dft.loc['D', 'rare_recall_mean'], dft.loc['A', 'rare_recall_mean']

st.title('LoRA-Boost')
st.caption('Generative augmentation for long-tail plant species classification · Pl@ntNet-300K')
st.markdown(f'**สรุป** — เทรน LoRA ต่อ species เพื่อเจนภาพเติม rare species ช่วยดัน recall ขึ้น '
            f'{(d_rec-a_rec):+.3f} เทียบ baseline โดย overall macro F1 ไม่ตก และชนะการเจนแบบ zero-shot ทุกครั้ง')

st.divider()
st.header('The long-tail problem')
st.write('Pl@ntNet-300K มี 399 species แต่จำนวนภาพต่อ species เหลื่อมล้ำกันมหาศาล '
         '15 species ที่หายากที่สุดมีภาพ 30–99 ใบ จมอยู่ปลายหาง classifier จึงเรียนกลุ่มนี้ได้แย่ '
         'เป้าหมายเราคือยกกลุ่มนี้ขึ้นด้วยภาพสังเคราะห์')
d = longtail.sort_values('n_train', ascending=False).reset_index(drop=True)
d['rank'] = range(1, len(d) + 1)
rr = d[d.is_rare]
fig = go.Figure()
fig.add_trace(go.Scatter(x=d['rank'], y=d['n_train'], mode='lines',
                         line=dict(color='#cccccc'), fill='tozeroy',
                         name='all 399 species', hoverinfo='skip'))
fig.add_trace(go.Scatter(x=rr['rank'], y=rr['n_train'], mode='markers',
                         marker=dict(color='#b1502b', size=9, line=dict(color='white', width=1)),
                         text=rr['scientific_name'], name='15 rare species',
                         hovertemplate='%{text}<br>%{y} train imgs<extra></extra>'))
fig.update_yaxes(type='log', title='train images per species (log scale)')
fig.update_xaxes(title='species rank (most → least images)')
st.plotly_chart(style_fig(fig, 420), use_container_width=True)

st.markdown('**Controlled subset — what the experiment trains on**')
C, R = float(main['C'].iloc[0]), float(main['R'].iloc[0])
kept = longtail[longtail.n_train <= C].copy()
kept['eff'] = np.where(kept.is_rare, kept.n_train, np.minimum(kept.n_train, R))
kept = kept.sort_values('eff', ascending=False).reset_index(drop=True)
kept['rank'] = range(1, len(kept) + 1)
kr = kept[kept.is_rare]
figk = go.Figure()
figk.add_trace(go.Scatter(x=kept['rank'], y=kept['eff'], mode='lines',
                          line=dict(color='#cccccc'), fill='tozeroy',
                          name=f'kept subset ({len(kept)} classes)', hoverinfo='skip'))
figk.add_trace(go.Scatter(x=kr['rank'], y=kr['eff'], mode='markers',
                          marker=dict(color='#b1502b', size=9, line=dict(color='white', width=1)),
                          text=kr['scientific_name'], name='15 rare species',
                          hovertemplate='%{text}<br>%{y} train imgs<extra></extra>'))
figk.add_hline(y=C, line_dash='dash', line_color='#888', annotation_text=f'C = {C:.0f} (cut head)', annotation_position='top right')
figk.add_hline(y=R, line_dash='dash', line_color='#5e7259', annotation_text=f'R = {R:.0f} (cap)', annotation_position='bottom right')
figk.update_yaxes(title='train images per species')
figk.update_xaxes(title='kept species rank')
st.plotly_chart(style_fig(figk, 360), use_container_width=True)
st.caption(f'ตัดหัว (class ที่ n_train > {C:.0f}) ออก เหลือ {len(kept)} class แล้ว cap non-rare ที่เกิน {R:.0f} ลงมา '
           f'ทำให้ long-tail แคบลง ศึกษาได้แฟร์ขึ้น · rare 15 ตัว = จุดสีแดง')

st.markdown('**15 rare species — augmentation target**')
st.dataframe(rare.sort_values('n_train'), use_container_width=True, hide_index=True)

st.divider()
st.header('How LoRA-Boost works')
st.write('เทรน LoRA แยกต่อ species บน FLUX.2-klein (DreamBooth) แล้วเจนภาพมาเติม rare species '
         'ตอนเทรน classifier ต่างจาก zero-shot ตรงที่ LoRA ได้เห็นรูปจริงของ species นั้นก่อน')
PIPE = [('NB01', 'clean + split 399 species'), ('NB02', 'train 15 LoRA'),
        ('NB03', 'generate synthetic'), ('NB04', 'train ResNet-50'),
        ('NB05', 'error analysis'), ('NB06', 'budget experiment')]
box = ('<div style="flex:1;min-width:120px;background:#f7f7f7;border:1px solid #e0e0e0;'
       'border-radius:8px;padding:10px 12px"><div style="font-weight:600">{t}</div>'
       '<div style="font-size:12px;color:#777">{d}</div></div>')
arrow = '<div style="align-self:center;color:#bbb;font-size:18px">→</div>'
st.markdown(f'<div style="display:flex;gap:6px;flex-wrap:wrap;align-items:stretch;margin:6px 0 18px">'
            f'{arrow.join(box.format(t=t, d=de) for t, de in PIPE)}</div>', unsafe_allow_html=True)

st.markdown('**Real vs zero-shot vs LoRA-Boost**')
opt = {r.scientific_name: r.species_id for r in gallery.itertuples()}
sid = opt[st.selectbox('species', list(opt), key='gal')]
for col, tag, label in zip(st.columns(3), ['real', 'zeroshot', 'lora'],
                           ['Real (Pl@ntNet)', 'FLUX zero-shot', 'LoRA-Boost (ours)']):
    col.markdown(f'**{label}**')
    for im in sorted((ASSETS/'gallery'/sid/tag).glob('*.jpg')):
        col.image(str(im), use_container_width=True)
st.caption('LoRA weights ทั้ง 15 species: (วาง link HuggingFace ตรงนี้)')

st.divider()
st.header('Does it work?')
METRICS = {'rare_recall': 'rare-15 recall (primary)', 'rare_f1': 'rare-15 macro F1',
           'rare_precision': 'rare-15 precision', 'overall_f1': 'overall macro F1'}
c1, c2 = st.columns([3, 1])
metric = c1.radio('metric', list(METRICS), format_func=lambda m: METRICS[m], horizontal=True)
cfgs = main[['exp_id', 'k', 'gnum', 'glabel']].drop_duplicates().sort_values(['k', 'gnum'])
cfg_labels = {f'k={r.k} · {r.glabel}': r.exp_id for r in cfgs.itertuples()}
keys = list(cfg_labels)
default = next((i for i, key in enumerate(keys) if 'k=1.0' in key and 'γ=2' in key), 0)
exp = cfg_labels[c2.selectbox('config', keys, index=default)]
dd = cond_table(exp)
conds = ['A', 'B', 'C', 'D']

m1, m2, m3, m4 = st.columns(4)
m1.metric('rare recall (D)', f"{dd.loc['D','rare_recall_mean']:.3f}", f"{dd.loc['D','rare_recall_mean']-dd.loc['A','rare_recall_mean']:+.3f} vs A")
m2.metric('rare macro F1 (D)', f"{dd.loc['D','rare_f1_mean']:.3f}", f"{dd.loc['D','rare_f1_mean']-dd.loc['A','rare_f1_mean']:+.3f} vs A")
m3.metric('overall macro F1 (D)', f"{dd.loc['D','overall_f1_mean']:.3f}", f"{dd.loc['D','overall_f1_mean']-dd.loc['A','overall_f1_mean']:+.3f} vs A")
m4.metric('recall vs zero-shot', f"{dd.loc['D','rare_recall_mean']:.3f}", f"{dd.loc['D','rare_recall_mean']-dd.loc['C','rare_recall_mean']:+.3f} vs C")

means = [dd.loc[c, f'{metric}_mean'] for c in conds]
stds = [dd.loc[c, f'{metric}_std'] for c in conds]
fig = go.Figure(go.Bar(x=[COND_NAME[c] for c in conds], y=means,
                       error_y=dict(type='data', array=stds),
                       marker_color=[COLORS[c] for c in conds],
                       text=[f'{m:.3f}' for m in means], textposition='outside'))
fig.update_layout(yaxis_title=METRICS[metric])
st.plotly_chart(style_fig(fig), use_container_width=True)

st.markdown('**Per-species: D vs A**')
psm = {'rare_recall': 'recall', 'rare_f1': 'f1', 'rare_precision': 'precision'}.get(metric, 'recall')
ps = per_sp[per_sp.exp_id == exp]
a = ps[ps.condition == 'A'].set_index('species_id')[psm]
dv = ps[ps.condition == 'D'].set_index('species_id')[psm]
nm = ps.drop_duplicates('species_id').set_index('species_id').scientific_name
delta = (dv - a).dropna().sort_values()
fig4 = go.Figure(go.Bar(y=[nm[i] for i in delta.index], x=delta.values, orientation='h',
                        marker_color=['#b1502b' if v < 0 else '#5e7259' for v in delta.values]))
fig4.add_vline(x=0, line_color='#333')
fig4.update_layout(xaxis_title=f'D − A ({psm}), + = ดีขึ้น')
st.plotly_chart(style_fig(fig4, 460), use_container_width=True)

st.divider()
st.subheader('ภาพรวมข้าม config (ไม่ขึ้นกับตัวเลือกด้านบน)')
cc1, cc2 = st.columns(2)
with cc1:
    st.markdown('**Budget sweep (γ=2)**')
    ks = main[(main.gnum == 2) & (main.loss == 'focal')].sort_values('k')
    pf = ks.pivot_table(index='k', columns='condition', values='rare_f1_mean')
    pr = ks.pivot_table(index='k', columns='condition', values='rare_recall_mean')
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=pf.index, y=pf['D'] - pf['A'], mode='lines+markers', name='Δ rare F1', line=dict(color='#b1502b')))
    fig2.add_trace(go.Scatter(x=pr.index, y=pr['D'] - pr['A'], mode='lines+markers', name='Δ recall', line=dict(color='#5e7259')))
    fig2.add_hline(y=0, line_dash='dash', line_color='#bbb')
    fig2.update_layout(xaxis_title='k (synthetic budget)', yaxis_title='D − A')
    st.plotly_chart(style_fig(fig2, 320), use_container_width=True)
    st.caption('k=1 ดีสุด · k=2 เริ่มติดลบ (synthetic เยอะไป → distribution shift)')
with cc2:
    st.markdown('**Robustness across γ (k=1)**')
    gg = main[main.k == 1].sort_values('gnum')
    piv = gg.pivot_table(index='gnum', columns='condition', values=f'{metric}_mean')
    fig3 = go.Figure()
    for c in ['A', 'D']:
        fig3.add_trace(go.Scatter(x=piv.index, y=piv[c], mode='lines+markers', name=c, line=dict(color=COLORS[c])))
    fig3.update_layout(xaxis_title='focal γ (0 = CE)', yaxis_title=METRICS[metric])
    st.plotly_chart(style_fig(fig3, 320), use_container_width=True)
    st.caption('D อยู่เหนือ A ทุก γ → ผลบวกไม่ขึ้นกับ γ ตัวเดียว')

st.divider()
st.header('Try it yourself')
st.info('ส่วนทำนายจะเปิดหลังเลือกโมเดล (A/B/C/D) แล้ว ตอนนี้เตรียมรูปตัวอย่าง + เฉลยไว้แล้ว')
pick = st.selectbox('rare species', sorted(samples.scientific_name.unique()), key='samp')
sub = samples[samples.scientific_name == pick]
for col, r in zip(st.columns(len(sub)), sub.itertuples()):
    col.image(str(ASSETS/r.rel), use_container_width=True, caption=r.organ)
st.caption(f'เฉลย: {pick} · {len(sub)} รูป (test set, held-out ไม่เคยเห็นตอนเทรน)')
st.button('Predict — A vs D', disabled=True)
st.caption('รอเลือกโมเดล → จะโหลด checkpoint แล้วทาย top-3 + เทียบ A กับ D ตรงนี้')