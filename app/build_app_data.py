from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml
from PIL import Image

ROOT = Path('/workspace/lora-boost')
META = ROOT/'data'/'metadata'
IMG_ROOT = ROOT/'data'/'raw'/'plantnet_300K'/'images'
SYNTH_ROOT = ROOT/'data'/'synthetic'
EXP_ROOT = ROOT/'results'/'experiments'

APP = Path(__file__).resolve().parent
OUT_DATA = APP/'data'
OUT_ASSETS = APP/'assets'
OUT_DATA.mkdir(parents=True, exist_ok=True)

GALLERY_SPECIES = ['1419807', '1406486', '1356076', '1409195', '1420787']   # winner + loser + hard pair
GALLERY_PER, SAMPLES_PER, MAX_PX = 3, 3, 512

species   = pd.read_csv(META/'species.csv', dtype={'species_id': str})
splits    = pd.read_csv(META/'splits.csv', dtype={'species_id': str})
lora_eval = pd.read_csv(ROOT/'results'/'nb02'/'lora_eval.csv', dtype={'species_id': str})

rare_ids = species.loc[species.is_rare, 'species_id'].tolist()
name_map = species.set_index('species_id').scientific_name.to_dict()
print('rare:', len(rare_ids))

def save_thumb(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert('RGB')
    im.thumbnail((MAX_PX, MAX_PX))
    im.save(dst, 'JPEG', quality=85)

# longtail + rare table
species[['species_id', 'scientific_name', 'is_rare', 'n_total', 'n_train']] \
    .to_csv(OUT_DATA/'longtail.csv', index=False)

rare = species[species.is_rare][['species_id', 'scientific_name', 'n_total', 'n_train']]
rare = rare.merge(lora_eval[['species_id', 'consistency', 'organ_acc', 'flag']], on='species_id', how='left')
rare.to_csv(OUT_DATA/'rare_species.csv', index=False)

# results: รวม metric ราย exp x condition (mean/std ข้าม seed) + per-species
sum_rows, ps_rows = [], []
for exp in sorted(EXP_ROOT.glob('exp_*')):
    cfg = yaml.safe_load(open(exp/'config.yaml'))
    meta = dict(exp_id=exp.name, k=cfg['budget']['k'], p=cfg['budget']['p'],
                loss=cfg['loss']['type'], gamma=cfg['loss'].get('gamma'),
                es_monitor=cfg['early_stop']['monitor'],
                C=cfg['_resolved']['C'], R=cfg['_resolved']['R'])
    for cond in cfg['conditions']:
        mfiles = [exp/'seeds'/f's{s}'/f'metrics_{cond}.json' for s in cfg['seeds']]
        ms = [json.load(open(f)) for f in mfiles if f.exists()]
        if not ms:
            continue
        row = {**meta, 'condition': cond, 'n_seeds': len(ms)}
        for short, key in [('rare_f1', 'test_rare_f1'), ('rare_recall', 'test_rare_recall'),
                           ('rare_precision', 'test_rare_precision'),
                           ('overall_f1', 'test_overall_f1'), ('overall_acc', 'test_overall_acc')]:
            vals = [m[key] for m in ms if key in m]
            row[f'{short}_mean'] = float(np.mean(vals)) if vals else np.nan
            row[f'{short}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        sum_rows.append(row)

        for tgt, key in [('f1', 'rare_f1_per_species'), ('recall', 'rare_recall_per_species'),
                         ('precision', 'rare_precision_per_species')]:
            if key not in ms[0]:
                continue
            for sid in ms[0][key]:
                ps_rows.append(dict(exp_id=exp.name, condition=cond, metric=tgt,
                                    species_id=sid, value=float(np.mean([m[key][sid] for m in ms]))))

pd.DataFrame(sum_rows).to_csv(OUT_DATA/'results_summary.csv', index=False)

per_sp = pd.DataFrame(ps_rows).pivot_table(index=['exp_id', 'condition', 'species_id'],
                                           columns='metric', values='value').reset_index()
per_sp['scientific_name'] = per_sp.species_id.map(name_map)
per_sp.to_csv(OUT_DATA/'per_species.csv', index=False)
print('exps:', pd.DataFrame(sum_rows).exp_id.nunique())

# gallery: real / zeroshot / lora
gal = [s for s in GALLERY_SPECIES if (SYNTH_ROOT/'lora_boost'/s).exists()]
for sid in gal:
    tr = splits[(splits.species_id == sid) & (splits.split == 'train')]
    organ = tr.organ.mode().iloc[0]                          # organ ที่ species นี้มีเยอะสุด
    for i, rel in enumerate(tr[tr.organ == organ].path.tolist()[:GALLERY_PER]):
        save_thumb(IMG_ROOT/rel, OUT_ASSETS/'gallery'/sid/'real'/f'{i}.jpg')
    for src, tag in [('flux_zeroshot', 'zeroshot'), ('lora_boost', 'lora')]:
        files = [f for f in sorted((SYNTH_ROOT/src/sid).glob('*.png')) if organ in f.name][:GALLERY_PER]
        for i, f in enumerate(files):
            save_thumb(f, OUT_ASSETS/'gallery'/sid/tag/f'{i}.jpg')
pd.DataFrame({'species_id': gal, 'scientific_name': [name_map[s] for s in gal]}) \
    .to_csv(OUT_DATA/'gallery_index.csv', index=False)

# samples: รูป test ของ rare (held-out) + เฉลย
srows = []
for sid in rare_ids:
    pool = splits[(splits.species_id == sid) & (splits.split == 'test')]
    for i, r in enumerate(pool.sample(min(SAMPLES_PER, len(pool)), random_state=42).itertuples()):
        save_thumb(IMG_ROOT/r.path, OUT_ASSETS/'samples'/sid/f'{i}.jpg')
        srows.append(dict(species_id=sid, scientific_name=name_map[sid], organ=r.organ,
                          rel=str(Path('samples')/sid/f'{i}.jpg')))
pd.DataFrame(srows).to_csv(OUT_DATA/'samples_index.csv', index=False)
print('samples:', len(srows))
