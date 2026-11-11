"""
Генератор синтетического датасета для модели профориентации абитуриентов.

Источник истины — файл сценария «question_ script.xlsx»
(листы «Вопросы» и «Варианты»): ID вариантов, баллы кластерам и пути диалога
берутся оттуда, а не из отдельного opts.json.

Строка датасета = один «прошедший опрос» респондент:
  * признаки   — ответы на вопросы бота (ID выбранного варианта; для мульти-выбора
                 ID через «|»; для шкал — число 1..5; для текстовых вопросов
                 B9/C4/E2/E4 — 12 счётчиков kw_K1..kw_K12: в скольких текстовых
                 ответах сработали ключевые слова кластера);
  * target     — кластер направления (K1..K12), который респондент выбрал.
Вопросы, которых нет на пути диалога респондента (FULL/SHORT/KNOWN/BROWSE) или
не прошедшие условие показа (E6, F3, F4), остаются пустыми — как в реальном чате.

Запуск:
    python generator.py                      # пути по умолчанию
    python generator.py --xlsx X.xlsx --out out.csv --check
"""
import argparse
import random
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

SEED = 42
N_ROWS = 5500
MAX_FEATURES = 50
MAX_CLASS_SHARE = 0.5          # ни один кластер не должен занимать >50% строк
MAX_UNIQUE_SHARE = 0.5         # число уникальных target ≤ 50% от числа строк

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XLSX = PROJECT_ROOT / "data" / "question_ script.xlsx"
DEFAULT_OUT = PROJECT_ROOT / "data" / "proforient_train_dataset.csv"

CLUSTERS = [f"K{i}" for i in range(1, 13)]
PATH_LETTER = {"П": "FULL", "С": "SHORT", "Н": "KNOWN", "О": "BROWSE"}
A3_BY_PATH = {"FULL": "A3_unknown", "SHORT": "A3_choose",
              "KNOWN": "A3_known", "BROWSE": "A3_browse"}

# Вопросы-признаки в порядке их появления в диалоге (порядок важен для условий).
# A1 (имя), A3 (дублирует path), B9/C4/E2/E4 (текст → kw_*), E7 и U_* (зависят
# от уже набранных баллов, т.е. были бы утечкой цели) — не включены.
FEATURE_QS = [
    "A2", "A4",
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8",
    "C1", "C2", "C3", "C5", "C6", "C7",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7",
    "E1", "E3", "E5", "E6",
    "F1", "F2", "F3", "F4", "F5", "F6", "F7",
]
TEXT_QS = ["B9", "C4", "E2", "E4"]
MULTI_DEFAULT_MAX = {"C2": 4, "F1": 3, "F7": 3}   # в листе «Вопросы» не заданы

# Распределение маршрутов (A3)
PATH_PROBS = {"FULL": 0.45, "SHORT": 0.25, "KNOWN": 0.15, "BROWSE": 0.15}

# Распределение целевых кластеров
CLUSTER_WEIGHTS = {
    "K1": 1.4, "K2": 1.3, "K5": 1.2, "K7": 1.1, "K6": 1.0, "K4": 1.0,
    "K8": 0.95, "K10": 0.9, "K11": 0.9, "K9": 0.85, "K3": 0.8, "K12": 0.75,
}
# Смежные кластеры — именно с ними чаще всего «колеблются» (второй интерес)
RELATED = {
    "K1": ["K2", "K3", "K4", "K7"], "K2": ["K1", "K6", "K11"],
    "K3": ["K1", "K4"], "K4": ["K1", "K12", "K7"],
    "K5": ["K6", "K8", "K10", "K1"], "K6": ["K5", "K2", "K10"],
    "K7": ["K12", "K8", "K1"], "K8": ["K5", "K7", "K9"],
    "K9": ["K10", "K8", "K12"], "K10": ["K9", "K5", "K6", "K8"],
    "K11": ["K2", "K9", "K10"], "K12": ["K7", "K4", "K9"],
}
P_SECONDARY = 0.40          # доля респондентов со вторым интересом
P_FOLLOW_PRIMARY = 0.65     # при втором интересе: ответ «от основного» кластера
NEUTRAL_W = 0.5             # вес опции без баллов в вопросе с баллами
OPTION_BONUS = 3.0          # во сколько раз балл повышает вес опции

# Базовый уровень шума (ответ наугад) по маршруту: чем «размытее» цель — тем выше
NOISE_RANGE = (0.10, 0.35)
NOISE_MULT = {"FULL": 1.0, "SHORT": 0.9, "KNOWN": 0.6, "BROWSE": 1.4}

# Вопросы без баллов (фильтры, опасения, класс) — распределения не зависят от кластера
MARGINALS = {
    "A2": {"A2_9": .10, "A2_10": .25, "A2_11": .55, "A2_done": .10},
    "E5": {"E5_mistake": .30, "E5_score": .25, "E5_hard": .15,
           "E5_job": .12, "E5_unknown": .15, "E5_none": .03},
    "E6": {"E6_self": .35, "E6_parents": .30, "E6_friends": .15, "E6_none": .20},
    "F2": {"F2_budget": .50, "F2_both": .35, "F2_paid": .15},
    "F3": {"F3_250": .22, "F3_450": .30, "F3_700": .25, "F3_more": .10, "F3_unknown": .13},
    "F4": {"F4_60": .10, "F4_75": .30, "F4_90": .35, "F4_100": .15, "F4_unknown": .10},
    "F5": {"F5_yes": .40, "F5_no": .45, "F5_unknown": .15},
    "F6": {"F6_full": .85, "F6_part": .06, "F6_extra": .03, "F6_any": .06},
}
# Мульти-выбор без баллов: вероятность выбора каждой опции независимо
F1_PROBS = {"F1_msk": .80, "F1_spb": .35, "F1_nn": .10, "F1_perm": .05}
P_F1_ANY = 0.12
F7_PROBS = {"F7_part_en": .10, "F7_full_en": .04, "F7_army": .12, "F7_access": .03}

# ЕГЭ (C2) связаны с направлением, хотя в сценарии баллов за них нет
EGE_PRIOR = {
    "K1": ["math", "inf", "phys"], "K2": ["math", "inf", "phys"],
    "K3": ["math", "inf", "phys"], "K4": ["math", "phys", "inf"],
    "K5": ["math", "soc", "foreign", "hist"], "K6": ["math", "soc", "foreign"],
    "K7": ["lit", "foreign", "soc", "hist"], "K8": ["lit", "foreign", "soc", "hist"],
    "K9": ["lit", "hist", "foreign", "soc"], "K10": ["soc", "hist", "foreign", "lit"],
    "K11": ["bio", "chem", "math", "soc"], "K12": ["math", "phys", "soc", "lit"],
}
P_EGE_UNKNOWN = 0.10

# Вопросы со шкалой, не имеющие баллов в сценарии: (среднее, σ) по группам кластеров
C6_MEAN = {**{c: 4.0 for c in ("K1", "K2", "K3", "K4", "K6")},
           **{c: 3.3 for c in ("K5", "K11", "K12")},
           **{c: 2.7 for c in ("K7", "K8", "K9", "K10")}}
C7_MEAN = {**{c: 3.8 for c in ("K1", "K5", "K9", "K10")},
           **{c: 3.2 for c in ("K2", "K3", "K4", "K6", "K7", "K8", "K11", "K12")}}
A4_MEAN_BY_PATH = {"FULL": 2.3, "SHORT": 3.2, "KNOWN": 4.3, "BROWSE": 2.5}

# Текстовые вопросы: вероятность, что ответ дан (не пропущен)
P_TEXT_ANSWERED = {"B9": .85, "C4": .65, "E2": .70, "E4": .80}
P_TEXT_ANSWERED_KNOWN_E2 = .95
P_HIT_PROFILE = 0.70        # в ответе найдены слова кластера профиля
P_HIT_PROFILE_KNOWN_E2 = 0.92
P_HIT_EXTRA = 0.20          # ... и ещё одного кластера
P_HIT_STRAY = 0.10          # случайное слово чужого кластера


# ------------------------------------------------------------------ сценарий

def parse_scores(s):
    """'K1:2;K7:1' -> {'K1': 2, 'K7': 1}"""
    out = {}
    if not s:
        return out
    for part in str(s).split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            try:
                out[k.strip()] = int(v.strip())
            except ValueError:
                pass
    return out


def load_scenario(xlsx_path):
    wb = load_workbook(xlsx_path, read_only=True, data_only=True)

    def read(sheet, first_col):
        it = wb[sheet].iter_rows(values_only=True)
        header = next(r for r in it if r and r[0] == first_col)
        return [dict(zip(header, r)) for r in it if r and r[0]]

    questions = {}
    for r in read("Вопросы", "ID"):
        letters = [x.strip() for x in str(r["Пути"] or "").split(",") if x.strip()]
        mx = r["Макс. выбор"]
        questions[r["ID"]] = {
            "type": str(r["Тип"]).lower(),
            "paths": {PATH_LETTER[x] for x in letters if x in PATH_LETTER},
            "max": int(mx) if mx not in (None, "") else None,
        }

    options = {}
    for r in read("Варианты", "ID вопроса"):
        options.setdefault(r["ID вопроса"], []).append(
            {"id": r["ID варианта"], "scores": parse_scores(r["Баллы кластерам"])}
        )

    for q in FEATURE_QS + TEXT_QS:
        assert q in questions, f"в листе «Вопросы» нет {q}"
    for q in FEATURE_QS:
        assert q in options, f"в листе «Варианты» нет {q}"
    return questions, options


class Scenario:
    def __init__(self, xlsx_path):
        self.q, self.opts = load_scenario(xlsx_path)
        self.scales = {q for q in FEATURE_QS if "шкала" in self.q[q]["type"]}
        self.multi = {q for q in FEATURE_QS if "несколько" in self.q[q]["type"]}
        self.scale_model = {q: self._scale_model(q) for q in self.scales
                            if q not in ("A4", "C6", "C7")}
        for q, d in MARGINALS.items():      # ID в таблицах вероятностей должны существовать
            ids = {o["id"] for o in self.opts[q]}
            assert set(d) <= ids, f"{q}: неизвестные ID {set(d) - ids}"

    def on_path(self, qid, path):
        return path in self.q[qid]["paths"]

    def _scale_model(self, qid):
        """Для каждого кластера: (среднее, σ) ответа на шкалу — из весов сценария."""
        per = {int(o["id"][-1]): o["scores"] for o in self.opts[qid]}
        one_sided = not per[1] and not per[2]       # B3–B8: «0 баллов» за 1–2
        model = {}
        for c in CLUSTERS:
            low = per[1].get(c, 0) + per[2].get(c, 0)
            high = per[4].get(c, 0) + per[5].get(c, 0)
            if high > low:
                model[c] = (4.3, 0.8)               # кластер «за» высокий ответ
            elif low > high:
                model[c] = (1.8, 0.8)               # кластер «за» низкий ответ (C5, D3, D6)
            elif per[3].get(c, 0) > 0:
                model[c] = (3.0, 0.6)               # кластер ждёт «середину» (D3)
            else:
                model[c] = (2.2, 0.9) if one_sided else (3.0, 1.0)
        return model


# ------------------------------------------------------------------ респондент

class Respondent:
    def __init__(self, rng, primary, secondary, path):
        self.rng, self.primary, self.secondary, self.path = rng, primary, secondary, path
        self.noise = rng.uniform(*NOISE_RANGE) * NOISE_MULT[path]

    def profile(self):
        """Кластер, «от имени» которого респондент отвечает на этот вопрос."""
        if self.secondary and self.rng.random() > P_FOLLOW_PRIMARY:
            return self.secondary
        return self.primary


def wchoice(rng, items, weights):
    return rng.choices(items, weights=weights, k=1)[0]


def sample_single(sc, R, qid):
    opts = sc.opts[qid]
    if qid in MARGINALS:
        d = MARGINALS[qid]
        return wchoice(R.rng, list(d), list(d.values()))
    if R.rng.random() < R.noise:
        return R.rng.choice(opts)["id"]
    c = R.profile()
    w = [(NEUTRAL_W if not o["scores"] else 0.1 + OPTION_BONUS * o["scores"].get(c, 0))
         for o in opts]
    return wchoice(R.rng, [o["id"] for o in opts], w)


def sample_multi(sc, R, qid):
    rng = R.rng
    if qid == "F1":
        if rng.random() < P_F1_ANY:
            return "F1_any"
        ids = [i for i, p in F1_PROBS.items() if rng.random() < p] or ["F1_msk"]
        return "|".join(ids)
    if qid == "F7":
        ids = [i for i, p in F7_PROBS.items() if rng.random() < p] or ["F7_none"]
        return "|".join(ids)
    if qid == "C2":
        if rng.random() < P_EGE_UNKNOWN:
            return "C2_unknown"
        fav = EGE_PRIOR[R.profile()]
        pool = [o["id"] for o in sc.opts["C2"] if o["id"] != "C2_unknown"]
        k = wchoice(rng, [1, 2, 3, 4], [.10, .40, .40, .10])
        return "|".join(_pick_many(R, pool, k, lambda i, c: 4.0 if i[3:] in EGE_PRIOR[c] else 0.3))
    # остальные мульти-вопросы (B1, C1, D4, E3) — по баллам сценария
    mx = sc.q[qid]["max"] or MULTI_DEFAULT_MAX.get(qid, 3)
    k = wchoice(rng, list(range(1, mx + 1)), [.30, .40, .30][:mx])
    score = {o["id"]: o["scores"] for o in sc.opts[qid]}
    pool = list(score)
    return "|".join(_pick_many(
        R, pool, k,
        lambda i, c: NEUTRAL_W if not score[i] else 0.1 + OPTION_BONUS * score[i].get(c, 0)))


def _pick_many(R, pool, k, weight_fn):
    pool, chosen = list(pool), []
    for _ in range(min(k, len(pool))):
        if R.rng.random() < R.noise:
            pick = R.rng.choice(pool)
        else:
            c = R.profile()
            pick = wchoice(R.rng, pool, [weight_fn(i, c) for i in pool])
        chosen.append(pick)
        pool.remove(pick)
    return chosen


def sample_scale(sc, R, qid):
    if R.rng.random() < R.noise:
        return R.rng.randint(1, 5)
    c = R.profile()
    if qid == "C6":
        mean, sd = C6_MEAN[c], 1.0
    elif qid == "C7":
        mean, sd = C7_MEAN[c], 1.1
    else:
        mean, sd = sc.scale_model[qid][c]
    return int(min(5, max(1, round(R.rng.gauss(mean, sd)))))


def sample_text_hits(R, qid, hits):
    """Эмулирует разбор текстового ответа: какие кластеры нашлись по ключевым словам."""
    rng = R.rng
    known_e2 = (qid == "E2" and R.path == "KNOWN")
    p_ans = P_TEXT_ANSWERED_KNOWN_E2 if known_e2 else P_TEXT_ANSWERED[qid]
    if rng.random() > p_ans:
        return
    found = set()
    if rng.random() < (P_HIT_PROFILE_KNOWN_E2 if known_e2 else P_HIT_PROFILE):
        found.add(R.profile() if not known_e2 else R.primary)
    if rng.random() < P_HIT_EXTRA:
        found.add(R.secondary or rng.choice(CLUSTERS))
    if rng.random() < P_HIT_STRAY:
        found.add(rng.choice(CLUSTERS))
    for c in found:
        hits[c] += 1


# ------------------------------------------------------------------ строка

def generate_row(sc, rng, primary, secondary):
    path = wchoice(rng, list(PATH_PROBS), list(PATH_PROBS.values()))
    R = Respondent(rng, primary, secondary, path)
    row = {q: "" for q in FEATURE_QS}

    for q in FEATURE_QS:
        if not sc.on_path(q, path):
            continue
        # условия показа
        if q == "A4":
            row[q] = int(min(5, max(1, round(rng.gauss(A4_MEAN_BY_PATH[path], 1.0)))))
            continue
        if q == "E6" and not (row["A4"] != "" and row["A4"] <= 3):
            continue
        if q == "F3" and row["F2"] == "F2_budget":
            continue
        if q == "F4" and (row["C2"] in ("", "C2_unknown")):
            continue
        # сэмплирование
        if q in sc.scales:
            row[q] = sample_scale(sc, R, q)
        elif q in sc.multi:
            row[q] = sample_multi(sc, R, q)
        else:
            row[q] = sample_single(sc, R, q)

    hits = {c: 0 for c in CLUSTERS}
    for q in TEXT_QS:
        if sc.on_path(q, path):
            sample_text_hits(R, q, hits)
    for c in CLUSTERS:
        row[f"kw_{c}"] = hits[c]

    row["path"] = path
    row["target_cluster"] = primary
    return row


# ------------------------------------------------------------------ проверки

def encode_for_check(df):
    """Грубое кодирование для контрольной модели: мульти-выбор → multi-hot."""
    X = pd.DataFrame(index=df.index)
    for col in df.columns:
        s = df[col]
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().sum() == s.notna().sum():          # числовой признак (шкала, kw_*)
            X[col] = num.fillna(0)
            X[col + "_na"] = num.isna().astype(int)
        elif col == "path":
            X = X.join(pd.get_dummies(s, prefix=col, dtype=int))
        else:                                             # ID варианта или ID через «|»
            X = X.join(s.fillna("").astype(str).str.get_dummies(sep="|").add_prefix(col + "="))
    return X


def baseline_check(df):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    import numpy as np
    feats = df.drop(columns=["target_cluster"])
    X = encode_for_check(feats)
    y = df["target_cluster"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    m = HistGradientBoostingClassifier(max_iter=150, random_state=SEED).fit(Xtr, ytr)
    proba = m.predict_proba(Xte)
    top1 = (m.classes_[proba.argmax(1)] == yte.values).mean()
    top3 = np.mean([t in m.classes_[np.argsort(p)[-3:]] for t, p in zip(yte.values, proba)])
    print(f"\n[check] baseline HistGB: accuracy={top1:.3f}, top-3={top3:.3f}")
    paths = feats.loc[Xte.index, "path"].values
    for p in ("FULL", "SHORT", "KNOWN", "BROWSE"):
        mask = paths == p
        acc = (m.classes_[proba[mask].argmax(1)] == yte.values[mask]).mean()
        print(f"[check]   path={p:7s} n={mask.sum():4d} accuracy={acc:.3f}")


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n", type=int, default=N_ROWS)
    ap.add_argument("--check", action="store_true", help="обучить контрольную модель")
    args = ap.parse_args()

    rng = random.Random(SEED)
    sc = Scenario(args.xlsx)

    targets = rng.choices(CLUSTERS, weights=[CLUSTER_WEIGHTS[c] for c in CLUSTERS], k=args.n)
    rows = []
    for t in targets:
        sec = None
        if rng.random() < P_SECONDARY:
            sec = rng.choice(RELATED[t]) if rng.random() < 0.7 else \
                rng.choice([c for c in CLUSTERS if c != t])
        rows.append(generate_row(sc, rng, t, sec))

    feat_cols = FEATURE_QS + [f"kw_{c}" for c in CLUSTERS] + ["path"]
    df = pd.DataFrame(rows)[feat_cols + ["target_cluster"]]

    # ---- жёсткие проверки контракта
    shares = df["target_cluster"].value_counts(normalize=True)
    assert len(df) >= 5000, f"rows={len(df)}"
    assert len(feat_cols) <= MAX_FEATURES, f"features={len(feat_cols)}"
    assert df["target_cluster"].nunique() / len(df) <= MAX_UNIQUE_SHARE
    assert shares.max() <= MAX_CLASS_SHARE, f"max target share={shares.max():.3f}"
    assert not df.duplicated().any(), "есть полностью одинаковые строки"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")   # utf-8-sig: корректно открывается в Excel

    print(f"Saved {len(df)} rows × {len(feat_cols)} features → {args.out}")
    print("\nTarget distribution:\n", df["target_cluster"].value_counts().sort_index().to_string())
    print(f"\nUnique targets: {df['target_cluster'].nunique()}, max share: {shares.max():.3f}")
    print("\nPath distribution:\n", df["path"].value_counts(normalize=True).round(3).to_string())
    empty = (df[feat_cols].replace("", pd.NA).isna().mean() * 100).round(1)
    print("\nПустых значений, % (по признакам с пропусками):\n",
          empty[empty > 0].to_string())
    if args.check:
        baseline_check(df.replace("", pd.NA))


if __name__ == "__main__":
    main()