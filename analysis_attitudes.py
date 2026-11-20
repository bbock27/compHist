"""
Whose optimism?  Consumer attitudes in the 1950 Survey of Consumer Finances by race and by
sex of the head of the spending unit.

    python3 clean_scf1950.py        # raw card images -> clean/
    python3 analysis_attitudes.py   # -> output/attitudes/*.csv and report.txt

All estimates, standard errors, tests and models go through `svy` with the course design
(weights only; see scf_survey.py).  Section 7 repeats the headline tests with the sample point
as a cluster, as a sensitivity check.

Outcomes are 0/1 indicators.  "Not ascertained" is missing for every outcome; a spoken
"don't know" is a real answer and is kept.  Questions not asked of a group (farm schedule,
self-employed) are missing for that group.
"""
import os
import warnings

import numpy as np
import pandas as pd
import polars as pl
import svy

import scf_survey
from scf_survey import _frame, _sample

warnings.filterwarnings("ignore")
OUT = "output/attitudes"
os.makedirs(OUT, exist_ok=True)
REPORT = []


def say(*lines):
    for line in lines:
        print(line)
        REPORT.append(str(line))


# ----------------------------------------------------------------------------- 1. data
def ind(series, yes, missing=("not ascertained",)):
    """1.0 if the answer is in `yes`, missing if in `missing`, else 0.0."""
    out = series.isin(list(yes)).astype(float)
    out[series.isin(list(missing)) | series.isna()] = np.nan
    return out


def prepare():
    d = _frame(labeled=True)
    d["race2"] = d["race"].replace("Negro", "black").where(d["race"].isin(["white", "Negro"]))  # codebook label recoded for display
    d["sex2"] = d["sex_head"].where(d["sex_head"].isin(["male", "female"]))
    d["black"] = (d["race2"] == "black").astype(float).where(d["race2"].notna())
    d["female"] = (d["sex2"] == "female").astype(float).where(d["sex2"].notna())
    d["inc3"] = pd.cut(d["income_1949"], [-1, 1999, 3999, 1e9],
                       labels=["under $2,000", "$2,000-3,999", "$4,000 and over"]).astype(object)
    fs = d["family_status"].astype(str)
    d["married"] = np.where(fs.str.contains("unmarried|not married"), "not married",
                            np.where(fs.str.contains("married"), "married", None))
    d["unmarried_sex"] = np.where(d["married"] == "not married", d["sex2"], None)

    na = ("not ascertained",)
    farm = ("farm schedule", "not ascertained")
    y = {}
    y["worse_off"] = ind(d.fin_position_vs_year_ago, ["worse off"])
    y["better_off"] = ind(d.fin_position_vs_year_ago, ["better off"])
    y["good_times_ahead"] = ind(d.times_next_12_months, ["fairly good", "very good"])
    y["bad_times_ahead"] = ind(d.times_next_12_months, ["fairly bad", "very bad"])
    y["dk_times_ahead"] = ind(d.times_next_12_months, ["undecided, don't know"])
    y["dk_or_na_times_ahead"] = ind(d.times_next_12_months, ["undecided, don't know", "not ascertained"], ())
    y["insecure"] = ind(d.financial_security, ["rather insecure", "very insecure"])
    y["very_insecure"] = ind(d.financial_security, ["very insecure"])
    y["income_will_rise"] = ind(d.income_expected_year_ahead, ["will make more"], farm)
    y["dk_own_income"] = ind(d.income_expected_year_ahead, ["don't know, it depends"], farm)
    y["dk_prices"] = ind(d.prices_expected_1950, ["undecided, don't know"])
    y["prices_will_fall"] = ind(d.prices_expected_1950, ["prices will go down"])
    y["dk_times_now"] = ind(d.times_now, ["undecided, don't know"])
    y["plans_durable"] = ind(d.durables_plan_1950, ["definitely plans to buy (or has bought in 1950)", "probably will buy"])
    jc = d.job_chances.astype(str)
    y["job_hard_to_replace"] = ind(d.job_chances, ["same pay: would be difficult", "same pay: impossible"])
    y["job_hard_to_replace"][~jc.str.startswith(("same pay", "less pay", "in a depression", "don't know"))] = np.nan
    for k, v in y.items():
        d[k] = v
    # the Bulletin's "basic ingredient of consumer optimism": good times ahead and own income
    # expected to hold or rise (non-farm units with both answers ascertained)
    inc_ok = ind(d.income_expected_year_ahead, ["will make more", "about the same"], farm)
    d["fed_optimist"] = (d["good_times_ahead"] * inc_ok)
    d["dk_count"] = d[["dk_times_ahead", "dk_prices", "dk_times_now"]].sum(axis=1, min_count=3)
    d["any_dk"] = (d["dk_count"] > 0).astype(float).where(d["dk_count"].notna())
    rs = d["sex_respondent"].where(d["sex_respondent"].isin(["male", "female"]))
    d["woman_answered"] = (rs == "female").astype(float).where(rs.notna())
    # A: inside male-headed units, did a woman (almost always the wife) or a man answer?
    d["resp_in_male_headed"] = np.where(d.sex2 == "male", rs, None)
    # B: women heads against male heads who answered for themselves
    d["head_self_answered"] = np.where(d.sex2 == "female", "female", np.where((d.sex2 == "male") & (rs == "male"), "male", None))
    # C: women respondents only: "female" = woman heads her own unit, "male" = wife in a male-headed unit
    d["women_resp_by_head"] = np.where(rs == "female", d.sex2, None)
    d["white_male_head"] = ((d.race2 == "white") & (d.sex2 == "male")).astype(float).where(d.race2.notna() & d.sex2.notna())
    return d


OUTCOMES = ["worse_off", "better_off", "insecure", "very_insecure", "good_times_ahead", "bad_times_ahead",
            "dk_times_ahead", "dk_own_income", "dk_prices", "any_dk", "income_will_rise", "prices_will_fall",
            "job_hard_to_replace", "plans_durable", "fed_optimist"]
HEADLINE = ["worse_off", "better_off", "very_insecure", "good_times_ahead", "dk_times_ahead",
            "dk_own_income", "any_dk", "plans_durable", "fed_optimist"]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ----------------------------------------------------------------------------- 2. gaps
def gaps(sample, d, group, a, b, outcomes, where=None, mask=None, label="all"):
    """Design-based two-sample tests of each outcome: level a minus level b of `group`."""
    rows = []
    mask = pd.Series(True, index=d.index) if mask is None else mask
    for yv in outcomes:
        sub = d[mask & d[group].isin([a, b]) & d[yv].notna()]
        na_, nb_ = (sub[group] == a).sum(), (sub[group] == b).sum()
        if min(na_, nb_) < 2:
            continue
        t = sample.categorical.ttest(yv, group=group, where=where, drop_nulls=True)
        tp = t.to_polars().to_dicts()[0]
        ea, eb = _lvl(t, a), _lvl(t, b)
        diff = ea[0] - eb[0]
        rows.append(dict(subset=label, outcome=yv, group=group,
                         **{f"pct_{a}": 100 * ea[0], f"se_{a}": 100 * ea[1], f"n_{a}": na_,
                            f"pct_{b}": 100 * eb[0], f"se_{b}": 100 * eb[1], f"n_{b}": nb_},
                         diff_pts=100 * diff, se_diff=100 * tp["se"],
                         ci_low=100 * (diff - 1.96 * tp["se"]), ci_high=100 * (diff + 1.96 * tp["se"]),
                         p=_pval(t, tp), small_cell=min(na_, nb_) < 30))
    out = pd.DataFrame(rows)
    return out


def _lvl(t, level):
    for e in t.estimates:
        lv = getattr(e, "group_level", None) or getattr(e, "level", None) or getattr(e, "by_level", None)
        if isinstance(lv, (list, tuple)):
            lv = lv[0]
        if str(lv) == str(level):
            return float(e.est), float(e.se)
    raise KeyError(level)


def _pval(t, tp):
    for k in ("p_value", "pvalue", "p"):
        if k in tp and tp[k] is not None:
            return float(tp[k])
    s = t.stats
    return float(getattr(s, "p_value", np.nan))


def show(df, a, b, title):
    say("", title)
    say(f"  {'outcome':22s} {a:>16s} {b:>16s} {'diff':>8s} {'95% CI':>16s} {'p':>8s}")
    for _, r in df.iterrows():
        flag = " *n<30" if r.small_cell else ""
        say(f"  {r.outcome:22s} {r[f'pct_{a}']:5.1f} ({r[f'se_{a}']:4.1f}) n={int(r[f'n_{a}']):<4d}"
            f" {r[f'pct_{b}']:5.1f} ({r[f'se_{b}']:4.1f}) n={int(r[f'n_{b}']):<4d}"
            f" {r.diff_pts:+7.1f} [{r.ci_low:+6.1f},{r.ci_high:+6.1f}] {r.p:8.4f}{flag}")


# ----------------------------------------------------------------------------- 3. models
CONTROLS = [("income_1949_class", "$2,000-2,999"), ("age_head", "35-44"),
            ("occupation_group", "skilled or semi-skilled"), ("place_size", "metropolitan area"),
            ("married", "married")]


def models(sample, d, outcomes):
    rows = []
    for yv in outcomes:
        for name, xs in [("M1 race + sex", ["black", "female"]),
                         ("M2 + income, age, occupation, place, marital", ["black", "female"] +
                          [svy.Cat(c, ref=r) for c, r in CONTROLS])]:
            try:
                m = sample.glm.fit(yv, x=xs, family="binomial", drop_nulls=True)
            except Exception as e:  # noqa: BLE001
                say(f"  model failed: {yv} / {name}: {type(e).__name__}: {e}")
                continue
            tab = m.to_polars().to_pandas()
            pcol = next((c for c in tab.columns if c.lower() in ("p_value", "pvalue", "p")), None)
            nobs = getattr(m.stats, "n", None)
            for term in ("black", "female"):
                r = tab[tab.term == term]
                if r.empty:
                    continue
                r = r.iloc[0]
                t = r.est / r.se
                rows.append(dict(outcome=yv, model=name, term=term, logit=r.est, se=r.se,
                                 odds_ratio=np.exp(r.est), or_low=np.exp(r.est - 1.96 * r.se),
                                 or_high=np.exp(r.est + 1.96 * r.se),
                                 p=float(r[pcol]) if pcol else np.nan, t=t, n=nobs))
    return pd.DataFrame(rows)


def adjusted_gap(sample, d, yv):
    """Average adjusted difference in probability (black - white, female - male) from M2,
    by prediction at observed covariates with the group switched (weights from the survey)."""
    xs = ["black", "female"] + [svy.Cat(c, ref=r) for c, r in CONTROLS]
    m = sample.glm.fit(yv, x=xs, family="binomial", drop_nulls=True)
    tab = m.to_polars().to_pandas().set_index("term")["est"]
    use = d.dropna(subset=[yv, "black", "female"] + [c for c, _ in CONTROLS]).copy()
    eta = np.full(len(use), tab["_intercept_"])
    for c, ref in CONTROLS:
        for lvl in use[c].unique():
            key = f"{c}_{lvl}"
            if key in tab.index:
                eta = eta + tab[key] * (use[c] == lvl).to_numpy()
    out = {}
    for term in ("black", "female"):
        other = "female" if term == "black" else "black"
        base = eta + tab[other] * use[other].to_numpy()
        p1 = 1 / (1 + np.exp(-(base + tab[term])))
        p0 = 1 / (1 + np.exp(-base))
        out[term] = 100 * np.average(p1 - p0, weights=use["weight"])
    return out


# ----------------------------------------------------------------------------- main
def main():
    d = prepare()
    s = _sample(d, "weight")
    say("WHOSE OPTIMISM?  1950 Survey of Consumer Finances, attitude questions",
        f"svy {svy.__version__}; design: weights only (course rule); n = {len(d)} spending units",
        "Percentages are weighted; standard errors in parentheses; differences in percentage points.")

    # 1. everyone
    say("", "1. ALL SPENDING UNITS")
    rows = []
    for yv in OUTCOMES:
        e = s.estimation.mean(yv, drop_nulls=True).to_polars().to_dicts()[0]
        rows.append(dict(outcome=yv, pct=100 * e["est"], se=100 * e["se"], n=int(d[yv].notna().sum())))
        say(f"  {yv:22s} {100*e['est']:5.1f} ({100*e['se']:.1f})  n={int(d[yv].notna().sum())}")
    pd.DataFrame(rows).to_csv(f"{OUT}/1_overall.csv", index=False)

    # 2. race
    allg = []
    g = gaps(s, d, "race2", "black", "white", OUTCOMES)
    g["p_holm"] = np.nan
    hl = g.outcome.isin(HEADLINE)
    g.loc[hl, "p_holm"] = holm(g.loc[hl, "p"])
    show(g, "black", "white", "2. RACE: black minus white, all incomes")
    say("  Holm-adjusted p (9 headline outcomes): " +
        ", ".join(f"{r.outcome} {r.p_holm:.3f}" for _, r in g[hl].iterrows()))
    allg.append(g)
    for band in ["under $2,000", "$2,000-3,999"]:
        gb = gaps(s, d, "race2", "black", "white", OUTCOMES, where=svy.col("inc3") == band,
                  mask=d.inc3 == band, label=band)
        show(gb, "black", "white", f"   RACE within 1949 income {band}")
        allg.append(gb)
    pd.concat(allg).to_csv(f"{OUT}/2_race_gaps.csv", index=False)

    # 3. sex of head
    allg = []
    g = gaps(s, d, "sex2", "female", "male", OUTCOMES)
    g["p_holm"] = np.nan
    hl = g.outcome.isin(HEADLINE)
    g.loc[hl, "p_holm"] = holm(g.loc[hl, "p"])
    show(g, "female", "male", "3. SEX OF HEAD: female minus male, all incomes")
    say("  Holm-adjusted p (9 headline outcomes): " +
        ", ".join(f"{r.outcome} {r.p_holm:.3f}" for _, r in g[hl].iterrows()))
    allg.append(g)
    for band in ["under $2,000", "$2,000-3,999", "$4,000 and over"]:
        gb = gaps(s, d, "sex2", "female", "male", OUTCOMES, where=svy.col("inc3") == band,
                  mask=d.inc3 == band, label=band)
        show(gb, "female", "male", f"   SEX within 1949 income {band}")
        allg.append(gb)
    gu = gaps(s, d, "unmarried_sex", "female", "male", OUTCOMES, label="unmarried heads only")
    show(gu, "female", "male", "   SEX among UNMARRIED heads only (women heads vs unmarried men)")
    allg.append(gu)
    pd.concat(allg).to_csv(f"{OUT}/3_sex_gaps.csv", index=False)

    # 4. models
    say("", "4. LOGISTIC REGRESSIONS (svy GLM, binomial).  Odds ratios for black (vs white) and",
        "   female head (vs male).  M2 adds income class, age, occupation group, size of place, marital status.")
    mt = models(s, d, HEADLINE + ["insecure"])
    mt.to_csv(f"{OUT}/4_models.csv", index=False)
    for yv in mt.outcome.unique():
        say(f"  {yv}")
        for _, r in mt[mt.outcome == yv].iterrows():
            say(f"     {r.model[:2]} {r.term:7s} OR {r.odds_ratio:5.2f} [{r.or_low:4.2f},{r.or_high:5.2f}]  p={r.p:.4f}")
    say("", "   Adjusted gaps from M2 (average change in predicted probability, percentage points):")
    adj = []
    for yv in HEADLINE + ["insecure"]:
        try:
            a = adjusted_gap(s, d, yv)
            adj.append(dict(outcome=yv, black_minus_white=a["black"], female_minus_male=a["female"]))
            say(f"     {yv:22s} black-white {a['black']:+5.1f}   female-male {a['female']:+5.1f}")
        except Exception as e:  # noqa: BLE001
            say(f"     {yv}: failed ({e})")
    pd.DataFrame(adj).to_csv(f"{OUT}/4_adjusted_gaps.csv", index=False)

    # 5. who the optimists were
    say("", "5. WHO THE 'OPTIMISTIC CONSUMER' WAS",
        "   Share of each group of answers that came from white, male-headed spending units.")
    rows = []
    e = s.estimation.mean("white_male_head", drop_nulls=True).to_polars().to_dicts()[0]
    say(f"  all spending units                      {100*e['est']:5.1f} ({100*e['se']:.1f})")
    rows.append(dict(group="all spending units", pct_white_male=100 * e["est"], se=100 * e["se"]))
    for yv, lab in [("fed_optimist", "optimists (good times + income holds/rises)"),
                    ("good_times_ahead", "expect good times"), ("better_off", "better off than a year ago"),
                    ("plans_durable", "plan a durable purchase"), ("dk_times_ahead", "don't know about times ahead"),
                    ("very_insecure", "very insecure"), ("worse_off", "worse off")]:
        e = s.estimation.mean("white_male_head", where=svy.col(yv) == 1, drop_nulls=True).to_polars().to_dicts()[0]
        say(f"  {lab:40s}{100*e['est']:5.1f} ({100*e['se']:.1f})")
        rows.append(dict(group=lab, pct_white_male=100 * e["est"], se=100 * e["se"]))
    pd.DataFrame(rows).to_csv(f"{OUT}/5_composition.csv", index=False)

    # 6. four groups
    say("", "6. RACE BY SEX OF HEAD (weighted %, unweighted n)")
    d["rs"] = np.where(d.race2.notna() & d.sex2.notna(), d.race2.astype(str) + ", " + d.sex2.astype(str) + " head", None)
    s2 = _sample(d, "weight")
    rows = []
    for yv in HEADLINE:
        t = s2.estimation.mean(yv, by="rs", drop_nulls=True).to_polars().to_pandas()
        keycol = "rs" if "rs" in t.columns else [c for c in t.columns if c not in ("est", "se")][0]
        line = f"  {yv:22s}"
        for _, r in t.iterrows():
            n = int(((d.rs == r[keycol]) & d[yv].notna()).sum())
            rows.append(dict(outcome=yv, group=r[keycol], pct=100 * r.est, se=100 * r.se, n=n))
            line += f" | {r[keycol]}: {100*r.est:4.1f} ({100*r.se:.1f}) n={n}"
        say(line)
    pd.DataFrame(rows).to_csv(f"{OUT}/6_race_by_sex.csv", index=False)

    # 7. clustering sensitivity
    say("", "7. SENSITIVITY: same headline tests with the sample point as a cluster (66 points).",
        "   Point estimates are unchanged; standard errors and p-values change.")
    scf_survey.CLUSTERED = True
    sc = _sample(d, "weight")
    rows = []
    for grp, a, b in [("race2", "black", "white"), ("sex2", "female", "male")]:
        gc = gaps(sc, d, grp, a, b, HEADLINE, label="clustered")
        for _, r in gc.iterrows():
            rows.append(dict(group=grp, outcome=r.outcome, diff_pts=r.diff_pts, se_clustered=r.se_diff, p_clustered=r.p))
            say(f"  {grp:5s} {r.outcome:22s} diff {r.diff_pts:+6.1f}  SE {r.se_diff:4.1f}  p={r.p:.4f}")
    scf_survey.CLUSTERED = False
    pd.DataFrame(rows).to_csv(f"{OUT}/7_clustered_sensitivity.csv", index=False)

    # 8. who was speaking
    say("", "8. SEX OF THE PERSON WHO ANSWERED",
        "   The survey records the sex of the respondent as well as of the head.")
    e = s.estimation.mean("woman_answered", drop_nulls=True).to_polars().to_dicts()[0]
    say(f"  interviews answered by a woman: {100*e['est']:.1f}% ({100*e['se']:.1f}); unweighted "
        f"{int(d.woman_answered.sum())} of {int(d.woman_answered.notna().sum())}")
    keep = ["good_times_ahead", "dk_times_ahead", "any_dk", "dk_own_income", "very_insecure", "insecure",
            "worse_off", "fed_optimist"]
    parts = []
    for col, lab in [("resp_in_male_headed", "A. MALE-HEADED units: a woman answered vs a man answered"),
                     ("head_self_answered", "B. Women heads vs male heads who answered for themselves"),
                     ("women_resp_by_head", "C. WOMEN RESPONDENTS only: women heads ('female') vs wives in male-headed units ('male')")]:
        gx = gaps(s, d, col, "female", "male", keep, label=lab)
        show(gx, "female", "male", "   " + lab)
        parts.append(gx)
    pd.concat(parts).to_csv(f"{OUT}/8_respondent_sex.csv", index=False)
    say("", "   Logit with both sex of head and sex of respondent (plus M2 controls):")
    rows = []
    for yv in ["good_times_ahead", "any_dk", "very_insecure", "fed_optimist"]:
        m = s.glm.fit(yv, x=["black", "female", "woman_answered"] + [svy.Cat(c, ref=r) for c, r in CONTROLS],
                      family="binomial", drop_nulls=True)
        tab = m.to_polars().to_pandas().set_index("term")
        for term in ("female", "woman_answered"):
            r = tab.loc[term]
            rows.append(dict(outcome=yv, term=term, odds_ratio=np.exp(r.est), or_low=np.exp(r.est - 1.96 * r.se),
                             or_high=np.exp(r.est + 1.96 * r.se), p=r.p_value))
            say(f"     {yv:18s} {term:15s} OR {np.exp(r.est):5.2f} [{np.exp(r.est-1.96*r.se):4.2f},{np.exp(r.est+1.96*r.se):5.2f}] p={r.p_value:.4f}")
    pd.DataFrame(rows).to_csv(f"{OUT}/8_respondent_sex_models.csv", index=False)

    open(f"{OUT}/report.txt", "w").write("\n".join(REPORT) + "\n")
    print(f"\nwrote {OUT}/")


if __name__ == "__main__":
    main()
