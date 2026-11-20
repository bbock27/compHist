"""
Supporting analyses for the attitudes paper (run after analysis_attitudes.py).

    python3 analysis_attitudes_2.py   # -> output/attitudes/9_*.csv ... 15_*.csv, report_2.txt

Same data preparation and svy design as analysis_attitudes.py.
"""
import warnings
import numpy as np
import pandas as pd
import svy
from analysis_attitudes import prepare, gaps, show, say, REPORT, OUT, ind
from scf_survey import _sample

warnings.filterwarnings("ignore")


def by(s, d, y, g, where=None, mask=None):
    """Weighted % (SE) and unweighted n of indicator y within levels of g."""
    t = s.estimation.mean(y, by=g, where=where, drop_nulls=True).to_polars().to_pandas()
    m = d[y].notna() & d[g].notna() & (True if mask is None else mask)
    n = d[m].groupby(g).size()
    t["n"] = t[g].map(n).fillna(0).astype(int)
    return pd.DataFrame(dict(outcome=y, group=g, level=t[g], pct=100 * t.est, se=100 * t.se, n=t.n))


def dist(s, d, var, g, top=8, drop=()):
    """Distribution of a labelled variable within levels of g (svy means of indicators)."""
    levels = [v for v in d[var].value_counts().index if v not in drop][:top]
    out = []
    ok = ~d[var].isin(list(drop)) & d[var].notna()
    for lv in levels:
        d["_i"] = (d[var] == lv).astype(float).where(ok)
        t = by(_sample(d, "weight"), d, "_i", g)
        t["answer"] = lv
        out.append(t)
    r = pd.concat(out)
    return r.pivot(index="answer", columns="level", values="pct").loc[levels].round(1), int(ok.sum())


def main():
    d = prepare()
    rs = d["sex_respondent"].where(d["sex_respondent"].isin(["male", "female"]))
    d["speaker"] = np.select(
        [(d.sex2 == "male") & (rs == "male"), (d.sex2 == "male") & (rs == "female"), (d.sex2 == "female")],
        ["man answering, male-headed unit", "woman answering, male-headed unit", "woman heading her own unit"], default=None)
    d["speaker"] = d["speaker"].where(d["speaker"].notna(), None)
    d["middle_times_ahead"] = ind(d.times_next_12_months, ["middle position"])
    d["liq3"] = pd.cut(d.liquid_assets, [-1, 0, 499, 1e12], labels=["none", "$1-499", "$500 or more"]).astype(object)
    d["inc_up_25"] = (d.income_1949_vs_1948 == "much larger in 1949 (+25% or more)")
    d["price_exp"] = d.prices_expected_1950.where(d.prices_expected_1950.isin(
        ["prices will go down", "stay about the same", "prices will go up"]))
    s = _sample(d, "weight")
    say("SUPPORTING ANALYSES (svy, weights only)")

    say("", "9. OUTLOOK FOR THE NEXT 12 MONTHS BY WHO WAS SPEAKING (% of ascertained answers)")
    parts = [by(s, d, y, "speaker") for y in ["good_times_ahead", "middle_times_ahead", "bad_times_ahead", "dk_times_ahead"]]
    t = pd.concat(parts); t.to_csv(f"{OUT}/9_outlook_by_speaker.csv", index=False)
    for lv in t.level.unique():
        r = t[t.level == lv]
        say(f"  {lv:36s} n={r.n.iloc[0]:<5d} " + "  ".join(f"{o.split('_')[0]} {p:4.1f} ({e:.1f})" for o, p, e in zip(r.outcome, r.pct, r.se)))

    say("", "10. MALE-HEADED UNITS: HOW OFTEN A WOMAN ANSWERED, BY TYPE OF UNIT")
    mh = d.sex2 == "male"
    d["wa_mh"] = d.woman_answered.where(mh)
    s = _sample(d, "weight")
    parts = []
    for g in ["inc3", "age_head", "occupation_group", "place_size", "married"]:
        t = by(s, d, "wa_mh", g); parts.append(t)
        say(f"  by {g}: " + "; ".join(f"{l}: {p:.0f}% (n={n})" for l, p, n in zip(t.level, t.pct, t.n) if n >= 30))
    pd.concat(parts).to_csv(f"{OUT}/10_who_answered.csv", index=False)

    say("", "11. INSECURITY BY LIQUID ASSETS AND SEX OF HEAD")
    d["sex_liq"] = np.where(d.sex2.notna() & d.liq3.notna(), d.sex2.astype(str) + " head | " + d.liq3.astype(str), None)
    s = _sample(d, "weight")
    parts = []
    for y in ["insecure", "very_insecure", "good_times_ahead", "any_dk"]:
        t = by(s, d, y, "sex_liq"); parts.append(t)
        say(f"  {y}: " + "; ".join(f"{l}: {p:.1f} ({e:.1f}) n={n}" for l, p, e, n in zip(t.level, t.pct, t.se, t.n)))
    pd.concat(parts).to_csv(f"{OUT}/11_insecurity_by_liquid_assets.csv", index=False)
    for lq in ["none", "$1-499", "$500 or more"]:
        g = gaps(s, d, "sex2", "female", "male", ["very_insecure", "insecure"], where=svy.col("liq3") == lq, mask=d.liq3 == lq, label=lq)
        show(g, "female", "male", f"   liquid assets {lq}: female minus male head")
    t = by(s, d, "insecure", "liq3"); say("  all units, insecure by liquid assets: " + "; ".join(f"{l}: {p:.1f} ({e:.1f})" for l, p, e in zip(t.level, t.pct, t.se)))
    t = by(s, d, "very_insecure", "liq3"); say("  all units, very insecure by liquid assets: " + "; ".join(f"{l}: {p:.1f} ({e:.1f})" for l, p, e in zip(t.level, t.pct, t.se)))
    t = by(s, d, "insecure", "income_1949_class"); say("  all units, insecure by income class: " + "; ".join(f"{l}: {p:.0f}" for l, p in zip(t.level, t.pct)))

    say("", "12. REASONS GIVEN (first reason; % of those giving a codable reason), by sex of head")
    blanks = ("no reason / no second reason", "not ascertained", "none / no second reason", "no second frame of reference")
    for var, title in [("times_next_reason_1", "why times will be good or bad"), ("security_frame_1", "what security depends on"),
                       ("fin_change_reason_1", "why better or worse off")]:
        tab, n = dist(s, d, var, "sex2", drop=blanks)
        blank = 100 * np.average(d[var].isin(blanks)[d.sex2 == "female"], weights=d.weight[d.sex2 == "female"]), \
            100 * np.average(d[var].isin(blanks)[d.sex2 == "male"], weights=d.weight[d.sex2 == "male"])
        say(f"  {title} ({var}); no codable reason: female {blank[0]:.0f}%, male {blank[1]:.0f}%; n with reason = {n}")
        say(tab.to_string())
        tab.to_csv(f"{OUT}/12_reasons_{var}.csv")
    tab, n = dist(s, d, "times_next_reason_1", "race2", drop=blanks); say("  why times will be good or bad, by race"); say(tab.to_string())
    tab, n = dist(s, d, "fin_change_reason_1", "race2", drop=blanks); say("  why better or worse off, by race"); say(tab.to_string())

    say("", "13. PROFILES (weighted)")
    rows = []
    for g in ["sex2", "race2"]:
        for lv in d[g].dropna().unique():
            w = svy.col(g) == lv
            sub = d[d[g] == lv]
            r = dict(group=lv, n=len(sub))
            r["median_income_1949"] = s.estimation.median("income_1949_neg0", where=w, drop_nulls=True).to_polars()["est"][0]
            r["mean_income_1949"] = s.estimation.mean("income_1949_neg0", where=w, drop_nulls=True).to_polars()["est"][0]
            r["median_liquid_assets"] = s.estimation.median("liquid_assets", where=w, drop_nulls=True).to_polars()["est"][0]
            for nm, cond in [("pct_no_liquid_assets", sub.liquid_assets == 0), ("pct_65_plus", sub.age_head == "65 or over"),
                             ("pct_under_35", sub.age_head.isin(["18-24", "25-34"])), ("pct_not_married", sub.married == "not married"),
                             ("pct_children", sub.children_under_18.astype(str).str.extract(r"(\d+)")[0].fillna("0").astype(int) > 0),
                             ("pct_metro", sub.place_size.isin(["metropolitan area", "metropolitan rural"])),
                             ("pct_secondary_unit", sub.is_primary_su == 0), ("pct_income_under_2000", sub.income_1949_neg0 < 2000),
                             ("pct_retired_or_not_working", sub.occupation_group.isin(["retired", "students, housewives, protective service", "unemployed"]))]:
                r[nm] = 100 * np.average(cond.fillna(False), weights=sub.weight)
            rows.append(r)
            say("  " + "; ".join(f"{k}={v:.0f}" if isinstance(v, (int, float, np.floating)) else f"{k}={v}" for k, v in r.items()))
    pd.DataFrame(rows).to_csv(f"{OUT}/13_profiles.csv", index=False)
    for g in ["sex2", "race2"]:
        tab, n = dist(s, d, "occupation_group", g, top=12); say(f"  occupation group by {g} (%)"); say(tab.to_string())

    say("", "14. TWO FINDINGS FOR THE WHOLE SAMPLE")
    t = by(s, d, "worse_off", "income_1949_vs_1948")
    say("  worse off, by change in annual income 1948-49: " + "; ".join(f"{l}: {p:.1f} ({e:.1f}) n={n}" for l, p, e, n in zip(t.level, t.pct, t.se, t.n)))
    t.to_csv(f"{OUT}/14_worse_off_by_income_change.csv", index=False)
    d["pe_inc"] = np.where(d.price_exp.notna() & d.inc3.notna(), d.inc3.astype(str) + " | " + d.price_exp.astype(str), None)
    s = _sample(d, "weight")
    t = by(s, d, "plans_durable", "price_exp"); t2 = by(s, d, "plans_durable", "pe_inc")
    say("  plans a durable purchase, by price expectation: " + "; ".join(f"{l}: {p:.1f} ({e:.1f}) n={n}" for l, p, e, n in zip(t.level, t.pct, t.se, t.n)))
    say("  same within income: " + "; ".join(f"{l}: {p:.1f} ({e:.1f}) n={n}" for l, p, e, n in zip(t2.level, t2.pct, t2.se, t2.n)))
    pd.concat([t, t2]).to_csv(f"{OUT}/14_plans_by_price_expectation.csv", index=False)
    d["pe2"] = d.price_exp.where(d.price_exp != "stay about the same")
    s = _sample(d, "weight")
    g = gaps(s, d, "pe2", "prices will go down", "prices will go up", ["plans_durable", "good_times_ahead"])
    show(g, "prices will go down", "prices will go up", "   expecting lower prices minus expecting higher prices")
    t = by(s, d, "good_times_ahead", "plans_durable".replace("plans_durable", "price_exp"))
    for y, g_ in [("plans_durable", "fin_position_vs_year_ago"), ("plans_durable", "times_next_12_months"),
                  ("good_times_ahead", "fin_position_vs_year_ago")]:
        t = by(s, d, y, g_); say(f"  {y} by {g_}: " + "; ".join(f"{l}: {p:.1f} ({e:.1f})" for l, p, e in zip(t.level, t.pct, t.se)))

    say("", "15. DISTRIBUTIONS OF THE ATTITUDE QUESTIONS, ALL UNITS (weighted %, all answers incl. not ascertained)")
    for var in ["fin_position_vs_year_ago", "times_now", "times_vs_year_ago", "times_next_12_months", "income_expected_year_ahead",
                "financial_security", "prices_expected_1950", "durable_prices_expected", "job_chances"]:
        v = (d.groupby(var).weight.sum() / d.weight.sum() * 100).round(1).sort_values(ascending=False)
        say(f"  {var}: " + "; ".join(f"{k} {x}" for k, x in v.items()))
    say(f"  interview dates: " + "; ".join(f"{k} {x}" for k, x in d.interview_date.value_counts().items()))
    open(f"{OUT}/report_2.txt", "w").write("\n".join(REPORT) + "\n")


if __name__ == "__main__":
    main()
