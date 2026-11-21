"""
Figures for the attitudes paper.  Run after analysis_attitudes.py and analysis_attitudes_2.py.

    python3 figures_attitudes.py    # -> output/attitudes/fig1..fig6 .png

Every mark is a weighted percentage from svy; the thin line through it is the 95% range
(estimate +/- 1.96 SE).  Formatting follows the Assignment 4 figures: bold descriptive title,
grey units line, source line at the bottom, no legend box (series are labelled directly),
no gridlines, only a light baseline.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

O = "output/attitudes"
BLACK = "black"                              # text only
MALE_C, ORANGE, DARK_BLUE = "#3D405B", "#E07A5F", "#3A9188"   # men, women heading a unit, wives answering
GRAY, LIGHT_GRAY = "#7f7f7f", "#c8c8c8"
WHITE_C, BLK_C = "#E9A03B", "#6A4C93"       # race categories
SOURCE = "Source: 1950 Survey of Consumer Finances (ICPSR 3612), author's calculations."
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

LAB = {"good_times_ahead": "Expects good times in the next 12 months",
       "bad_times_ahead": "Expects bad times",
       "dk_times_ahead": "\"Don't know\" what times will be like",
       "any_dk": "\"Don't know\" to at least one of three outlook questions",
       "dk_own_income": "\"Don't know\" own income a year ahead (non-farm)",
       "very_insecure": "Feels very insecure financially",
       "insecure": "Feels rather or very insecure",
       "worse_off": "Worse off than a year ago",
       "better_off": "Better off than a year ago",
       "plans_durable": "Plans to buy a durable good in 1950",
       "fed_optimist": "Good times ahead and own income to hold or rise",
       "prices_will_fall": "Expects prices to fall"}


def chart_titles(fig, title, units, source=SOURCE):
    """Bold title at the top, units in gray below it, source at the bottom."""
    title, units = (s.replace("$", r"\$") for s in (title, units))
    fig.text(0.02, 0.97, title, fontsize=13, fontweight="bold", ha="left", va="top")
    fig.text(0.02, 0.915, units, fontsize=10, color=GRAY, ha="left", va="top")
    fig.text(0.02, 0.02, source, fontsize=8, color=GRAY, ha="left", va="bottom")


def clean_axes(ax):
    """No box, no gridlines, no tick marks; a light baseline only."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(LIGHT_GRAY)
    ax.grid(False)
    ax.tick_params(axis="both", length=0, labelsize=11)


def pct_axis(ax, lo, hi, step=10):
    ticks = list(range(lo, hi + 1, step))
    ax.set_xlim(lo, hi); ax.set_xticks(ticks); ax.set_xticklabels([f"{t}%" for t in ticks])


def dot(ax, x, y, se, color, size=140, hollow=False):
    """One estimate with its 95% range."""
    ax.plot([x - 1.96 * se, x + 1.96 * se], [y, y], color=color, linewidth=1.5, zorder=2)
    ax.scatter([x], [y], s=size, zorder=3, linewidths=2,
               facecolors="white" if hollow else color, edgecolors=color)


def save(fig, name):
    fig.savefig(f"{O}/{name}", dpi=300, facecolor="white"); plt.close(fig)


def pair(df, outcomes, a, b, ca, cb, la, lb, title, units, name, xmax):
    """Two groups compared on several yes/no measures, one row per measure."""
    df = df[df.subset == "all"].set_index("outcome").loc[outcomes]
    k = len(outcomes)
    fig, ax = plt.subplots(figsize=(11, 0.62 * k + 2.3))
    fig.subplots_adjust(left=0.40, right=0.95, top=1 - 1.25 / fig.get_figheight(), bottom=0.95 / fig.get_figheight())
    y = np.arange(k)[::-1]
    for i, (o, r) in enumerate(df.iterrows()):
        pa, pb = r[f"pct_{a}"], r[f"pct_{b}"]
        ax.plot([pa, pb], [y[i], y[i]], color=LIGHT_GRAY, linewidth=2, zorder=1)
        dot(ax, pa, y[i], r[f"se_{a}"], ca); dot(ax, pb, y[i], r[f"se_{b}"], cb)
    # label the two series directly, above the first row
    r = df.iloc[0]; pa, pb = r[f"pct_{a}"], r[f"pct_{b}"]
    na, nb = int(df[f"n_{a}"].max()), int(df[f"n_{b}"].max())
    for p, other, c, lab, n in ((pa, pb, ca, la, na), (pb, pa, cb, lb, nb)):
        ax.text(p, y[0] + 0.42, lab, color=c, fontsize=10, fontweight="bold",
                ha="center", va="bottom")
    ax.set_yticks(y); ax.set_yticklabels([LAB[o] for o in outcomes])
    ax.set_ylim(-0.6, k - 0.1); pct_axis(ax, 0, xmax)
    clean_axes(ax); chart_titles(fig, title, units); save(fig, name)


UNITS = "% of spending units"

# Figure 1: outlook by who was speaking
t = pd.read_csv(f"{O}/9_outlook_by_speaker.csv")
series = [("man answering, male-headed unit", "Man answering for a male-headed unit", MALE_C, 0.26),
          ("woman answering, male-headed unit", "Woman answering for a male-headed unit", DARK_BLUE, 0.0),
          ("woman heading her own unit", "Woman heading her own unit", ORANGE, -0.26)]
ans = [("good_times_ahead", "Good times"), ("middle_times_ahead", "In between"),
       ("bad_times_ahead", "Bad times"), ("dk_times_ahead", "\"Don't know\"")]
fig, ax = plt.subplots(figsize=(11, 6))
fig.subplots_adjust(left=0.16, right=0.70, top=0.82, bottom=0.14)
yy = np.arange(len(ans))[::-1]
for lv, nm, c, dy in series:
    n = int(t[t.level == lv].n.iloc[0])
    for i, (o, _) in enumerate(ans):
        r = t[(t.level == lv) & (t.outcome == o)].iloc[0]
        dot(ax, r.pct, yy[i] + dy, r.se, c, size=110)
        ax.text(r.pct + 1.96 * r.se + 1.0, yy[i] + dy, f"{r.pct:.0f}%" + (f"   {nm}" if i == 0 else ""),
                va="center", fontsize=10, color=c, fontweight="bold" if i == 0 else "normal")
ax.set_yticks(yy); ax.set_yticklabels([a[1] for a in ans]); ax.set_ylim(-0.6, len(ans) - 0.4); pct_axis(ax, 0, 60)
clean_axes(ax)
chart_titles(fig, "Expected business conditions for the next 12 months by who answered the interview (early 1950)",
             "% of answers")
save(fig, "fig1_outlook_by_speaker.png")

# Figure 2: sex of head
pair(pd.read_csv(f"{O}/3_sex_gaps.csv"),
     ["good_times_ahead", "fed_optimist", "dk_times_ahead", "any_dk", "very_insecure", "insecure", "worse_off", "better_off"],
     "female", "male", ORANGE, MALE_C, "Female head", "Male head",
     "Attitudes of spending units headed by women and by men (early 1950)", UNITS, "fig2_sex_of_head.png", xmax=60)

# Figure 3: race
pair(pd.read_csv(f"{O}/2_race_gaps.csv"),
     ["worse_off", "better_off", "dk_own_income", "very_insecure", "prices_will_fall", "good_times_ahead", "dk_times_ahead", "plans_durable"],
     "black", "white", BLK_C, WHITE_C, "Black", "White",
     "Attitudes of black and white spending units (early 1950)", UNITS, "fig3_race.png", xmax=70)

# Figure 4: odds ratios with controls
m = pd.read_csv(f"{O}/4_models.csv"); m = m[m.model.str.startswith("M2")]
outs = ["good_times_ahead", "fed_optimist", "any_dk", "dk_times_ahead", "very_insecure", "insecure", "worse_off", "plans_durable"]
fig, ax = plt.subplots(figsize=(11, 6.6))
fig.subplots_adjust(left=0.40, right=0.95, top=0.80, bottom=0.14)
y = np.arange(len(outs))[::-1]
ax.axvline(1, color=LIGHT_GRAY, linestyle="--", zorder=1)
ax.text(1.03, len(outs) - 0.35, "1 = same odds as the comparison group", fontsize=10, color=GRAY, va="center")
for term, c, dy, lab in (("female", ORANGE, 0.16, "Female head (vs male head)"), ("black", BLK_C, -0.16, "Black (vs white)")):
    for i, o in enumerate(outs):
        r = m[(m.outcome == o) & (m.term == term)].iloc[0]
        ax.plot([r.or_low, r.or_high], [y[i] + dy] * 2, color=c, linewidth=1.5, zorder=2)
        ax.scatter([r.odds_ratio], [y[i] + dy], s=110, zorder=3, linewidths=2,
                   facecolors=c if r.p < 0.05 else "white", edgecolors=c)
        if i == 0 and term == "female":   # label above its dot, clear of the reference line
            ax.text(r.odds_ratio, y[i] + dy + 0.2, lab, color=c, fontsize=10, fontweight="bold", ha="center", va="bottom")
        elif i == 0:
            ax.text(r.or_high * 1.06, y[i] + dy, lab, color=c, fontsize=10, fontweight="bold", va="center")
ax.set_xscale("log"); ax.set_xlim(0.28, 5.4); ax.set_xticks([0.33, 0.5, 1, 2, 3, 5]); ax.set_xticklabels(["0.33", "0.5", "1", "2", "3", "5"])
ax.minorticks_off(); ax.set_yticks(y); ax.set_yticklabels([LAB[o] for o in outs]); ax.set_ylim(-0.6, len(outs) - 0.1)
clean_axes(ax)
chart_titles(fig, "Odds of each answer for female-headed and for black spending units, with controls (early 1950)",
             "Odds ratio (log scale)")
save(fig, "fig4_odds_ratios.png")

# Figure 5: very insecure by liquid assets and sex of head
t = pd.read_csv(f"{O}/11_insecurity_by_liquid_assets.csv"); t = t[t.outcome == "very_insecure"]
liq = [("none", "No liquid assets"), ("$1-499", r"\$1–499"), ("$500 or more", r"\$500 or more")]
fig, ax = plt.subplots(figsize=(11, 5.5))
fig.subplots_adjust(left=0.18, right=0.95, top=0.80, bottom=0.15)
y = np.arange(len(liq))[::-1]
for i, (l, _) in enumerate(liq):
    f, mm = (t[t.level == f"{s} head | {l}"].iloc[0] for s in ("female", "male"))
    ax.plot([f.pct, mm.pct], [y[i], y[i]], color=LIGHT_GRAY, linewidth=2, zorder=1)
    for r, c, lab, dy, va in ((f, ORANGE, "Female head: ", 0.2, "bottom"), (mm, MALE_C, "Male head: ", -0.2, "top")):
        dot(ax, r.pct, y[i], r.se, c, size=160)
        ax.text(r.pct, y[i] + dy, (lab if i == 0 else "") + f"{r.pct:.0f}%", color=c, fontsize=10,
                fontweight="bold" if i == 0 else "normal", ha="center", va=va)
ax.set_yticks(y); ax.set_yticklabels([n for _, n in liq]); ax.set_ylim(-0.6, len(liq) - 0.4); pct_axis(ax, 0, 50)
clean_axes(ax)
chart_titles(fig, "Share feeling \"very insecure\" by liquid assets and sex of head (early 1950)",
             "% of spending units")
save(fig, "fig5_insecurity_liquid_assets.png")

# Figure 6: whose answers made up each group
t = pd.read_csv(f"{O}/5_composition.csv"); base = t.pct_white_male.iloc[0]; t = t.iloc[1:].sort_values("pct_white_male")
short = {"optimists (good times + income holds/rises)": "Good times ahead and income to hold or rise", "expect good times": "Expect good times",
         "better off than a year ago": "Better off than a year ago", "plan a durable purchase": "Plan to buy a durable good",
         "don't know about times ahead": "\"Don't know\" about times ahead", "very insecure": "Very insecure", "worse off": "Worse off than a year ago"}
fig, ax = plt.subplots(figsize=(11, 6))
fig.subplots_adjust(left=0.36, right=0.93, top=0.80, bottom=0.14)
y = np.arange(len(t))
ax.axvline(base, color=LIGHT_GRAY, linestyle="--", zorder=1)
ax.text(base - 0.4, len(t) - 0.45, f"All spending units: {base:.0f}%", color=GRAY, fontsize=10, ha="right", va="center")
for yi, (_, r) in zip(y, t.iterrows()):
    c = MALE_C   # one colour: every dot is the same group (white, male-headed units)
    dot(ax, r.pct_white_male, yi, r.se, c)
    ax.text(r.pct_white_male + 1.96 * r.se + 0.5, yi, f"{r.pct_white_male:.0f}%", va="center", fontsize=10, color=c)
ax.set_yticks(y); ax.set_yticklabels([short[g] for g in t.group]); ax.set_ylim(-0.6, len(t) - 0.1); pct_axis(ax, 55, 95)
clean_axes(ax)
chart_titles(fig, "Share of each answer that came from white, male-headed spending units (early 1950)",
             "% of spending units")
save(fig, "fig6_composition.png")
print("figures written to", O)
