"""
Figures for the attitudes paper.  Run after analysis_attitudes.py and analysis_attitudes_2.py.

    python3 figures_attitudes.py    # -> output/attitudes/fig1..fig6 .png

Every mark is a weighted percentage from svy with its 95% range (estimate +/- 1.96 SE).
Course chart rules: bold factual title, grey units line, no gridlines, n shown.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

O = "output/attitudes"
BLUE, ORANGE, BLACK, GREY = "#2a78d6", "#eb6834", "#222222", "#6b6b6b"
WHITE_C, BLK_C = "#1D4098", "#C00000"       # course colours for the race categories
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.spines.left": False, "figure.dpi": 200,
                     "savefig.bbox": "tight", "savefig.facecolor": "white"})

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


def head(ax, title, units):
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11.5, pad=24)
    ax.text(0, 1.0, units, transform=ax.transAxes, color=GREY, fontsize=9, va="bottom")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=GREY)
    ax.spines["bottom"].set_color(GREY)


def pair(df, outcomes, a, b, ca, cb, la, lb, title, units, path, xmax=65):
    df = df[df.subset == "all"].set_index("outcome").loc[outcomes]
    fig, ax = plt.subplots(figsize=(8.4, 0.62 * len(outcomes) + 1.6))
    y = np.arange(len(outcomes))[::-1]
    for i, (o, r) in enumerate(df.iterrows()):
        ax.plot([r[f"pct_{a}"], r[f"pct_{b}"]], [y[i], y[i]], color="#cfcfcf", lw=1.5, zorder=1)
        for g, c, dy in ((a, ca, 0.0), (b, cb, 0.0)):
            ax.errorbar(r[f"pct_{g}"], y[i] + dy, xerr=1.96 * r[f"se_{g}"], fmt="o", color=c, ms=7,
                        elinewidth=1.6, capsize=0, zorder=3, markeredgecolor="white", markeredgewidth=1)
        star = "" if r.p >= 0.05 else ("  gap %+.0f pts" % r.diff_pts)
        ax.text(xmax + 1, y[i], f"{r[f'pct_{a}']:.0f} vs {r[f'pct_{b}']:.0f}{star}", va="center", fontsize=8.5,
                color=BLACK if r.p < 0.05 else GREY)
    ax.set_yticks(y); ax.set_yticklabels([LAB[o] for o in outcomes])
    ax.set_xlim(0, xmax); ax.set_ylim(-0.7, len(outcomes) - 0.3)
    na, nb = int(df[f"n_{a}"].max()), int(df[f"n_{b}"].max())
    ax.plot([], [], "o", color=ca, label=f"{la} (n up to {na:,})"); ax.plot([], [], "o", color=cb, label=f"{lb} (n up to {nb:,})")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=9)
    head(ax, title, units)
    fig.savefig(path); plt.close(fig)


# Figure 1: outlook by who was speaking
t = pd.read_csv(f"{O}/9_outlook_by_speaker.csv")
order = ["man answering, male-headed unit", "woman answering, male-headed unit", "woman heading her own unit"]
names = ["Man answering for a male-headed unit", "Woman answering for a male-headed unit", "Woman heading her own unit"]
cols = [BLACK, BLUE, ORANGE]
ans = [("good_times_ahead", "Good times"), ("middle_times_ahead", "In between"), ("bad_times_ahead", "Bad times"), ("dk_times_ahead", "\"Don't know\"")]
fig, ax = plt.subplots(figsize=(8.4, 4.3))
for j, (lv, nm, c) in enumerate(zip(order, names, cols)):
    for i, (o, _) in enumerate(ans):
        r = t[(t.level == lv) & (t.outcome == o)].iloc[0]
        x = i + (j - 1) * 0.22
        ax.errorbar(x, r.pct, yerr=1.96 * r.se, fmt="o", color=c, ms=8, elinewidth=1.8, markeredgecolor="white", markeredgewidth=1)
        ax.text(x, r.pct + 1.96 * r.se + 1.2, f"{r.pct:.0f}", ha="center", fontsize=8.5, color=BLACK)
    n = int(t[t.level == lv].n.iloc[0])
    ax.plot([], [], "o", color=c, label=f"{nm} (n = {n:,})")
ax.set_xticks(range(4)); ax.set_xticklabels([a[1] for a in ans]); ax.set_ylim(0, 58); ax.set_yticks([0, 10, 20, 30, 40, 50])
ax.tick_params(axis="y", colors=GREY); ax.spines["left"].set_visible(True); ax.spines["left"].set_color(GREY)
ax.legend(frameon=False, loc="upper right", fontsize=9)
head(ax, "The outlook a spending unit reported depended on who answered",
     "Expected business conditions over the next 12 months, % of ascertained answers, with 95% range; early 1950")
fig.savefig(f"{O}/fig1_outlook_by_speaker.png"); plt.close(fig)

# Figure 2: sex of head
pair(pd.read_csv(f"{O}/3_sex_gaps.csv"),
     ["good_times_ahead", "fed_optimist", "dk_times_ahead", "any_dk", "very_insecure", "insecure", "worse_off", "better_off"],
     "female", "male", ORANGE, BLACK, "Female head", "Male head",
     "Women heads were less hopeful and less sure, not likelier to be worse off",
     "% of spending units with 95% range; the gap is printed where it differs from zero at the 5% level",
     f"{O}/fig2_sex_of_head.png")

# Figure 3: race
pair(pd.read_csv(f"{O}/2_race_gaps.csv"),
     ["worse_off", "better_off", "dk_own_income", "very_insecure", "prices_will_fall", "good_times_ahead", "dk_times_ahead", "plans_durable"],
     "black", "white", BLK_C, WHITE_C, "Black", "White",
     "Black units reported worse finances but the same national outlook",
     "% of spending units with 95% range; the gap is printed where it differs from zero at the 5% level",
     f"{O}/fig3_race.png", xmax=70)

# Figure 4: odds ratios with controls
m = pd.read_csv(f"{O}/4_models.csv"); m = m[m.model.str.startswith("M2")]
outs = ["good_times_ahead", "fed_optimist", "any_dk", "dk_times_ahead", "very_insecure", "insecure", "worse_off", "plans_durable"]
fig, ax = plt.subplots(figsize=(8.4, 5.6)); y = np.arange(len(outs))[::-1]
for term, c, dy, lab in (("female", ORANGE, 0.14, "Female head (vs male head)"), ("black", BLK_C, -0.14, "Black (vs white)")):
    for i, o in enumerate(outs):
        r = m[(m.outcome == o) & (m.term == term)].iloc[0]
        ax.plot([r.or_low, r.or_high], [y[i] + dy] * 2, color=c, lw=1.8)
        ax.plot(r.odds_ratio, y[i] + dy, "o", color=c, ms=7, markeredgewidth=1.5,
                markerfacecolor=c if r.p < 0.05 else "white", markeredgecolor=c)
    ax.plot([], [], "o", color=c, label=lab)
ax.axvline(1, color=GREY, lw=1); ax.set_xscale("log"); ax.set_xticks([0.33, 0.5, 1, 2, 3, 5]); ax.set_xticklabels(["0.33", "0.5", "1", "2", "3", "5"])
ax.minorticks_off(); ax.set_yticks(y); ax.set_yticklabels([LAB[o] for o in outs]); ax.set_xlim(0.28, 5.4)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)
head(ax, "Other things equal, the sex gap stays and the race gap mostly goes",
     "Odds ratios, 95% range, log scale; controls: income, age, occupation, place, marriage; hollow = not different from 1")
fig.savefig(f"{O}/fig4_odds_ratios.png"); plt.close(fig)

# Figure 5: very insecure by liquid assets and sex of head
t = pd.read_csv(f"{O}/11_insecurity_by_liquid_assets.csv"); t = t[t.outcome == "very_insecure"]
liq = ["none", "$1-499", "$500 or more"]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for sx, c, dx, lab in (("female", ORANGE, 0.1, "Female head"), ("male", BLACK, -0.1, "Male head")):
    for i, l in enumerate(liq):
        r = t[t.level == f"{sx} head | {l}"].iloc[0]
        ax.errorbar(i + dx, r.pct, yerr=1.96 * r.se, fmt="o", color=c, ms=8, elinewidth=1.8, markeredgecolor="white", markeredgewidth=1)
        ax.text(i + dx + (0.07 if dx > 0 else -0.07), r.pct, f"{r.pct:.0f}%\nn={int(r.n)}", ha="left" if dx > 0 else "right", va="center", fontsize=8.5)
    ax.plot([], [], "o", color=c, label=lab)
ax.set_xticks(range(3)); ax.set_xticklabels(["No liquid assets", "$1-499", "$500 or more"]); ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, 46)
ax.tick_params(axis="y", colors=GREY); ax.spines["left"].set_visible(True); ax.spines["left"].set_color(GREY); ax.legend(frameon=False, loc="upper right")
head(ax, "Savings bought security for everyone, and less of it for women", "% of spending units feeling \"very insecure\", with 95% range, by liquid assets held in early 1950")
fig.savefig(f"{O}/fig5_insecurity_liquid_assets.png"); plt.close(fig)

# Figure 6: whose answers made up each group
t = pd.read_csv(f"{O}/5_composition.csv"); base = t.pct_white_male.iloc[0]; t = t.iloc[1:].sort_values("pct_white_male")
short = {"optimists (good times + income holds/rises)": "Good times ahead and income to hold or rise", "expect good times": "Expect good times",
         "better off than a year ago": "Better off than a year ago", "plan a durable purchase": "Plan to buy a durable good",
         "don't know about times ahead": "\"Don't know\" about times ahead", "very insecure": "Very insecure", "worse off": "Worse off than a year ago"}
fig, ax = plt.subplots(figsize=(8.0, 3.9)); y = np.arange(len(t))
ax.axvline(base, color=GREY, lw=1); ax.text(base - 0.4, 2.5, f"all spending units: {base:.0f}%", color=GREY, fontsize=8.5, ha="right")
ax.errorbar(t.pct_white_male, y, xerr=1.96 * t.se, fmt="o", color=BLUE, ms=8, elinewidth=1.8, markeredgecolor="white", markeredgewidth=1)
for yi, v in zip(y, t.pct_white_male):
    ax.text(92.5, yi, f"{v:.0f}%", va="center", fontsize=9)
ax.set_yticks(y); ax.set_yticklabels([short[g] for g in t.group]); ax.set_xlim(58, 92); ax.set_ylim(-0.6, len(t) - 0.1)
head(ax, "Confident answers came mostly from white, male-headed units", "% of spending units giving each answer that were white and male-headed, with 95% range")
fig.savefig(f"{O}/fig6_composition.png"); plt.close(fig)
print("figures written to", O)
