"""
Survey-design setup for analysing the cleaned 1950 Survey of Consumer Finances with `svy`.

    pip install svy          # https://pypi.org/project/svy/  (needs Python >= 3.11)
    python3 clean_scf1950.py # writes clean/scf1950_clean.csv
    python3 scf_survey.py    # prints a few benchmark estimates with standard errors

Use in an analysis script:

    import svy
    from scf_survey import spending_units, housing, families

    su = spending_units()
    su.estimation.mean("income_1949_neg0", drop_nulls=True).to_polars()
    su.estimation.mean("liquid_assets", by="age_head", drop_nulls=True).to_polars()
    su.categorical.tabulate("owns_stock", "income_1949_class", units="percent", share_of="col")

DESIGN.  Course rule (AS.100.145, Week 8 "Complex Survey Analysis" and Week 9 Lab 8): use the
weights punched on the cards and ignore cluster effects, i.e. a weights-only design

    svy.Design(wgt=<the weight of the card being analysed>)

This reproduces the lab's expected output (mean income $3,308.41, SE 59.55, n = 3,479).

The 1950 survey was in fact an area probability sample (66 sample points), and the first two
digits of the sample book number identify the sample point.  That is kept as the `psu` column, and
`CLUSTERED = True` below switches every sample to svy.Design(wgt=..., psu="psu").  Clustering
roughly doubles the standard error of mean income (about $109 instead of $60); point estimates do
not change.  The default follows the course, and the paper should say so.

Three units of analysis, three weights:
    spending_units()  weight          all 3,512 records
    housing()         weight_housing  nonfarm primary units and single-unit families (card 3 variables)
    families()        weight_family   one record per family (fam_* variables)
For housing and families the records that do not belong (weight 0) are kept in the design but
excluded with a domain flag, as `svy` requires for correct variances: pass `where=IN_SCOPE`
to every estimate on those two samples (see the examples at the bottom of this file).
"""
import polars as pl
import svy

from clean_scf1950 import load

IN_SCOPE = svy.col("in_scope") == 1
CLUSTERED = False   # course rule: weights only.  True = also cluster on the sample point.


def _frame(labeled=False):
    df = load(labeled=labeled)
    df["psu"] = load()["sample_book_number"].str[:2].values if labeled else df["sample_book_number"].str[:2]
    # The 32 negative incomes have no amount punched.  The published 1950 tables put them in the
    # bottom bracket, so this column counts them as $0 instead of dropping them.
    neg = df["income_1949_status"] == "negative"
    df["income_1949_neg0"] = df["income_1949"].where(~neg, 0.0)
    return df


def _sample(df, weight):
    design = svy.Design(wgt=weight, psu="psu") if CLUSTERED else svy.Design(wgt=weight)
    return svy.Sample(pl.from_pandas(df), design)


def spending_units(labeled=False):
    """All 3,512 spending units, weighted by `weight`."""
    df = _frame(labeled)
    df["in_scope"] = 1
    return _sample(df, "weight")


def housing(labeled=False):
    """Housing (card 3) variables.  Estimate with where=IN_SCOPE."""
    df = _frame(labeled)
    df["in_scope"] = (df["weight_housing"] > 0).astype(int)
    df["design_weight"] = df["weight_housing"].where(df["weight_housing"] > 0, df["weight"])
    return _sample(df, "design_weight")


def families(labeled=False):
    """Family (fam_*) variables, one record per family.  Estimate with where=IN_SCOPE."""
    df = _frame(labeled)
    ok = (df["weight_family"] > 0) & df["family_filter"].astype(str).str[0].isin(["1", "2", "5"])
    if labeled:
        ok = (df["weight_family"] > 0) & load()["family_filter"].isin(["1", "2", "5"]).values
    df["in_scope"] = ok.astype(int)
    df["design_weight"] = df["weight_family"].where(df["weight_family"] > 0, df["weight"])
    return _sample(df, "design_weight")


if __name__ == "__main__":
    def show(title, result, cols=("est", "se", "lci", "uci", "n")):
        t = result.to_polars()
        keep = [c for c in t.columns if c in cols or c not in
                ("y_var", "param", "variance", "alpha", "cv", "df", "share_of", "units", "est", "se", "lci", "uci", "n")]
        print(f"\n{title}")
        print(t.select(keep + [c for c in cols if c in t.columns and c not in keep]))

    su = spending_units()
    print(f"spending units: {su.n_records:,} records; design: "
          f"{'weights + sample-point clusters' if CLUSTERED else 'weights only (course rule)'}")
    show("Mean 1949 income, negative incomes missing (course lab: 3,308.41, SE 59.55, n 3,479)",
         su.estimation.mean("income_1949", drop_nulls=True))
    show("Mean 1949 income, negative incomes as $0 (published: $3,270)",
         su.estimation.mean("income_1949_neg0", drop_nulls=True))
    show("Median 1949 income (published: $2,700)",
         su.estimation.median("income_1949_neg0", drop_nulls=True))
    show("Income class, percent of spending units",
         su.categorical.tabulate("income_1949_class", units="percent"))

    h = housing()
    show("Home ownership, nonfarm units with housing data (owns_home = 1)",
         h.estimation.mean("owns_home", where=IN_SCOPE, drop_nulls=True))
    show("Median value of owned homes",
         h.estimation.median("house_value", where=IN_SCOPE, drop_nulls=True))

    f = families()
    show("Mean 1949 family income (negative incomes excluded)",
         f.estimation.mean("fam_income_1949", where=IN_SCOPE, drop_nulls=True))
