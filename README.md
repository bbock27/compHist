# Assignment 5: the 1950 Survey of Consumer Finances

AS.100.145, Introduction to Computational History.

## Papers

- `Optimistic_Consumer_essay.docx`: the historical paper.
- `Technical_Paper.docx`: the technical paper.
- `Optimistic_Consumer_papers.docx`: both papers in one file.

## Data

- `03612-0001-Data-card_image.txt`: the raw punch-card file (ICPSR study 3612).
- `03612-0001-Codebook-card_image.pdf`, `Codebook SCF 1950 Final Version.xlsx`, `codebook_small.txt`: codebooks.

## Reproducing the analysis

Python 3.12 with `svy` 0.37.0, `polars`, `pandas` and `matplotlib`. Run from this folder, in order:

```
python clean_scf1950.py          # raw cards -> clean/*.csv
python analysis_attitudes.py     # group gaps, models, clustered check
python analysis_attitudes_2.py   # speaker, savings and reasons tables
python figures_attitudes.py      # figures
```

Tables, figures and two plain-text reports are written to `output/attitudes/`. `scf_survey.py` holds the shared survey-design helpers.
