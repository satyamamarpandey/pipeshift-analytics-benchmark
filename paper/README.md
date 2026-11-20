# paper/

This directory holds only the machine-generated parts of the paper:

- `generated_macros.tex`: every number reported in the paper, as a LaTeX macro;
- `generated_tables/`: the table bodies.

Both are written by `pipeshift/analysis.py`. `pipeshift/verify_numbers.py` recomputes the headline numbers
from the episode records with independent code and compares them with these macros.

The manuscript text is not part of this artifact.
