# Anonymized supplementary bundle

This is the anonymized version of the project repository, prepared for
double-blind review.

Changes from the author-facing repository:

- The non-anonymized paper source (`paper/paper.tex`, `paper/paper.pdf`) is
  removed. `paper/neurips/` contains the anonymized submission source.
- Account names, institution names, and cloud-notebook paths are replaced with
  neutral placeholders throughout, including inside the notebook outputs.
- The citation block naming the author is removed from the README.

The notebooks retain their execution outputs. This is deliberate: that stdout
is the only surviving record of the original per-task results and is the
source from which `results_reconstructed/` was rebuilt. 773 per-task markers
are preserved so the reconstruction remains auditable.
