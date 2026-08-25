# NeurIPS 2026 Workshop SLM-Agents — anonymized submission

**Submit as: Long paper (up to 6 content pages).** Deadline Aug 29, 2026,
6:30:00 PM (= 13:00 UTC).

## Status

    pdflatex main && pdflatex main

0 errors, 0 undefined references. Current layout:

  * pp. 1-6   content  (exactly at the 6-page long-paper limit)
  * p. 7      references
  * pp. 8-15  NeurIPS paper checklist

References and checklist are excluded from the limit. **There is no slack
left** — any text you add pushes content onto page 7 and risks desk
rejection. Re-check pagination after every edit.

## Track option

    \usepackage[dblblindworkshop]{neurips_2026}
    \workshoptitle{SLM-Agents: 1st NeurIPS Workshop on SLMs for Agentic Systems}

The workshop is double-blind, so this is correct. Do not use the bare default
(that is the main track) or `sglblindworkshop` (not anonymous). Line numbers
and the "Submitted to ... Do not distribute." footer are correct for
submission mode.

## Verified

  * No occurrence of author name, institution, email, GitHub URL, cloud notebook or
    W&B strings in the compiled PDF
  * PDF Author and Title metadata fields are empty
  * Official `neurips_2026.sty`, unmodified; no margin/font/spacing changes
  * Checklist uses the official question wording with the instruction block
    removed as the kit directs; all 16 answered with the provided macros

## Remaining before you submit

1. Replace `ANONYMIZED-REPO-URL` with an https://anonymous.4open.science link.
   Not a GitHub URL — it contains the author username.
2. Sanitize whatever you link or upload as supplementary. The OpenReview form
   requires *linked* materials to be anonymized too. In the repo,
   `notebooks/` contains a W&B entity name and `/workdir/` paths, and
   `README.md` has a citation block with the author name. Strip these or
   exclude those files from the anonymized bundle.
3. OpenReview also needs: keywords, primary subject area, TL;DR (optional),
   abstract (paste from the PDF), a reviewer nomination (as sole author you
   must nominate yourself and accept the up-to-three-reviews commitment), and
   a license selection.

## Files

    main.tex          paper, anonymized, inline bibliography
    checklist.tex     official checklist, answered
    neurips_2026.sty  official style file (unmodified)
