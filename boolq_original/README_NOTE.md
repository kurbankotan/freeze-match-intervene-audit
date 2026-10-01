# Note on this directory

`boolq_original/` is the archived BoolQ study, copied unchanged from the earlier repository
`Adapter-Contribution-Audit` (superseded by this repository). File and directory names are retained for hash
stability. In particular, `results/human_validated_contrast/` and `results/human_validation/` hold the second
screening pass of the polarity contrast set described in Appendix E of the paper; the provenance of its coding files
could not be verified, so the paper reports that subset only as a sensitivity analysis and makes no claim of human
validation. `lcca` is the legacy identifier of the module called LRAA in the paper.

Integrity note: `FILE_MANIFEST_SHA256.txt` in this directory is the original repository's manifest. Three of its
entries were already stale in the original repository and are not errors of this copy: `.gitignore` and
`data/LICENSE.md` were never committed there, and `data/DATASET_CARD.md` was edited after the manifest was written.
Every other listed file matches its recorded SHA-256. The repository's `.gitattributes` (`* -text`) prevents Git from
changing line endings, so the hashes hold on every operating system.
