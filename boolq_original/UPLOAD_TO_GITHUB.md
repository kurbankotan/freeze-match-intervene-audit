# Updating the public GitHub repository

The repository is https://github.com/kurbankotan/Adapter-Contribution-Audit. Do not upload the unpublished manuscript.

## GitHub website

1. Open the repository and choose **Add file > Upload files**.
2. Open this package folder on your computer.
3. Select the contents inside the prepared artifact folder (not the outer folder itself) and drag them into the upload page.
4. Wait until all files finish uploading. The six new V13 checkpoints are approximately 4.2 MB each and remain below GitHub's per-file web-upload limit.
5. Enter a commit message such as `Add completed V13 mixer and LoRA controls` and choose **Commit changes**.
6. Confirm that the repository root displays `README.md` and the folders `checkpoints`, `data`, `notebooks`, `results`, and `scripts`. The unpublished manuscript must not be present.

Repository software is licensed under MIT. The derived contrast data use CC BY-SA 3.0 with BoolQ attribution in `data/LICENSE.md`.

For the prepared V13 update ZIP, upload the contents inside its `Adapter-Contribution-Audit` folder and allow GitHub to replace the existing root `README.md`, `FILE_MANIFEST_SHA256.txt`, and `UPLOAD_TO_GITHUB.md`. The ZIP intentionally contains no manuscript.

## Git command alternative

Run these commands from inside the package folder after replacing the remote only if needed:

```bash
git init
git add .
git commit -m "Add adapter-contribution reproducibility package"
git branch -M main
git remote add origin https://github.com/kurbankotan/Adapter-Contribution-Audit.git
git push -u origin main
```

If the empty GitHub repository was initialized with a README or license, pull and reconcile that remote commit before pushing rather than force-pushing.
