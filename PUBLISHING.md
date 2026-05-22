# Publishing `fjohansen` to PyPI

Step-by-step guide for releasing a new version. Run all commands from the
repository root (`C:\Users\HP\Documents\xtpmg\fjohansen`).

---

## 0. One-time setup

### a) Install build & upload tools

```powershell
pip install --upgrade build twine
```

### b) Create your PyPI accounts

* **TestPyPI** (staging) — <https://test.pypi.org/account/register/>
* **PyPI** (production) — <https://pypi.org/account/register/>

Enable 2-factor authentication on both, then generate an **API token**:

* PyPI → Account settings → API tokens → *Add API token*
  → scope = "Entire account" (first release) or "Project: fjohansen" (later).
* Save the token starting with `pypi-…` (or `pypi-AgEN…` for TestPyPI).

### c) Configure `~/.pypirc` so `twine` finds the tokens

Create `C:\Users\HP\.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
  username = __token__
  password = pypi-AgEN-...your-prod-token...

[testpypi]
  repository = https://test.pypi.org/legacy/
  username = __token__
  password = pypi-AgEN-...your-test-token...
```

(Token is the literal **password**; the username must be exactly `__token__`.)

---

## 1. Pre-release checks

```powershell
# 1.1 -- bump the version in BOTH places
#       pyproject.toml :  version = "0.1.0"
#       fjohansen/__init__.py : __version__ = "0.1.0"

# 1.2 -- update CHANGELOG.md with the new section

# 1.3 -- run the test suite
python -m pytest tests/ -q

# 1.4 -- run the examples to make sure imports still resolve
python examples/example_section6_jgb.py     # close the plot windows
python examples/example_limit_distributions.py
```

---

## 2. Build the distribution

```powershell
# wipe previous builds first
Remove-Item -Recurse -Force build, dist, *.egg-info -ErrorAction SilentlyContinue

# build sdist + wheel
python -m build
```

You should now see two artefacts in `dist/`:

```
dist/
  fjohansen-0.1.0.tar.gz       # source distribution
  fjohansen-0.1.0-py3-none-any.whl   # wheel
```

Sanity-check them:

```powershell
python -m twine check dist/*
```

Expect `PASSED` for both.

---

## 3. Upload to **TestPyPI** first (recommended)

```powershell
python -m twine upload --repository testpypi dist/*
```

Then verify in a fresh virtual environment:

```powershell
python -m venv .venv-test
.\.venv-test\Scripts\Activate.ps1
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            fjohansen
python -c "import fjohansen as fj; print(fj.__version__); fj.sample_jgb_data(60)"
deactivate
Remove-Item -Recurse -Force .venv-test
```

If the import works and the version is correct, you are ready for the real
upload.

---

## 4. Upload to **PyPI** (production)

```powershell
python -m twine upload dist/*
```

Open <https://pypi.org/project/fjohansen/> and confirm the new release.

Anyone can now install with

```bash
pip install fjohansen
```

---

## 5. Tag the release on GitHub

```powershell
git add .
git commit -m "Release v0.1.0"
git tag -a v0.1.0 -m "fjohansen v0.1.0"
git push origin main --tags
```

Then on GitHub, *Releases → Draft a new release → tag = v0.1.0*, paste the
relevant section of `CHANGELOG.md`, and publish.

---

## 6. Subsequent releases

For every new version:

1. Bump `version` in `pyproject.toml` and `__version__` in
   `fjohansen/__init__.py`.
2. Add a new `## [x.y.z] - YYYY-MM-DD` section to `CHANGELOG.md`.
3. `python -m pytest -q`.
4. `Remove-Item -Recurse -Force build, dist, *.egg-info; python -m build`.
5. `python -m twine upload dist/*`.
6. `git tag -a vX.Y.Z -m "..."; git push --tags`.

---

## 7. Optional: automate with GitHub Actions

A minimal `.github/workflows/publish.yml` (uses the trusted-publisher flow,
no token in the repo):

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write    # required for trusted publishing
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install --upgrade build
      - run: python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

Pair this with a "Trusted publisher" entry on PyPI (Account → Publishing →
Add a new pending publisher).

---

## Troubleshooting

| Symptom                                                | Fix                                                                          |
|--------------------------------------------------------|------------------------------------------------------------------------------|
| `ERROR: File already exists`                           | You cannot overwrite a released file. Bump the version and rebuild.          |
| `Invalid distribution metadata`                        | `python -m twine check dist/*` — it points to the offending line.            |
| Tests fail because plot windows pop up                 | Run with `MPLBACKEND=Agg python -m pytest -q`.                               |
| Wheel contains unintended files (`__pycache__`, etc.)  | Check `MANIFEST.in` — the `prune`/`recursive-exclude` lines must cover them. |
| `400 Client Error: invalid token`                      | Regenerate the API token in PyPI and update `~/.pypirc`.                     |

---

## File checklist before tagging a release

* [ ] `pyproject.toml` — version bumped
* [ ] `fjohansen/__init__.py` — `__version__` bumped to match
* [ ] `CHANGELOG.md` — new section added
* [ ] `README.md` — install command still correct, examples up-to-date
* [ ] `MANIFEST.in` — covers any new top-level files
* [ ] Tests pass: `python -m pytest -q`
* [ ] `python -m build` produces a clean wheel + sdist
* [ ] `python -m twine check dist/*` is green
