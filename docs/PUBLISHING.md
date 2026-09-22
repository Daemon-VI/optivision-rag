# Publishing to PyPI and npm

The project ships twice from one version number:

- **PyPI `optivision-rag`** is the real package: the library and the `optivision` CLI.
- **npm `optivision-rag`** is a 5 kB launcher in `npm/`. `npx optivision-rag` creates a
  private virtualenv in the user cache folder, pip-installs the PyPI release with the
  *same version*, and forwards every argument to it.

So **always publish to PyPI first**. An npm release whose PyPI twin does not exist yet
fails on every user's first run.

## One-time setup

1. PyPI: create an account at https://pypi.org, turn on 2FA, and make an API token
   (Account settings -> API tokens). Give it "entire account" scope for the first
   upload, then replace it with a token scoped to `optivision-rag`.
2. npm: create an account at https://www.npmjs.com with 2FA, then run `npm login`.
3. Optional dry run: https://test.pypi.org takes the same upload with
   `--repository testpypi`.

## Each release

1. Bump the version in **both** `src/optivision/__init__.py` (`__version__`) and
   `npm/package.json`. `tests/test_release_versions.py` fails if they differ.
2. Check and build:

   ```bash
   pip install build twine
   pytest -q && ruff check src tests
   rm -rf dist build
   python -m build
   twine check dist/*
   ```

3. Test the npm launcher against the fresh wheel before anything is public:

   ```bash
   cd npm
   OPTIVISION_PIP_SPEC="optivision-rag[corpus] @ file:///$(cygpath -m "$(ls ../dist/*.whl)")" npm test
   cd ..
   ```

   (On Linux/macOS drop `cygpath -m` and use the absolute wheel path.)

4. Upload to PyPI. Username `__token__`, password is the API token:

   ```bash
   twine upload dist/*
   ```

5. Publish the launcher, then run the smoke test against the real PyPI release:

   ```bash
   cd npm
   npm publish --access public
   npm test
   ```

6. Tag the release: `git tag v0.1.0 && git push origin v0.1.0`.

A version number can never be re-uploaded to PyPI or npm, even after deleting it. If
something is wrong after upload, fix it and release the next patch version.
