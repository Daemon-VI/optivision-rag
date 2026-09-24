# Releasing to PyPI and npm

The project ships twice from one version number:

- **PyPI `optivision-rag`** is the real package: the library and the `optivision` CLI.
- **npm `optivision-rag`** is a small launcher in `npm/`. `npx optivision-rag` creates a private
  virtualenv in the user cache folder, pip-installs the PyPI release with the *same version*, and
  forwards every argument to it. It ships no VLM code of its own (torch and colpali-engine have no
  JS equivalent).

**PyPI must go first.** An npm release whose PyPI twin does not exist yet fails on every user's
first run. `publish.yml` enforces the order (`npm` needs `pypi`).

Releases are published by `.github/workflows/publish.yml` through trusted publishing on both
registries: each registry trusts this one workflow in this one repository, so no API token exists
anywhere, not in repository secrets and not on anyone's machine.

## One-time setup

1. **On pypi.org**, open *Account settings -> Publishing* and add a **pending publisher** (for a
   project that does not exist on PyPI yet):

   | Field | Value |
   |---|---|
   | PyPI project name | `optivision-rag` |
   | Owner | `Daemon-VI` |
   | Repository name | `optivision-rag` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

   A pending publisher does not reserve the name. The first successful upload creates the project.

2. **In the GitHub repository**, open *Settings -> Environments* and create an environment named
   exactly `pypi` (PyPI checks the name). Adding yourself as a required reviewer is optional: it
   makes every upload wait for a click.

3. **On npmjs.com**, open the package's *Settings -> Trusted publishing*, choose GitHub Actions and
   enter owner `Daemon-VI`, repository `optivision-rag`, workflow `publish.yml`, environment blank.
   Then set *Publishing access* to require 2FA and disallow tokens. (npm only configures a trusted
   publisher on an existing package; `optivision-rag` already exists.) Trusted publishing needs
   `repository.url` in `npm/package.json` to match the GitHub repository, and npm CLI 11.5.1 or
   later, which the workflow installs.

4. **Deprecate npm 0.1.0**, which was published by hand from an earlier launcher that pinned PyPI
   `0.1.0`, a version that was never released:

   ```sh
   npm deprecate optivision-rag@0.1.0 "Broken: superseded by 0.1.1. Use npx optivision-rag@latest"
   ```

## Each release

1. Bump the version in **both** places, and keep them identical:
   - `__version__` in `src/optivision/__init__.py` (`pyproject.toml` reads it)
   - `version` in `npm/package.json`

   `tests/test_release_versions.py` fails if they differ, and the publish workflow checks both
   against the tag.

2. Check that everything passes and that both packages build:

   ```sh
   .venv/Scripts/python.exe -m pytest -q
   .venv/Scripts/python.exe -m ruff check src tests
   .venv/Scripts/python.exe -m build
   ```

   To smoke-test the launcher against the wheel you just built, before anything is public (this is
   also what CI does):

   ```sh
   cd npm
   OPTIVISION_PIP_SPEC="optivision-rag[corpus] @ file:///C:/absolute/path/to/dist/optivision_rag-X.Y.Z-py3-none-any.whl" npm test
   ```

3. Commit and push, then wait for CI to pass on `main`.

4. On GitHub, create a release (*Releases -> Draft a new release*) with a **new tag named
   `vX.Y.Z`** that matches the version. Publish the release.

5. The `publish` workflow runs on the published release:
   - `build` checks the tag against both version strings, runs the tests, builds `dist/`.
   - `pypi` uploads `dist/` with `pypa/gh-action-pypi-publish`, which exchanges the workflow's OIDC
     token with PyPI. No password or token is involved.
   - `npm` runs after `pypi` and executes `npm publish` from `npm/`, with a provenance attestation.

   A tag that does not match the version fails `build` before anything is uploaded. Neither
   registry ever accepts the same version twice, even after deletion, so a mistake means bumping to
   the next patch version.

## Verify

After the workflow finishes, from a clean environment:

```sh
pipx install optivision-rag          # or: pipx upgrade optivision-rag
optivision --version                 # should print the new version
npx optivision-rag@X.Y.Z --version   # the npm launcher, which installs the same release
```

<https://pypi.org/project/optivision-rag/> and <https://www.npmjs.com/package/optivision-rag>.

## Manual fallback

If the workflow is unavailable, `pip install build twine`, `python -m build`,
`twine upload dist/*` (username `__token__`, password is a PyPI API token), then
`cd npm && npm publish --access public`. Enter tokens only at the tool's own prompt, never in a
chat or a file.

## VS Code extension

The extension in `vscode-extension/` is versioned and published independently of the library. It
wraps whatever `optivision` CLI is already on the user's machine rather than pinning a version.
See `vscode-extension/PUBLISHING.md`.
