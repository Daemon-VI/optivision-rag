# Publishing the OptiVision RAG VS Code extension

The extension publishes to two registries under the **`daemon-vi`** publisher:

- **VS Code Marketplace** — https://marketplace.visualstudio.com/items?itemName=daemon-vi.optivision-rag
- **Open VSX** — https://open-vsx.org/extension/daemon-vi/optivision-rag

Both names were unclaimed as of 2026-09-22 — no rename dance needed.

This pass built and packaged the extension locally only; no CI workflow exists yet (Darkwatch's
`.github/workflows/publish-vscode.yml` at the repo root is a reasonable model to copy later, but
adding it is a repo-root change and out of scope for a `vscode-extension/`-only pass). Until then,
publish by hand as below.

## One-time setup

### Tokens

1. **`VSCE_PAT`** (Marketplace) — from **Azure DevOps**, not the Azure portal:
   - Open https://dev.azure.com/_usersSettings/tokens (create an empty org at
     https://aex.dev.azure.com/ first if it bounces you).
   - Sign in with the Microsoft account that owns (or will own) the `daemon-vi` publisher.
   - **+ New Token** → Organization **All accessible organizations** → Scopes: *Show all scopes* →
     **Marketplace → Manage** → **Create**. Copy it (shown once).
2. **`OVSX_PAT`** (Open VSX) — from https://open-vsx.org (GitHub login):
   - Sign the **Eclipse Open VSX Publisher Agreement** once (account settings).
   - **Settings → Access Tokens → Generate New Token**. Copy it.
   - If a publish ever fails with "namespace not owned", claim it once:
     `npx ovsx create-namespace daemon-vi -p <OVSX_PAT>`.

Never paste a token into a chat, a file, or a commit. Enter it only in the CLI prompt or a
GitHub secrets field.

## Releasing a version

```sh
cd vscode-extension
npm install
npm run compile
npm run lint
npm run package        # produces optivision-rag-X.Y.Z.vsix — sanity-check it locally first
```

Then, from a machine with the tokens above:

```sh
npx vsce publish -p <VSCE_PAT>
npx ovsx publish optivision-rag-X.Y.Z.vsix -p <OVSX_PAT>
```

Bump `version` in `vscode-extension/package.json` first (the Marketplace rejects a duplicate
version) and add a `CHANGELOG.md` entry. This extension's version is independent of the PyPI/npm
library version — see the note in `../docs/RELEASING.md`.

## Verify

- Marketplace API (immediate):
  ```sh
  curl -sX POST https://marketplace.visualstudio.com/_apis/public/gallery/extensionquery \
    -H "Accept: application/json;api-version=7.1-preview.1" -H "Content-Type: application/json" \
    -d '{"filters":[{"criteria":[{"filterType":7,"value":"daemon-vi.optivision-rag"}]}],"flags":914}'
  ```
- Open VSX (indexes a few minutes after publish): `curl -s https://open-vsx.org/api/daemon-vi/optivision-rag/latest`
- Install from either: `code --install-extension daemon-vi.optivision-rag`

## Notes

- The extension is a front end over the `optivision` CLI. It has no `--json` mode, so the panel
  streams raw stdout/stderr rather than parsing structured state; `explain`'s output figures are
  found by listing the `--out` directory afterwards, not by parsing the CLI's own output.
- `media/icon.png` (and its `icon.svg` source) is a placeholder generated for this pass — flat
  navy background, teal nested squares as a compression motif. Replace it with real branding
  before a real publish if one exists by then.
- Built and packaged locally on 2026-09-22; not yet published to either registry.
