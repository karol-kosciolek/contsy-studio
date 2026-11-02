# Contributing

Thank you for helping improve Contsy Studio. The project is free software under the [GNU GPL v3.0 only](LICENSE) (`GPL-3.0-only`).

## Contribution terms

By submitting a contribution (a pull request, patch or code attached to an issue), you agree that:

1. You wrote it yourself or otherwise have the right to submit it under these terms.
2. It is licensed to everyone under the GNU GPL v3.0 only, like the rest of Contsy's own code.
3. You also grant the project owner, Karol Kosciolek, a perpetual, worldwide, non-exclusive, royalty-free and irrevocable license to use, modify, sublicense and relicense your contribution under any terms, including proprietary ones. This lets the owner also offer Contsy under other licenses, for example in a commercial or hosted edition. Your contribution stays available to everyone under the GPL.

Pull requests may be asked to confirm these terms through a contributor license agreement check. [docs/LICENSING.md](docs/LICENSING.md) explains why the project uses these terms.

## Ground rules

- **SFW only.** Contributions that add, unlock or work around adult content, nudity or sexualized depictions are not accepted. Neither are features that weaken the prompt guards or the mandatory visual safety check.
- **No impersonation.** Do not add features for depicting or cloning the face, body or voice of real people without their consent.
- **Pinned downloads.** New models or tools must be pinned to an exact revision, with their size and SHA-256 hash where the provider publishes them. Each must also be listed in `MODEL_LICENSES.md` or `THIRD_PARTY_NOTICES.md`.
- **Third-party code.** Say where any copied or adapted code comes from. Its license must be compatible with the GPL-3.0, its copyright notice must stay in place, and it must be listed in `THIRD_PARTY_NOTICES.md`.
- **License headers.** New source files start with the two `SPDX-` lines used in the existing files.
- **No private data.** Never commit models, LoRAs, generated media, character photos, voices of real people, tokens or local settings.
- **English** for UI text, documentation and new code comments.

## Development

1. Install with `install.bat` (Windows 11, NVIDIA GPU; see the README for hardware).
2. Run the checks from the project folder:

   ```bat
   venv\Scripts\python.exe -m app.setup_windows --check
   venv\Scripts\python.exe -m unittest discover -s tests -v
   ```

3. Add or update tests for every behavior change. Do not run GPU-heavy tests in CI. Describe any manual GPU verification you performed in the pull request.

## Reporting bugs

Open an issue with:
- your GPU model and VRAM, system RAM, NVIDIA driver version and Windows version;
- the screen and mode you used (for example Video → Fast);
- the steps that reproduce the bug;
- the relevant part of `comfyui.log`.

Leave out private photos and personal data. Report security problems privately as described in [SECURITY.md](SECURITY.md).
