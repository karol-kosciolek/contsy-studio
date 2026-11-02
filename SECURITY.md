# Security

Contsy Studio is a local desktop application. The app (port 7860) and its ComfyUI engine (port 8188) listen only on `127.0.0.1`, reject foreign hosts and cross-origin writes, and serve only approved output files. Do not expose either port through a proxy, tunnel or port forwarding.

## Reporting a vulnerability

Report vulnerabilities privately through **Security → Report a vulnerability** on the GitHub repository. Do not open a public issue for a vulnerability. Include the affected version or commit, the steps to reproduce and the impact you observed. Do not attach private photos, voices or generated media.

## Known dependency advisories

The following advisories apply to dependencies at their newest mutually compatible versions (checked on 1 November 2026). Each is blocked by an upstream version bound, not by Contsy. Contsy's own code does not reach the affected functionality. `tests/test_dependency_advisories.py` checks that this remains true.

| Package | Advisory | Why it is not reachable | Blocked by |
| --- | --- | --- | --- |
| Pillow 12.3.0 | [GHSA-22hg-r758-rrw9](https://github.com/advisories/GHSA-22hg-r758-rrw9): infinite loop in the MIC decoder | The MIC decoder requires `olefile`, which is not installed, so Pillow cannot open MIC files. | Gradio 6.31.0 requires `pillow<13`. The fix is in Pillow 13.0.0. |
| fsspec 2025.3.0 | [GHSA-27vj-qcqg-25rc](https://github.com/advisories/GHSA-27vj-qcqg-25rc): template injection in `ReferenceFileSystem` | Contsy never opens fsspec URLs. Its downloads use pinned HTTPS URLs and SHA-256 checks. | datasets 3.6.0 caps fsspec. The fix is in fsspec 2026.6.0. |
| datasets 3.6.0 | [GHSA-379c-qx7v-6h59](https://github.com/advisories/GHSA-379c-qx7v-6h59): path traversal in folder-based builders | Only VoxCPM's training module imports it. Contsy uses VoxCPM for inference only and never loads datasets. | VoxCPM 2.0.3 requires `datasets<4`. The fix is in datasets 5.0.1. |

These pins will be updated when Gradio and VoxCPM publish releases that accept the patched versions.
