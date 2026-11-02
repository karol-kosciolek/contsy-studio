# Licensing

This page explains which license covers which part of Contsy Studio, how Contsy uses ComfyUI and the AI models, and what the licenses allow and require. It is the project's own analysis, written to be checked by a lawyer. It is not legal advice.

## Summary

- **Contsy's own code** is licensed under the **GNU General Public License v3.0 only** (SPDX: `GPL-3.0-only`), © 2026 Karol Kosciolek. ComfyUI uses the same license. The full text is in [LICENSE](../LICENSE).
- **Third-party parts keep their own licenses.** The repository contains a few adapted files (listed below) and no third-party engine, Python packages or model weights. `install.bat` and the in-app model manager download those from their upstream sources onto the user's PC.
- **You may** use Contsy, including commercially, and study, change, share and host it.
- **You must** keep the license and copyright notices. If you share Contsy, changed or not, you must provide its source code under the GPL. Hosting it for others over a network is not sharing a copy, so it carries no source-code duty.
- **The AI models have their own licenses.** Some of them add conditions for commercial or hosted use. See [Models](#models-downloaded-automatically).

## What the repository contains

| Part | License | Copyright |
| --- | --- | --- |
| Contsy's own code and files: `app/`, `tests/`, `tools/`, `install.bat`, `start.bat`, `config/`, documentation | GPL-3.0-only | 2026 Karol Kosciolek |
| `app/lipsync_worker.py`, adapted from LatentSync's `scripts/inference.py` | Apache-2.0 for the original parts, GPL-3.0-only for Contsy's changes | 2024 Bytedance Ltd. and/or its affiliates; 2026 Karol Kosciolek |
| `patches/ComfyUI-Frame-Interpolation/`, changes to an MIT extension | MIT, like the extension it changes | 2023 Fannovel16; 2026 Karol Kosciolek |
| `workflows/*.json`, node graphs following ComfyUI's official workflow templates | GPL-3.0-only for Contsy's adaptations, MIT for the template structure | 2026 Karol Kosciolek; 2023-present Comfy Org |
| Sidebar icons in `app/ui_theme.py`, following Lucide geometry | ISC ([notice](../LICENSES/ISC-Lucide.txt)) | 2026 Lucide Icons and Contributors |
| `assets/voices/*.wav`, four synthetic voices generated with VoxCPM2 from text alone (see [Resolved](#resolved)) | GPL-3.0-only | 2026 Karol Kosciolek (AI-generated) |

Each source file names its license in an SPDX header. [REUSE.toml](../REUSE.toml) covers files that cannot carry a header. License texts are in [LICENSES/](../LICENSES). Apache-2.0, MIT and ISC code may be combined with GPLv3 code. The adapted files keep their original notices, as those licenses require.

## How Contsy uses ComfyUI

ComfyUI is licensed under the GNU General Public License v3.0. Contsy uses it as its generation engine. Here is what Contsy does and does not do with it, checked against the code:

| Question | Answer | Details |
| --- | --- | --- |
| A. Does Contsy copy ComfyUI code? | **No** | No file in the repository contains ComfyUI source code. `workflows/*.json` are node graphs, that is data naming ComfyUI node types and their inputs. Their structure follows the official MIT workflow templates. |
| B. Does Contsy modify ComfyUI? | **No** | The installer checks out the pinned ComfyUI commit unchanged. Contsy's only patch changes the MIT extension ComfyUI-Frame-Interpolation, not ComfyUI. Character LoRAs are stored in ComfyUI's `models/loras` folder, which is data, not code. |
| C. Does Contsy import ComfyUI as a library? | **No** | No Contsy module imports ComfyUI modules (`comfy`, `nodes`, `folder_paths`, `server` and so on). The Contsy app process never loads ComfyUI code. Both programs use the same Python environment (`venv`), but they run as separate processes. |
| D. Does Contsy talk to ComfyUI through its API? | **Yes** | It uses ComfyUI's local HTTP and WebSocket API on `127.0.0.1:8188`: `POST /prompt`, `/upload/image`, `/interrupt` and `/free`; `GET /history`, `/view` and `/system_stats`; and the `/ws` WebSocket. Contsy sends workflow graphs and receives the result files. |
| E. Does ComfyUI run as a separate process? | **Yes** | `app/comfy_process.py` starts ComfyUI's `main.py` as a child process, or connects to an instance that is already running. A seven-line launcher switches Python's certificate store to the Windows one (truststore) and then runs `main.py` unchanged. |
| F. Is ComfyUI downloaded during installation? | **Yes** | `install.bat` runs `app/setup_windows.py`, which clones `Comfy-Org/ComfyUI` at commit `7cebca5` from GitHub on the user's PC and installs its requirements. |
| G. Does Contsy distribute ComfyUI? | **No** | The repository contains no ComfyUI code. Each user downloads ComfyUI directly from its upstream repository. |

### Assessment

1. **Running ComfyUI is unrestricted.** GPLv3 section 2 affirms the unlimited permission to run the unmodified program. Its conditions apply when someone conveys (distributes) ComfyUI or a work based on it. Contsy neither distributes nor modifies ComfyUI.
2. **The API boundary alone does not settle the question.** Separate processes talking through sockets usually indicate separate programs. However, the FSF's GPL FAQ says that programs exchanging complex internal data structures can form one combined program. Contsy cannot work without ComfyUI, and its workflow graphs name ComfyUI node types and inputs. This analysis therefore does not rely on the API alone.
3. **Either way, there is no conflict.** Contsy's own code uses the same license as ComfyUI, GPLv3. Whether the two are separate programs or one combined work, the whole can be shared under GPLv3. The Elastic License 2.0 used before is not GPL-compatible. Under the combined-work reading, that would have been a conflict.
4. **The extensions fit too.** ComfyUI-VideoHelperSuite, comfyui-embedded-docs and comfy-aimdo (GPL-3.0) run inside ComfyUI. ComfyUI-GGUF (Apache-2.0) and ComfyUI-Frame-Interpolation (MIT, including Contsy's patch) are GPLv3-compatible. FFmpeg (a GPL build) runs as a separate program that Contsy calls with command-line arguments.
5. **No architecture change is needed.** A change would be needed only if Contsy distributed ComfyUI, or code based on it, under a license that is not GPL-compatible. It does neither.

### If you distribute a bundle

A ready-to-run package that includes ComfyUI, its extensions, FFmpeg, Python packages or model weights (for example a portable ZIP, an installer or a Docker image) distributes those components. You then have to meet each component's license yourself. That includes the GPL-3.0 text and the complete corresponding source for ComfyUI, VideoHelperSuite and the exact FFmpeg build, plus all license notices and every bundled model's terms. The official repository avoids all of this by downloading the components on the user's PC.

## Models downloaded automatically

Contsy downloads models from their providers' repositories at pinned revisions. It does not redistribute any weights. Each model's license applies to whoever downloads and uses it, independently of Contsy's GPL. [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) lists every model with its license, and [MODEL_LICENSES.md](../MODEL_LICENSES.md) lists the sources.

- **Apache-2.0 or MIT, so commercial use and hosting are allowed:** Z-Image Turbo, Wan 2.2 and its conversions and LoRAs, Qwen Image Edit 2511, Qwen2.5-VL, Qwen3.5 9B, SeedVR2, VoxCPM2, Whisper tiny, sd-vae-ft-mse, T5-base, OpenCV YuNet and SFace, CLIP ViT-B/32, and the MediaPipe Face Landmarker models.
- **LatentSync 1.6 (lip sync) uses CreativeML Open RAIL++-M.** You may use it commercially, but the license lists uses that are not allowed. If you let other people use the model, for example in a hosted service, you must require them to follow the same restrictions.
- **Stable Audio Open 1.0 (background sound) uses the Stability AI Community License.**
  - It is free for individuals and organizations with less than USD 1 million in annual revenue. Commercial users must register with Stability AI, and organizations above that threshold need an enterprise license.
  - It requires the notice that is in [NOTICE](../NOTICE), a copy of the license, and "Powered by Stability AI". Contsy shows these in **Settings → About**.
  - Stability AI's Acceptable Use Policy applies.
- **The text encoder for reference photos now downloads from an Apache-2.0 repository.** It used to come from a repository whose card names the Tencent Hunyuan Community License, which excludes the EU, the UK and South Korea. The identical file now comes from `Comfy-Org/Qwen-Image_ComfyUI` (Apache-2.0).

**What you create.** The GPL covers Contsy's code, not the photos, videos and sound you create with it. Whether you may use an output commercially depends on the models that produced it.

## Commercial use

- **Contsy's own code:** allowed without a fee, including for client work, agencies and businesses.
- **Models:** see above. Stable Audio Open is the main conditional one. If its terms do not fit your use, avoid the features that use it.
- **Outputs:** not restricted by Contsy's license.

## Your obligations under the GPL

| What you do | What the GPL requires |
| --- | --- |
| Use Contsy unmodified, privately or commercially, on your own PC | Nothing beyond the third-party licenses |
| Change Contsy for your own use only | Nothing |
| Let people use Contsy, changed or not, over a network without giving them a copy | No duty to publish the source code. GPLv3 section 0 says that interacting with users over a network without transferring a copy is not conveying. The third-party licenses still apply. |
| Share copies of Contsy, changed or not | Pass on the GPL-3.0-only and keep the copyright, SPDX and license notices, NOTICE and the license texts. Provide the complete source code of what you share, as section 6 describes. Do not add further restrictions. |
| Share a changed version | All of the above. Also mark your changes with a date (section 5a), and keep the legal notices visible in the UI: **Settings → About** (section 5d). Setting `app.source_url` makes that card link to your version's source code. |

## Hosting Contsy as a service

- **The GPL allows hosting**, including commercially and for a fee, with or without changes.
- **Operators do not have to publish their changes.** The GPL's source-code duty applies only when copies are shared. Anyone may therefore run a changed Contsy as an online service and keep the changes private. That includes competitors.
- **Practical limits are not license limits.** Contsy is a single-user local app. It listens only on `127.0.0.1`, rejects foreign hosts and has no accounts or access control. A hosted version needs its own authentication, isolation and abuse protection (see [SECURITY.md](../SECURITY.md)). It must also respect the models' terms for hosted use, especially Stable Audio Open and LatentSync.

## How to get the source code

- **Repository:** Contsy's complete source code is the repository this file belongs to.
- **In the app:** **Settings → About** shows the license, the notices and where the source code is: the `app.source_url` setting if it is set, otherwise the installation folder. Contsy runs directly from its Python source files, so every installation contains its complete source code.

## Future commercialization (for example Contsy Cloud)

Nothing in this section is implemented. Contsy has no paid or closed features.

1. **A hosted service is possible under the GPL alone.** The owner, like anyone else, may run Contsy with private changes on servers, for example as Contsy Cloud, without publishing those changes. This holds as long as no copies of the changed software are given to users. Code from contributors under the GPL may be used this way too.
2. **The owner's own code can be dual-licensed.** As the copyright holder, the owner may also license that code under other terms. Examples are a closed desktop edition, or a commercial license for companies that want to ship Contsy inside closed products. Versions already published under the GPL stay available under the GPL; that cannot be withdrawn.
3. **External contributions limit dual licensing.** Contributors keep the copyright in their code. Without extra permission, the owner may share their code only under the GPL, like everyone else. [CONTRIBUTING.md](../CONTRIBUTING.md) therefore asks each contributor for an additional relicensing license. To rely on it, record each contributor's explicit agreement before merging. A CLA check bot (for example CLA Assistant) with a CLA text reviewed by a lawyer does this. As of 1 November 2026 the history contains only the owner's commits, so no external contributions need clearing yet.
4. **CLA or DCO?**

   | | CLA (contributor license agreement) | DCO (Developer Certificate of Origin) |
   | --- | --- | --- |
   | What the contributor does | Signs once, usually through a bot | Adds a `Signed-off-by` line to each commit |
   | Grants extra rights to the owner | Yes, for example the right to relicense | No, it only confirms the right to submit under the project license |
   | Allows a closed or dual-licensed edition that is distributed | Yes | No, unless every contributor agrees separately |
   | Needed for a hosted service | No | No |
   | Effect on contributors | More friction; some developers avoid CLA projects | Light |

   If a closed edition that is distributed or a commercial license is planned, use a CLA, as `CONTRIBUTING.md` already describes. A DCO is enough if every distributed version will stay under the GPL.
5. **Limits of a future commercial edition:**
   - Other people's GPL contributions cannot be distributed under closed terms without their permission. Such code would have to be removed or rewritten first.
   - Third-party licenses stay in force. Running ComfyUI on your own servers does not require publishing anything. Distributing anything that contains ComfyUI, for example a desktop client, triggers the GPL for that distribution.
   - Model licenses decide whether hosted use is allowed: see Stable Audio Open and LatentSync above. Check every new model before adding it.
   - The GPL grants no trademark rights, and GPLv3 section 7(e) allows declining them explicitly. Because anyone may host Contsy, the name "Contsy Studio" is the main way to set an official service apart. Registering it as a trademark and publishing a trademark policy protects it.
   - A "no SaaS" restriction cannot be added on top of the GPL. Sections 7 and 10 forbid further restrictions, and recipients may remove them.
6. **GPL compared with AGPL.** The AGPL-3.0 would oblige anyone who offers a changed Contsy over a network to publish their changes. The GPL-3.0 does not. The owner chose the GPL-3.0 on 1 November 2026, the same license as ComfyUI. The trade-off is that competitors may host changed versions without sharing their changes, while a future Contsy Cloud has the same freedom.
7. **Parts that can be separated cleanly.** A future commercial layer can be new code written only by the owner, kept in a separate repository. It would talk to the GPL app through documented interfaces. Examples are multi-user accounts, billing, a GPU cluster scheduler, cloud storage and team features. Track which files contain external contributions, using the git history and the CLA records, so it stays clear what the owner may relicense.
8. **AI-assisted code and AI-generated voices.** Much of Contsy was written with AI assistance, and the voice samples are AI-generated. Whether and how far such material is protected by copyright differs between jurisdictions. That affects how strong the owner's rights are, including for dual licensing. Have a lawyer assess it.

## Resolved

- **Voice samples.** The four files in `assets/voices` were generated during development with VoxCPM2 (Apache-2.0) from text alone, with no reference recording and no voice description. 24 random voices were generated, and four were chosen by pitch and a speech-quality score (UTMOS). They are synthetic, not recordings of real people. Evidence:
  - the generation script and the copy step are recorded in the development log;
  - each file's pitch matches the selection notes in `app/tts.py`;
  - each file has VoxCPM2's output format (48 kHz mono).
- **kaldiio.** Its license is NTT's "Software License Agreement for Evaluation" (internal evaluation only), not an open-source license. pip used to install it because VoxCPM 2.0.3 declares funasr, which requires kaldiio. Neither Contsy nor VoxCPM ever imports either package.
  - The installer now installs VoxCPM without funasr, and older installations lose funasr and kaldiio on their next `install.bat` run. That removes 18 packages that only funasr needed.
  - The dependency check accepts only the missing funasr requirement.
  - Tested on 1 November 2026: installation, the dependency check, all unit tests and speech generation with all four voices, in an environment without any of the 18 packages.

## Open items

These points are unverified or unresolved. Settle them before or soon after publication:

| # | Item | Status | Suggested action |
| --- | --- | --- | --- |
| 1 | ComfyUI-LatentSyncWrapper @ `360d528` | The README says Apache-2.0, but there is no LICENSE file. The code Contsy imports from it is ByteDance's Apache-2.0 LatentSync code with small changes. | Ask the author to add the LICENSE file, or switch to the upstream `bytedance/LatentSync` at a pinned commit, followed by a lip-sync test. |
| 2 | musubi-tuner @ `f8a1b03` | Apache-2.0 according to its README; no license file. Some subfolders follow the licenses of the models they adapt. | Fine to download on install; check before redistributing. |
| 3 | RIFE 4.9 weights | ComfyUI-Frame-Interpolation downloads them from GitHub release mirrors that state no license for the file. The RIFE code is MIT. | Confirm the terms with the RIFE authors. |
| 4 | Stable Audio Open notices | The notice and "Powered by Stability AI" are in NOTICE and **Settings → About**. | Have a lawyer confirm this is enough for an app that downloads the model, and check the registration duty for commercial users. |
| 5 | LatentSync use restrictions | Contsy's SFW and consent rules fit them, but hosted users would also have to accept them. | Show the restrictions to users of any hosted version. |
| 6 | Trademark "Contsy Studio" | Not checked. | Search the EUIPO, USPTO and WIPO registers before launch. |
| 7 | Copyright holder name | "Karol Kosciolek" without Polish diacritics. | Confirm the legal name or company, then change it everywhere at once. |
| 8 | AI-assisted code and AI-generated voices | Copyright status depends on the jurisdiction. | Lawyer. |
| 9 | CLA tooling | Not set up. | Set it up before merging the first external pull request. |
| 10 | REUSE compliance | Checked with a script: every file has an SPDX header or a `REUSE.toml` entry. The official `reuse lint` tool was not run. | Run `reuse lint` in CI after publication. |
