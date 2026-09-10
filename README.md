# ComfyUI Image Music Karaoke

`comfyui-image-music-karaoke` is a local ComfyUI custom node and workflow for directing a complete image-to-song production. It plans a structured song, generates sectioned music with ComfyUI's native ACE-Step 1.5 nodes, produces a coordinated still-image scene sequence, optionally converts either a separated lead or a clean ACE-Step Base LEGO lead to an RVC voice, mixes uploaded or described sound effects, aligns lyrics, and exports:

- `song.flac` — lossless full mix
- `song.mp3` — MP3 full mix
- `music_video.mp4` — scene sequence with the full mix
- `karaoke_video.mp4` — scene sequence with synchronized lyrics and the karaoke mix

The full mix contains the instrumental, lead vocal, backing vocals, and scheduled SFX. The karaoke mix uses the clean separated instrumental plus scheduled SFX. Vocal-stem material is not mixed back into the karaoke track.

Everything runs locally. The repository contains code and a sanitized workflow only. It contains no model weights, voice models, photographs, recordings, generated media, or credentials.

## Features

- Director/planner driven by a local multimodal Ollama model
- Arbitrary song duration divided into configurable ACE-Step sections
- Optional exact lyrics, including explicit `[SFX]` markers
- Zero, one, or many reference images through a variable-length custom UI
- A separate editable identity/use instruction for every reference image
- Zero, one, or many uploaded or described sound effects
- Automatic or user-directed SFX placement and repeat instructions
- Reference-conditioned Flux image-to-image scene generation
- Automatic visuals when no image is supplied
- Optional lead-only RVC conversion after Demucs source separation
- Separate full-song and karaoke mixes
- Final-performance lyric alignment with a Whisper backend when available
- FLAC, MP3, music-video MP4, and karaoke-video MP4 outputs

## Repository layout

```text
comfyui-image-music-karaoke/
├── __init__.py
├── nodes.py
├── planner.py
├── audio_pipeline.py
├── visual_pipeline.py
├── exporter.py
├── requirements.txt
├── web/
│   └── karaoke.js
└── workflows/
    └── image-music-karaoke.json
```

## Requirements

Use a current ComfyUI build containing the native ACE-Step 1.5 node types, including `TextEncodeAceStepAudio1.5`, `EmptyAceStep1.5LatentAudio`, and `VAEDecodeAudioTiled`.

The workflow also uses these custom-node projects:

1. [ComfyUI_Demucs](https://github.com/smthemex/ComfyUI_Demucs) for the legacy instrumental/vocal separation path. It remains required when using **Legacy / separated vocal**.
2. [TTS Audio Suite](https://github.com/diodiogod/TTS-Audio-Suite) for the optional trained RVC voice loader/converter and optional neural sound-effect engine. The included workflow contains its `LoadRVCModelNode` as a disconnected placeholder.

The `PreviewAny` and `MarkdownNote` utility nodes shown in the included workflow must also be available. If your ComfyUI installation does not have them, remove those display-only nodes; they do not participate in generation.

System requirements:

- A CUDA-capable GPU is strongly recommended. The configured ACE-Step XL and Flux models are large.
- Enough disk space for all model files and generated intermediates.
- [Ollama](https://ollama.com/) running locally on its default `127.0.0.1:11434` endpoint.
- FFmpeg with `libass` and `libx264`. Installing `imageio-ffmpeg` supplies a usable FFmpeg binary in many environments; set `IMAGEIO_FFMPEG_EXE` if you need to select another build.

## Installation

### ComfyUI

Clone this repository into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone <YOUR-GITHUB-REPOSITORY-URL> comfyui-image-music-karaoke
```

Install this project's direct Python dependencies with the same Python environment that runs ComfyUI:

```bash
python -m pip install -r comfyui-image-music-karaoke/requirements.txt
```

Install the two supporting custom-node projects and their dependencies according to their own READMEs:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/smthemex/ComfyUI_Demucs.git
git clone https://github.com/diodiogod/TTS-Audio-Suite.git tts_audio_suite
```

For `ComfyUI_Demucs`, install `requirements_minimal.txt`. TTS Audio Suite has its own installer because its dependency set varies by Python and platform; follow that project's current installation instructions rather than copying its dependency list into this environment manually.

Restart ComfyUI after installing custom nodes or adding models.

### Comfy Desktop

1. Stop the running ComfyUI instance from Comfy Desktop.
2. Open the instance's ComfyUI folder and place this repository at `custom_nodes/comfyui-image-music-karaoke`.
3. Install `requirements.txt` with the instance's Python environment. A typical Windows Desktop environment uses `ComfyUI/.venv/Scripts/python.exe`, but the exact path can vary by Desktop release.
4. Install ComfyUI_Demucs and TTS Audio Suite under the same `custom_nodes` directory.
5. Start the instance again and check the terminal for custom-node import errors.

## Local Director model

Install Ollama, then download the multimodal model selected by the workflow:

```bash
ollama pull gemma4:12b
```

The Director sends song text and supplied reference images to this local model. `gemma4:12b` is stored and managed by Ollama, not in a ComfyUI model folder. The workflow will fail during planning if Ollama is stopped or the selected model is unavailable.

## Required ComfyUI models

The filenames below match the included workflow. Download model files from their original or ComfyUI-packaged repositories and accept any applicable model licenses. Do not place model files inside this Git repository.

### ACE-Step 1.5 music

Download the ComfyUI-packaged files from [Comfy-Org/ace_step_1.5_ComfyUI_files](https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files).

| File | Place under `ComfyUI/models` | Purpose |
| --- | --- | --- |
| `acestep_v1.5_xl_sft_bf16.safetensors` | `diffusion_models/` | ACE-Step XL SFT music diffusion model |
| `qwen_0.6b_ace15.safetensors` | `text_encoders/` | ACE text/metadata encoder |
| `qwen_4b_ace15.safetensors` | `text_encoders/` | ACE audio-code language model |
| `ace_1.5_vae.safetensors` | `vae/` | ACE audio VAE |

These exact files are required by the workflow defaults. You may select another compatible ACE-Step 1.5 model in the render node, but its sampling needs can differ from the provided XL SFT settings.

### Flux scene images

The default image model is [Comfy-Org's Flux.1 Schnell FP8](https://huggingface.co/Comfy-Org/flux1-schnell). The text encoders are available from [comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders). ComfyUI's [Flux workflow guide](https://docs.comfy.org/tutorials/flux/flux-1-text-to-image) links the encoder and VAE downloads.

| File | Place under `ComfyUI/models` | Purpose |
| --- | --- | --- |
| `flux1-schnell-fp8.safetensors` | `diffusion_models/` | Text-to-image scenes and generated-background fallback |
| `flux1-fill-dev.safetensors` | `diffusion_models/` | Required masked background/scene editing around protected reference identities |
| `clip_l.safetensors` | `text_encoders/` | Flux CLIP-L encoder |
| `t5xxl_fp16.safetensors` | `text_encoders/` | Flux T5-XXL encoder |
| `ae.safetensors` | `vae/` | Flux autoencoder |

The FP16 T5 encoder and Flux models require substantial memory. The render node's selected image model handles text-to-image scenes. When a scene has assigned references, the visual pipeline automatically uses `flux1-fill-dev.safetensors` for masked editing when that file is installed. If it is absent or cannot run, the pipeline renders a fresh background with the selected image model and composites only the protected identity pixels; it records that fallback in the export report instead of returning the unchanged source frame.

### Demucs vocal separation

The karaoke path prefers the fine-tuned `htdemucs_ft` model through ComfyUI_Demucs, with 50% overlap and two shift passes. ComfyUI_Demucs downloads and caches its four official checkpoints when needed; they do not belong in this repository. If that model cannot load or run, the pipeline retries locally with `htdemucs` and records the fallback and reason in `generation.json` and the export report.

Karaoke uses only the model's instrumental output. It does not add the stereo-side estimate from the vocal stem back into the mix. This removes lead vocals much more strongly. Backing vocals that the model groups with the lead are also removed; backing vocals embedded in an instrumental stem can remain because they are not independently separable.

### Lyric alignment

`faster-whisper` is listed in `requirements.txt`. The exporter looks for a local `base.en`, `small.en`, or `medium.en` faster-whisper snapshot in the Hugging Face cache. It can also use an OpenAI Whisper `.pt` model if the `openai-whisper` package is installed; the conventional explicit path is:

```text
ComfyUI/models/stt/whisper/base.en.pt
```

With `lyric_timing=auto`, unavailable or low-confidence recognition falls back to estimated timing and records that fact in the export report. With `lyric_timing=required`, a missing or inadequate alignment model is an error.

### Optional trained RVC voice

Never commit a personal voice model. TTS Audio Suite searches these preferred locations:

```text
ComfyUI/models/TTS/RVC/YOUR_TRAINED_VOICE.pth
ComfyUI/models/TTS/RVC/.index/YOUR_TRAINED_VOICE.index
```

It also supports the legacy `ComfyUI/models/RVC` path and custom paths configured through `extra_model_paths.yaml`. The `.index` file is optional. The pipeline selects the `content-vec-best` HuBERT model; TTS Audio Suite can download it into `ComfyUI/models/TTS/hubert/` when first used.

The public workflow deliberately contains the placeholder `local:YOUR_TRAINED_VOICE.pth`. After copying your own model outside the repository, restart ComfyUI and select it in **3. Singing Voice (optional)**. Its `rvc_model` output is connected to the render node's `trained_voice_model` input in the bundled workflow.

### Optional neural SFX model

Uploaded sound effects need no generation model. Described effects work without another model by using the built-in procedural fallback. For higher-quality described effects, connect a compatible TTS Audio Suite `TTS_ENGINE` such as its MOSS SoundEffect v2 engine to the render node's `sfx_engine` input. TTS Audio Suite normally manages that model under `ComfyUI/models/TTS/moss_soundeffect_v2/`; follow its README because model names and installers can change.

## Opening the workflow

After installation and restart:

1. In ComfyUI, choose **Workflows → Open** or press `Ctrl+O`.
2. Open `custom_nodes/comfyui-image-music-karaoke/workflows/image-music-karaoke.json`.
3. Confirm that **1. Director — song plus variable images and SFX** and **3. Generate full song + music video + karaoke** are present without red missing-node errors.
4. Select the model filenames listed above in the render node if ComfyUI did not retain them automatically.

The included workflow is sanitized. It starts with a generic prompt, no images, no SFX, and no connected personal voice model.

## Using the Director

1. Enter a complete creative brief in `song_request`: subject, genre, mood, instruments, vocalist, language, tempo, visual tone, and any story beats.
2. Set `duration` in seconds, `MM:SS`, or `HH:MM:SS`.
3. Set `section_seconds` between 10 and 180. The Director divides the requested duration into sections and the renderer generates each section with ACE-Step.
4. Optionally enter exact lyrics in `lyrics_override`. Separate explicit section blocks with a line containing `---`. Standalone `[SFX]` markers are arrangement cues and are removed before lyrics reach ACE-Step.
5. Keep `writer_model` on `gemma4:12b` unless you have installed another compatible local Ollama model and updated the node definition.

The Director creates the music style, tempo, key, section lyrics, visual scenes, reference assignments, continuity notes, and SFX schedule. It writes intermediate plans under ComfyUI's output directory for troubleshooting.

## Reference images

- In the left-side **Reference Images** node, click **+ Add Image** for every source asset.
- Choose an image and describe who or what it contains and how the Director may use it.
- Give identity instructions explicitly, such as “Main performer; preserve face, glasses, hair, and clothing across chorus scenes.”
- Use **Remove** to delete an entry. No image is required.

References are uploaded to ComfyUI's input directory and passed as actual pixel-conditioned inputs. The Director may assign different references to different song scenes or combine several references in one scene. Assigned portraits use a protected face/hair/glasses region and a masked FLUX Fill edit for the surrounding scene. Clothing remains protected unless the scene prompt explicitly requests a wardrobe change. With no references, the pipeline uses ordinary text-to-image generation.

## Sound effects

- In the left-side **Sound Effects** node, click **+ Add Sound Effect** for each effect.
- Upload WAV, MP3, FLAC, M4A, AAC, OGG, or Opus audio, or leave the file empty and describe a sound to generate.
- Add optional placement/use instructions such as “quietly after the first chorus” or “twice during the outro.”
- Expand **Advanced timing and level** for occurrence, generated-duration, and gain controls.
- Leave placement blank or set it to `automatic` to let the Director schedule the effect.

SFX are normalized and mixed into both the full and karaoke arrangements before final mastering. Multiple effects and multiple occurrences are supported. A workflow with no SFX entries is valid.

## Trained singing voice

The renderer's `vocal_mode` selector provides two production paths:

- **Legacy / separated vocal** is the backward-compatible default. ACE-Step XL SFT generates the complete performance. The established `htdemucs` path separates the vocal stem, sends only the lead through RVC when a trained model is connected, and uses `htdemucs_ft` for the karaoke instrumental exactly as before.
- **ACE LEGO → RVC** asks XL SFT for the instrumental arrangement, unloads the ComfyUI models, generates an isolated `track_name="vocals"` lead with the local official ACE-Step 1.5 2B Base service, converts that clean lead through the selected Singing Voice RVC model, and remixes it directly with the instrumental. Demucs is not loaded or invoked in this mode; the generated instrumental also supplies karaoke audio.

The render node's optional `ace_voice_reference` AUDIO input controls ACE-Step timbre conditioning only. It remains separate from, and does not replace, the trained `.pth`/`.index` RVC model selected in **Singing Voice**. ACE LEGO mode requires that trained RVC model. When this mode is selected, the node checks the official Base API at `http://127.0.0.1:8001`. If needed, it starts the project's existing isolated ACE-Step runtime, waits for `/health`, and reuses that process for later sections. It never starts the service in Legacy mode and never downloads missing models. Set `IMAGE_MUSIC_KARAOKE_ACESTEP_BASE_ROOT` to the official ACE-Step repository if automatic discovery cannot find it.

To replace the voice later, select another `.pth`/`.index` pair in the loader. The song pipeline itself does not need to change.

## Running and outputs

Queue the workflow normally. The planner runs first, then the renderer generates scene images and ACE-Step song sections, processes vocals and SFX, masters the two mixes, aligns lyrics against the final full vocal performance, and exports the four deliverables.

Outputs are written beneath:

```text
ComfyUI/output/image_music_karaoke/<planner-job>/<render-job>/
```

The frontend extension adds output cards for the FLAC, MP3, music video, and karaoke video. The export report contains the resolved paths and lyric-timing mode.

## Known limitations

- The default ACE-Step XL, Flux, and Gemma models require substantial disk space, RAM, and VRAM. CPU-only operation is impractically slow for typical songs.
- Sections are generated independently. The planner keeps style and refrain consistent, but long songs can have audible changes at section boundaries.
- Reference preservation keeps identity-critical source pixels and uses masked FLUX Fill generation for editable regions. It strongly preserves the supplied face, glasses, and hair, but segmentation boundaries and extreme pose changes can still produce visible seams; it is not a biometric identity guarantee.
- Multiple references assigned to one scene are arranged into a shared conditioning canvas before masked composition. Crowded, differently lit, or conflicting references reduce composition quality.
- Visual scenes are still images synchronized to the structure; this project does not synthesize character animation or lip movement.
- No separator can guarantee perfect isolation. `htdemucs_ft` can leave quiet vocal residue or remove backing harmonies that overlap the lead; the fallback `htdemucs` model is faster but usually less clean.
- RVC quality depends on the training set, pitch range, source vocal, and separation quality. Extreme singing can produce artifacts.
- Whisper alignment can fall back to estimated timing when no local model is available or recognition differs too much from the supplied lyrics.
- Described SFX use a limited procedural fallback unless a compatible neural SFX engine is connected.
- FFmpeg must include `libass` for burned-in karaoke subtitles and `libx264` for MP4 output.
- Ollama must remain available throughout planning. The current planner expects its local HTTP API on the default port.
- The repository does not include third-party models or custom nodes; their licenses and installation requirements apply separately.

## Experimental ACE-Step Base LEGO vocal milestone

The optional **ACE-Step Base — LEGO Lead Vocal (Milestone)** node accepts an existing
instrumental as ComfyUI `AUDIO` and saves a separate `lead_vocal.wav`. It is intentionally
kept as a standalone inspection node. The production renderer now reuses the same Base
client, clean-vocal RVC converter, and direct-remix implementation when its `vocal_mode`
is **ACE LEGO → RVC**.

The optional `voice_reference` input uses ACE-Step's official `reference_audio` timbre
conditioning while the instrumental remains the separate LEGO `src_audio`. A normal
ComfyUI `AUDIO` value supplies one recording; a batched/list AUDIO value may supply several.
The node trims outer silence, selects an equal active contribution from every recording,
and writes one 30-second composite because the official HTTP API accepts one reference
file. ACE-Step then VAE-encodes that file and applies its global timbre encoder. With the
input disconnected, the request retains the original milestone fields and behavior.

Connect the existing TTS Audio Suite `RVC_MODEL` output to `trained_voice_model` to send
the already-isolated LEGO lead directly through the same production RVC configuration:
pitch `0`, index ratio `0.75`, consonant protection `0.25`, volume envelope `0.25`,
`content-vec-best`, one refinement pass, and smart 30-second chunks. This experimental
route does not run Demucs before RVC. Leaving the model disconnected preserves the earlier
LEGO outputs and skips the two RVC-specific files.

This node calls the official ACE-Step 1.5 local API at `http://127.0.0.1:8001`. The production
ACE LEGO mode automatically starts and reuses the existing isolated runtime when necessary.
Configure that runtime with the local 2B `acestep-v15-base` checkpoint. The node checks `/v1/model_inventory`
before `/v1/init` and refuses to proceed unless the Base model is already reported, so it
cannot start an implicit model download. Start the official service in offline mode when
you need a hard server-side guarantee as well:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:ACESTEP_CONFIG_PATH = "acestep-v15-base"
$env:ACESTEP_OFFLOAD_TO_CPU = "true"
$env:ACESTEP_NO_INIT = "false"
python -m acestep.api_server
```

The official current inference path treats LEGO as direct source-audio conditioning and
bypasses its language model. The integration records `acestep-5Hz-lm-1.7B` as the preferred
future LM but does not spend VRAM loading an unused Qwen model for this milestone.

For an isolated test, connect a ComfyUI audio loader to `instrumental`, enter the desired
vocal performance and lyrics, and queue only this node. Before the request it unloads
ComfyUI models and empties the CUDA cache. Each run creates:

```text
ComfyUI/output/image_music_karaoke/<job>/base_lego_vocals/
├── instrumental.wav
├── voice_reference.wav       # only when reference conditioning is used
├── voice_reference_001.wav   # one selected contribution per supplied recording
├── lead_vocal.wav
├── lead_vocal_rvc.wav        # only when a trained RVC model is connected
├── final_remix.wav           # instrumental + converted LEGO vocal
└── lego_vocals_report.json
```

In a controlled 10-second comparison using the same LEGO vocal in both branches, direct
clean-stem RVC produced 1.19 dB stronger harmonic-to-residual ratio, 22% lower spectral
flatness, and lower lyric word error than converting a Demucs-separated copy. This supports
using the LEGO stem directly, although RVC can still introduce diction and timbre artifacts.

The report records the model, task, source and output paths, generated duration, and
fallback status, plus the conditioning mechanism and reference count. There is deliberately
no alternate generator or separator fallback. Reference audio guides global acoustic
features and timbre; it is not a guaranteed speaker-identity clone. It can also influence
phrasing, register, and timing because ACE-Step exposes no independent timbre-strength
control for this path.

## Privacy and repository hygiene

The `.gitignore` blocks common model, voice, image, audio, video, output, cache, and credential formats. Before every public commit, still review staged files manually:

```bash
git status --short
git diff --cached --stat
git diff --cached
```

Do not use `git add -f` to bypass these exclusions for personal assets.
