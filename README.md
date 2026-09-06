# ComfyUI Image Music Karaoke

`comfyui-image-music-karaoke` is a local ComfyUI custom node and workflow for directing a complete image-to-song production. It plans a structured song, generates sectioned music with ComfyUI's native ACE-Step 1.5 nodes, produces a coordinated still-image scene sequence, optionally converts only the separated lead vocal to an RVC voice, mixes uploaded or described sound effects, aligns lyrics, and exports:

- `song.flac` — lossless full mix
- `song.mp3` — MP3 full mix
- `music_video.mp4` — scene sequence with the full mix
- `karaoke_video.mp4` — scene sequence with synchronized lyrics and the karaoke mix

The full mix contains the instrumental, lead vocal, feasible backing vocals, and scheduled SFX. The karaoke mix removes the separated lead vocal while retaining the instrumental, SFX, and feasible backing vocals.

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

1. [ComfyUI_Demucs](https://github.com/smthemex/ComfyUI_Demucs) for instrumental/vocal separation. This is required for both the full and karaoke mixes.
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
| `flux1-schnell-fp8.safetensors` | `diffusion_models/` | Scene generation and reference-conditioned editing |
| `clip_l.safetensors` | `text_encoders/` | Flux CLIP-L encoder |
| `t5xxl_fp16.safetensors` | `text_encoders/` | Flux T5-XXL encoder |
| `ae.safetensors` | `vae/` | Flux autoencoder |

The FP16 T5 encoder and the configured Flux model require substantial memory. Changing to a lower-memory encoder/model requires selecting a compatible filename in the render node and may change output quality.

### Demucs vocal separation

The pipeline selects `htdemucs` through ComfyUI_Demucs. That project downloads and caches its pretrained Demucs files when needed; it does not belong in this repository. Follow ComfyUI_Demucs if you need to pre-populate its cache for an offline machine.

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

The public workflow deliberately contains the placeholder `local:YOUR_TRAINED_VOICE.pth` and leaves the voice node disconnected. After copying your own model outside the repository, restart ComfyUI, select it in **Optional trained singing voice**, and connect `rvc_model` to the render node's `trained_voice_model` input.

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

- Click **+ Add Image** for every source asset.
- Choose an image and describe who or what it contains and how the Director may use it.
- Give identity instructions explicitly, such as “Main performer; preserve face, glasses, hair, and clothing across chorus scenes.”
- Use **Remove** to delete an entry. No image is required.

References are uploaded to ComfyUI's input directory and passed as actual pixel-conditioned image-to-image inputs. The Director may assign different references to different song scenes or combine several references in one scene. With no references, it creates scene prompts from the song and lyrics.

## Sound effects

- Click **+ Add Sound Effect** for each effect.
- Upload WAV, MP3, FLAC, M4A, AAC, OGG, or Opus audio, or leave the file empty and describe a sound to generate.
- Add optional placement/use instructions such as “quietly after the first chorus” or “twice during the outro.”
- Expand **Advanced timing and level** for occurrence, generated-duration, and gain controls.
- Leave placement blank or set it to `automatic` to let the Director schedule the effect.

SFX are normalized and mixed into both the full and karaoke arrangements before final mastering. Multiple effects and multiple occurrences are supported. A workflow with no SFX entries is valid.

## Trained singing voice

ACE-Step first generates the complete musical performance. Demucs separates the instrumental and vocal stem. The vocal stem is divided into lead and backing components; only the lead is sent through RVC. The converted lead is level-matched and recombined with the instrumental and backing vocal estimate. The instrumental is never sent through voice conversion.

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
- Reference preservation uses low-denoise image-to-image conditioning plus a pixel-space blend. It strongly favors the supplied subject but is not a biometric identity guarantee.
- Multiple references assigned to one scene are composed into a conditioning canvas. Crowded or conflicting references reduce fidelity.
- Visual scenes are still images synchronized to the structure; this project does not synthesize character animation or lip movement.
- Demucs and the lead/backing heuristic can leave vocal residue in karaoke audio or remove some backing harmonies.
- RVC quality depends on the training set, pitch range, source vocal, and separation quality. Extreme singing can produce artifacts.
- Whisper alignment can fall back to estimated timing when no local model is available or recognition differs too much from the supplied lyrics.
- Described SFX use a limited procedural fallback unless a compatible neural SFX engine is connected.
- FFmpeg must include `libass` for burned-in karaoke subtitles and `libx264` for MP4 output.
- Ollama must remain available throughout planning. The current planner expects its local HTTP API on the default port.
- The repository does not include third-party models or custom nodes; their licenses and installation requirements apply separately.

## Privacy and repository hygiene

The `.gitignore` blocks common model, voice, image, audio, video, output, cache, and credential formats. Before every public commit, still review staged files manually:

```bash
git status --short
git diff --cached --stat
git diff --cached
```

Do not use `git add -f` to bypass these exclusions for personal assets.
