import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EMPTY_MEDIA = { version: 1, references: [], sound_effects: [] };

function parseMediaState(value) {
  try {
    const parsed = typeof value === "string" ? JSON.parse(value) : value;
    return {
      version: 1,
      references: Array.isArray(parsed?.references) ? parsed.references.map((item) => ({ ...item })) : [],
      sound_effects: Array.isArray(parsed?.sound_effects)
        ? parsed.sound_effects.map((item) => ({ ...item }))
        : [],
    };
  } catch {
    return structuredClone(EMPTY_MEDIA);
  }
}

function assetURL(asset) {
  if (!asset?.filename) return "";
  return api.apiURL("/view?" + new URLSearchParams({
    filename: asset.filename,
    subfolder: asset.subfolder || "",
    type: asset.type || "input",
  }).toString());
}

async function uploadAsset(file, subfolder) {
  const body = new FormData();
  body.append("image", file);
  body.append("type", "input");
  body.append("subfolder", subfolder);
  const response = await api.fetchApi("/upload/image", { method: "POST", body });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  const data = await response.json();
  return {
    filename: data.name,
    subfolder: data.subfolder || "",
    type: data.type || "input",
    original_name: file.name,
  };
}

function chooseFile(accept, onChoose) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = accept;
  input.style.display = "none";
  input.addEventListener("change", async () => {
    try {
      if (input.files?.[0]) await onChoose(input.files[0]);
    } finally {
      input.remove();
    }
  }, { once: true });
  document.body.append(input);
  input.click();
}

function element(tag, options = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(options)) {
    if (key === "style") node.style.cssText = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node[key] = value;
  }
  node.append(...children);
  return node;
}

function field(label, control) {
  return element("label", { style: "box-sizing:border-box;display:grid;gap:4px;width:100%;min-width:0;max-width:100%;font-size:12px;color:#cbd5e1;" }, [
    element("span", { text: label }), control,
  ]);
}

function isolateEditorControl(control) {
  for (const eventName of ["pointerdown", "mousedown", "click", "dblclick"]) {
    control.addEventListener(eventName, (event) => event.stopPropagation());
  }
  return control;
}

function textArea(value, placeholder, onInput, rows = 3) {
  return isolateEditorControl(element("textarea", {
    value: value || "", placeholder, rows,
    style: "box-sizing:border-box;display:block;width:100%;min-width:0;max-width:100%;resize:vertical;overflow-x:hidden;overflow-wrap:anywhere;white-space:pre-wrap;border:1px solid #4b5563;border-radius:6px;background:#111827;color:#f8fafc;padding:7px;font:12px/1.35 sans-serif;",
    oninput: (event) => { event.stopPropagation(); onInput(event.target.value); },
  }));
}

function textInput(value, placeholder, onInput, type = "text") {
  return isolateEditorControl(element("input", {
    value: value ?? "", placeholder, type,
    style: "box-sizing:border-box;display:block;width:100%;min-width:0;max-width:100%;border:1px solid #4b5563;border-radius:6px;background:#111827;color:#f8fafc;padding:7px;font:12px sans-serif;",
    oninput: (event) => { event.stopPropagation(); onInput(event.target.value); },
  }));
}

function button(label, onClick, kind = "normal") {
  const colors = kind === "remove"
    ? "background:#4c1d1d;border-color:#7f1d1d;color:#fecaca;"
    : kind === "add"
      ? "background:#164e63;border-color:#0e7490;color:#cffafe;font-weight:700;"
      : "background:#273449;border-color:#475569;color:#e2e8f0;";
  return element("button", {
    type: "button", text: label,
    style: `box-sizing:border-box;flex:0 0 auto;max-width:100%;white-space:nowrap;cursor:pointer;border:1px solid;border-radius:6px;padding:7px 10px;font:12px sans-serif;${colors}`,
    onclick: (event) => { event.preventDefault(); event.stopPropagation(); onClick(event); },
  });
}

function hideStorageWidget(widget) {
  widget.type = "hidden";
  widget.hidden = true;
  widget.computeSize = () => [0, -4];
  for (const target of [widget.inputEl, widget.element]) {
    if (target?.style) target.style.display = "none";
  }
}

function installMediaEditor(node, kind) {
  const widgetName = kind === "references" ? "references_json" : "sound_effects_json";
  const rawWidget = node.widgets?.find((widget) => widget.name === widgetName);
  if (!rawWidget || node.karaokeMediaEditor) return;

  let state = parseMediaState(rawWidget.value);
  hideStorageWidget(rawWidget);
  const originalIndex = node.widgets.indexOf(rawWidget);
  node.widgets.splice(originalIndex, 1);

  const root = element("div", {
    style: "box-sizing:border-box;inline-size:100%;width:100%;min-inline-size:0;min-width:0;max-inline-size:100%;max-width:100%;height:650px;overflow-x:hidden;overflow-y:auto;contain:inline-size layout paint;display:flex;flex-direction:column;gap:12px;padding:10px;background:#0b1220;color:#f8fafc;border:1px solid #334155;border-radius:8px;",
  });
  root.addEventListener("pointerdown", (event) => event.stopPropagation());
  root.addEventListener("mousedown", (event) => event.stopPropagation());
  root.addEventListener("dblclick", (event) => event.stopPropagation());

  let editorWidget;
  const dirty = () => {
    editorWidget.callback?.(JSON.stringify(state));
    node.graph?.setDirtyCanvas(true, true);
  };

  const render = () => {
    root.replaceChildren();
    if (kind === "references") {
      root.append(element("div", { style: "box-sizing:border-box;display:flex;align-items:center;justify-content:space-between;gap:8px;width:100%;min-width:0;max-width:100%;" }, [
        element("strong", { text: `Reference Images (${state.references.length})`, style: "min-width:0;font:700 14px sans-serif;" }),
        button("+ Add Image", () => {
          state.references.push({ id: crypto.randomUUID(), filename: "", instruction: "" });
          render(); dirty();
        }, "add"),
      ]));
      if (!state.references.length) {
        root.append(element("div", { text: "No reference images. Visuals will be generated from the song plan.", style: "padding:8px;color:#94a3b8;font:12px sans-serif;" }));
      }
      state.references.forEach((item, index) => {
        const fileName = element("span", {
          text: item.original_name || item.filename || "No image selected",
          style: "min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#cbd5e1;font:12px sans-serif;",
        });
        const choose = button(item.filename ? "Replace Image" : "Choose Image", () => {
          choose.disabled = true;
          choose.textContent = "Uploading…";
          chooseFile("image/png,image/jpeg,image/webp,image/bmp,image/tiff", async (file) => {
            try {
              Object.assign(item, await uploadAsset(file, "image_music_karaoke/references"));
              render(); dirty();
            } catch (error) {
              choose.disabled = false;
              choose.textContent = item.filename ? "Replace Image" : "Choose Image";
              alert(`Image upload failed: ${error.message}`);
            }
          });
        });
        const card = element("div", { style: "box-sizing:border-box;display:grid;grid-template-columns:minmax(0,1fr);gap:8px;width:100%;min-width:0;max-width:100%;padding:10px;overflow:hidden;border:1px solid #334155;border-radius:8px;background:#111827;" }, [
          element("div", { style: "display:flex;align-items:center;justify-content:space-between;gap:8px;width:100%;min-width:0;max-width:100%;" }, [
            element("strong", { text: `Image ${index + 1}`, style: "min-width:0;font:700 13px sans-serif;" }),
            button("Remove", () => { state.references.splice(index, 1); render(); dirty(); }, "remove"),
          ]),
          element("div", { style: "display:grid;grid-template-columns:auto minmax(0,1fr);align-items:center;gap:8px;" }, [choose, fileName]),
        ]);
        if (item.filename) {
          card.append(element("img", { src: assetURL(item), alt: fileName.textContent,
            style: "box-sizing:border-box;display:block;width:100%;min-width:0;max-width:100%;height:auto;max-height:180px;object-fit:contain;border-radius:6px;background:#020617;" }));
        }
        card.append(field("Who or what is shown, and how should this image be used?",
          textArea(item.instruction, "Example: Main character. Preserve their identity and use them in chorus scenes.", (value) => { item.instruction = value; dirty(); }, 3)));
        root.append(card);
      });
    } else {
      root.append(element("div", { style: "box-sizing:border-box;display:flex;align-items:center;justify-content:space-between;gap:8px;width:100%;min-width:0;max-width:100%;" }, [
        element("strong", { text: `Sound Effects (${state.sound_effects.length})`, style: "min-width:0;font:700 14px sans-serif;" }),
        button("+ Add Sound Effect", () => {
          state.sound_effects.push({ id: crypto.randomUUID(), filename: "", description: "", placement: "", occurrences: "automatic", duration: 4, gain_db: -18 });
          render(); dirty();
        }, "add"),
      ]));
      if (!state.sound_effects.length) {
        root.append(element("div", { text: "No sound effects. The song can run with this list empty.", style: "padding:8px;color:#94a3b8;font:12px sans-serif;" }));
      }
      state.sound_effects.forEach((item, index) => {
        const fileName = element("span", {
          text: item.original_name || item.filename || "No audio selected — use the description below",
          style: "min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#cbd5e1;font:12px sans-serif;",
        });
        const choose = button(item.filename ? "Replace Audio" : "Upload Audio", () => {
          choose.disabled = true;
          choose.textContent = "Uploading…";
          chooseFile("audio/wav,audio/mpeg,audio/mp3,audio/flac,audio/mp4,audio/aac,audio/ogg,audio/opus", async (file) => {
            try {
              Object.assign(item, await uploadAsset(file, "image_music_karaoke/sound_effects"));
              render(); dirty();
            } catch (error) {
              choose.disabled = false;
              choose.textContent = item.filename ? "Replace Audio" : "Upload Audio";
              alert(`Audio upload failed: ${error.message}`);
            }
          });
        });
        const fileRow = element("div", { style: "display:grid;grid-template-columns:auto minmax(0,1fr) auto;align-items:center;gap:8px;" }, [choose, fileName]);
        if (item.filename) {
          fileRow.append(button("Clear Audio", () => {
            delete item.filename; delete item.subfolder; delete item.type; delete item.original_name;
            render(); dirty();
          }));
        }
        const card = element("div", { style: "box-sizing:border-box;display:grid;grid-template-columns:minmax(0,1fr);gap:8px;width:100%;min-width:0;max-width:100%;padding:10px;overflow:hidden;border:1px solid #334155;border-radius:8px;background:#111827;" }, [
          element("div", { style: "display:flex;align-items:center;justify-content:space-between;gap:8px;width:100%;min-width:0;max-width:100%;" }, [
            element("strong", { text: `Sound Effect ${index + 1}`, style: "min-width:0;font:700 13px sans-serif;" }),
            button("Remove", () => { state.sound_effects.splice(index, 1); render(); dirty(); }, "remove"),
          ]), fileRow,
        ]);
        if (item.filename) {
          card.append(element("audio", { src: assetURL(item), controls: true, preload: "metadata", style: "box-sizing:border-box;display:block;width:100%;min-width:0;max-width:100%;height:34px;" }));
        }
        card.append(
          field("Sound description (or a label for uploaded audio)",
            textArea(item.description, "Example: distant train horn", (value) => { item.description = value; dirty(); }, 2)),
          field("Optional placement / use instructions",
            textArea(item.placement, "Example: after the first chorus, quietly in the distance; leave blank for automatic placement", (value) => { item.placement = value; dirty(); }, 2)),
        );
        const advanced = element("details", {}, [element("summary", { text: "Advanced timing and level", style: "cursor:pointer;color:#94a3b8;font:12px sans-serif;" })]);
        advanced.append(element("div", { style: "display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:8px;" }, [
          field("Occurrences", textInput(item.occurrences || "automatic", "automatic", (value) => { item.occurrences = value; dirty(); })),
          field("Generated seconds", textInput(item.duration ?? 4, "4", (value) => { item.duration = Number(value) || 4; dirty(); }, "number")),
          field("Gain dB", textInput(item.gain_db ?? -18, "-18", (value) => { item.gain_db = Number(value); dirty(); }, "number")),
        ]));
        card.append(advanced);
        root.append(card);
      });
    }
  };

  editorWidget = node.addDOMWidget(widgetName, "karaoke_media_editor", root, {
    serialize: true,
    hideOnZoom: false,
    getValue() {
      return JSON.stringify(kind === "references"
        ? { version: 1, references: state.references }
        : { version: 1, sound_effects: state.sound_effects });
    },
    setValue(value) {
      state = parseMediaState(value);
      render();
    },
  });
  const stableWidth = Math.max(Number(node.size?.[0]) || 0, 520);
  editorWidget.computeSize = () => [stableWidth, 670];
  requestAnimationFrame(() => {
    const wrapper = root.parentElement;
    if (!wrapper) return;
    wrapper.style.boxSizing = "border-box";
    wrapper.style.width = "100%";
    wrapper.style.minWidth = "0";
    wrapper.style.maxWidth = "100%";
    wrapper.style.overflow = "hidden";
    wrapper.style.contain = "inline-size layout paint";
  });
  const addedIndex = node.widgets.indexOf(editorWidget);
  node.widgets.splice(addedIndex, 1);
  node.widgets.splice(originalIndex, 0, editorWidget);
  node.karaokeMediaEditor = {
    root,
    getState: () => state,
    load: (value) => {
      state = parseMediaState(value);
      render();
    },
    render,
  };
  render();
  node.setSize([stableWidth, Math.max(node.size[1], 720)]);
}

function hideLegacyPlannerMedia(node) {
  const widget = node.widgets?.find((item) => item.name === "media_inputs_json");
  if (!widget) return;
  hideStorageWidget(widget);
}

const VOCAL_MODES = ["Legacy / separated vocal", "ACE LEGO → RVC"];

function repairLegacyRendererWidgetOrder(node) {
  if (node.comfyClass !== "ImageSongRender") return false;
  const widgets = Object.fromEntries((node.widgets || []).map((widget) => [widget.name, widget]));
  const vocalMode = widgets.vocal_mode;
  if (!vocalMode) return false;

  // Workflows saved before Vocal Mode was inserted have every later positional
  // value shifted one widget to the right. Only migrate that recognizable shape.
  const hasLegacyShift = !VOCAL_MODES.includes(vocalMode.value)
    && Number.isFinite(Number(vocalMode.value))
    && typeof widgets.cfg?.value === "boolean"
    && typeof widgets.generate_audio_codes?.value === "string"
    && /^\d+x\d+$/i.test(String(widgets.lyric_timing?.value || ""));
  if (hasLegacyShift) {
    const shifted = {
      image_steps: vocalMode.value,
      image_edit_strength: widgets.image_steps?.value,
      identity_preservation: widgets.image_edit_strength?.value,
      steps: widgets.identity_preservation?.value,
      cfg: widgets.steps?.value,
      generate_audio_codes: widgets.cfg?.value,
      lyric_timing: widgets.generate_audio_codes?.value,
      resolution: widgets.lyric_timing?.value,
      filename: widgets.resolution?.value,
    };
    vocalMode.value = "Legacy / separated vocal";
    for (const [name, value] of Object.entries(shifted)) {
      if (widgets[name] && value !== undefined) widgets[name].value = value;
    }
  } else {
    // A user may select a valid Vocal Mode after loading the shifted workflow.
    // The remaining corruption still has a distinctive resolution/filename shape.
    const hasPartiallyEditedShift = VOCAL_MODES.includes(vocalMode.value)
      && Number(widgets.image_steps?.value) < 1
      && Number(widgets.identity_preservation?.value) > 1
      && /^\d+x\d+$/i.test(String(widgets.lyric_timing?.value || ""))
      && !/^\d+x\d+$/i.test(String(widgets.resolution?.value || ""));
    if (!hasPartiallyEditedShift) return false;
    const knownGood = {
      image_steps: 4,
      image_edit_strength: 0.3,
      identity_preservation: 0.58,
      steps: 65,
      cfg: 3,
      generate_audio_codes: true,
      lyric_timing: "auto",
      resolution: "1280x720",
      filename: "song",
    };
    for (const [name, value] of Object.entries(knownGood)) widgets[name].value = value;
  }
  node.setDirtyCanvas?.(true, true);
  node.graph?.setDirtyCanvas?.(true, true);
  return true;
}

app.registerExtension({
  name: "local.ImageMusicKaraoke",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "KaraokeReferenceImagesInput" || nodeData.name === "KaraokeSoundEffectsInput") {
      const originalCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const result = originalCreated?.apply(this, arguments);
        installMediaEditor(this, nodeData.name === "KaraokeReferenceImagesInput" ? "references" : "sound_effects");
        return result;
      };
      return;
    }
    if (nodeData.name === "ImageSongPlan") {
      const originalCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const result = originalCreated?.apply(this, arguments);
        hideLegacyPlannerMedia(this);
        return result;
      };
      return;
    }
    if (nodeData.name !== "ImageSongRender") return;
    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalCreated?.apply(this, arguments);
      const vocalMode = this.widgets?.find((widget) => widget.name === "vocal_mode");
      if (vocalMode) {
        vocalMode.label = "Vocal Mode";
        vocalMode.options = {
          ...(vocalMode.options || {}),
          values: VOCAL_MODES,
        };
      }
      return result;
    };
    const originalConfigured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = originalConfigured?.apply(this, arguments);
      setTimeout(() => repairLegacyRendererWidgetOrder(this), 0);
      return result;
    };
    const original = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      original?.apply(this, arguments);
      if (!this.songPreview) {
        const box = document.createElement("div");
        box.style.cssText = "display:flex;flex-direction:column;gap:8px;padding:8px;background:#15191e;color:#eee;overflow:auto;";
        this.songPreview = box;
        this.addDOMWidget("song_export_preview", "song_preview", box, { serialize: false, hideOnZoom: false });
      }
      const box = this.songPreview;
      box.replaceChildren();
      const makeURL = file => api.apiURL("/view?" + new URLSearchParams(file).toString());
      for (const [key, tag, label] of [
        ["music_video", "video", "Download music_video.mp4"],
        ["karaoke_video", "video", "Download karaoke_video.mp4"],
        ["song_flac", "audio", "Download song.flac"],
        ["song_mp3", "audio", "Download song.mp3"],
        ["song_video", "video", "Download karaoke MP4"],
        ["song_audio", "audio", "Download FLAC"],
      ]) {
        for (const file of message[key] || []) {
          const media = document.createElement(tag);
          media.controls = true;
          media.preload = "metadata";
          media.style.cssText = "width:100%;max-height:280px;";
          media.src = makeURL(file);
          box.append(media);
          const link = document.createElement("a");
          link.href = media.src;
          link.download = file.filename;
          link.textContent = label;
          link.style.color = "#8ad4ff";
          box.append(link);
        }
      }
      const status = document.createElement("div");
      status.textContent = (message.text || []).join("\n");
      status.style.cssText = "white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;";
      box.append(status);
      this.setSize([Math.max(this.size[0], 500), Math.max(this.size[1], 980)]);
      this.setDirtyCanvas(true, true);
    };
  },
  loadedGraphNode(node) {
    if (node.comfyClass === "ImageSongRender") {
      setTimeout(() => repairLegacyRendererWidgetOrder(node), 0);
    }
  },
});
