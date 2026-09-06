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
  return element("label", { style: "display:grid;gap:4px;font-size:12px;color:#cbd5e1;" }, [
    element("span", { text: label }), control,
  ]);
}

function textArea(value, placeholder, onInput, rows = 3) {
  return element("textarea", {
    value: value || "", placeholder, rows,
    style: "box-sizing:border-box;width:100%;resize:vertical;border:1px solid #4b5563;border-radius:6px;background:#111827;color:#f8fafc;padding:7px;font:12px/1.35 sans-serif;",
    oninput: (event) => onInput(event.target.value),
  });
}

function textInput(value, placeholder, onInput, type = "text") {
  return element("input", {
    value: value ?? "", placeholder, type,
    style: "box-sizing:border-box;width:100%;border:1px solid #4b5563;border-radius:6px;background:#111827;color:#f8fafc;padding:7px;font:12px sans-serif;",
    oninput: (event) => onInput(event.target.value),
  });
}

function button(label, onClick, kind = "normal") {
  const colors = kind === "remove"
    ? "background:#4c1d1d;border-color:#7f1d1d;color:#fecaca;"
    : kind === "add"
      ? "background:#164e63;border-color:#0e7490;color:#cffafe;font-weight:700;"
      : "background:#273449;border-color:#475569;color:#e2e8f0;";
  return element("button", {
    type: "button", text: label,
    style: `cursor:pointer;border:1px solid;border-radius:6px;padding:7px 10px;font:12px sans-serif;${colors}`,
    onclick: (event) => { event.preventDefault(); event.stopPropagation(); onClick(event); },
  });
}

function installMediaEditor(node) {
  const rawWidget = node.widgets?.find((widget) => widget.name === "media_inputs_json");
  if (!rawWidget || node.karaokeMediaEditor) return;

  const legacyInputs = new Set(["reference_images", "sound_effects", "image", "sfx_audio"]);
  for (let index = (node.inputs?.length || 0) - 1; index >= 0; index -= 1) {
    if (legacyInputs.has(node.inputs[index].name)) node.removeInput(index);
  }

  let state = parseMediaState(rawWidget.value);
  const originalIndex = node.widgets.indexOf(rawWidget);
  node.widgets.splice(originalIndex, 1);

  const root = element("div", {
    style: "box-sizing:border-box;height:650px;overflow:auto;display:flex;flex-direction:column;gap:12px;padding:10px;background:#0b1220;color:#f8fafc;border:1px solid #334155;border-radius:8px;",
  });
  root.addEventListener("pointerdown", (event) => event.stopPropagation());

  let editorWidget;
  const dirty = () => {
    editorWidget.callback?.(JSON.stringify(state));
    node.graph?.setDirtyCanvas(true, true);
  };

  const render = () => {
    root.replaceChildren();
    root.append(element("div", { style: "display:flex;align-items:center;justify-content:space-between;gap:8px;" }, [
      element("strong", { text: `Reference Images (${state.references.length})`, style: "font:700 14px sans-serif;" }),
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
      const card = element("div", { style: "display:grid;gap:8px;padding:10px;border:1px solid #334155;border-radius:8px;background:#111827;" }, [
        element("div", { style: "display:flex;align-items:center;justify-content:space-between;gap:8px;" }, [
          element("strong", { text: `Image ${index + 1}`, style: "font:700 13px sans-serif;" }),
          button("Remove", () => { state.references.splice(index, 1); render(); dirty(); }, "remove"),
        ]),
        element("div", { style: "display:grid;grid-template-columns:auto minmax(0,1fr);align-items:center;gap:8px;" }, [choose, fileName]),
      ]);
      if (item.filename) {
        card.append(element("img", { src: assetURL(item), alt: fileName.textContent,
          style: "width:100%;max-height:180px;object-fit:contain;border-radius:6px;background:#020617;" }));
      }
      card.append(field("Who or what is shown, and how should this image be used?",
        textArea(item.instruction, "Example: Main character. Preserve their identity and use them in chorus scenes.", (value) => { item.instruction = value; dirty(); }, 3)));
      root.append(card);
    });

    root.append(element("div", { style: "display:flex;align-items:center;justify-content:space-between;gap:8px;margin-top:4px;" }, [
      element("strong", { text: `Sound Effects (${state.sound_effects.length})`, style: "font:700 14px sans-serif;" }),
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
      const card = element("div", { style: "display:grid;gap:8px;padding:10px;border:1px solid #334155;border-radius:8px;background:#111827;" }, [
        element("div", { style: "display:flex;align-items:center;justify-content:space-between;gap:8px;" }, [
          element("strong", { text: `Sound Effect ${index + 1}`, style: "font:700 13px sans-serif;" }),
          button("Remove", () => { state.sound_effects.splice(index, 1); render(); dirty(); }, "remove"),
        ]), fileRow,
      ]);
      if (item.filename) {
        card.append(element("audio", { src: assetURL(item), controls: true, preload: "metadata", style: "width:100%;height:34px;" }));
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
  };

  editorWidget = node.addDOMWidget("media_inputs_json", "karaoke_media_editor", root, {
    serialize: true,
    hideOnZoom: false,
    getValue() {
      return JSON.stringify(state);
    },
    setValue(value) {
      state = parseMediaState(value);
      render();
    },
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
  node.setSize([Math.max(node.size[0], 680), Math.max(node.size[1], 1180)]);
}

app.registerExtension({
  name: "local.ImageMusicKaraoke",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "ImageSongPlan") {
      const originalCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const result = originalCreated?.apply(this, arguments);
        installMediaEditor(this);
        return result;
      };
      return;
    }
    if (nodeData.name !== "ImageSongRender") return;
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
});
