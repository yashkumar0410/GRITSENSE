/* ==========================================================================
   GRITSENSE — front end
   No build step, no dependencies. Talks to the existing Flask API.
   ========================================================================== */
(() => {
  "use strict";

  /* ---------------------------------------------------------------- utils */
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v, d = 0) => `${((Number(v) || 0) * 100).toFixed(d)}%`;
  const clock = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${(s % 60).toFixed(1).padStart(4, "0")}`;
  const dur = (s) => (s < 60 ? `${Math.round(s)}s` : `${Math.floor(s / 60)}m ${String(Math.round(s % 60)).padStart(2, "0")}s`);
  const ago = (t) => {
    const s = Date.now() / 1000 - t;
    if (s < 60) return "just now";
    if (s < 3600) return `${Math.floor(s / 60)} min ago`;
    if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
    return new Date(t * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" });
  };
  const dateFmt = (t) => new Date(t * 1000).toLocaleString(undefined, { month: "short", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit" });
  const cap = (s) => String(s ?? "").replace(/(^|[\s-])\S/g, (m) => m.toUpperCase());
  const clamp = (n, a, b) => Math.min(b, Math.max(a, n));
  const shortGpu = (s) => String(s || "")
    .replace(/NVIDIA\s+/i, "")
    .replace(/GeForce\s+/i, "")
    .replace(/\s*Laptop GPU/i, "")
    .replace(/\s*GPU\b/i, "")
    .trim();

  async function api(path, opts) {
    const res = await fetch(path, opts);
    if (!res.ok) {
      const text = await res.text();
      const msg = (text.match(/<p>(.*?)<\/p>/s) || [])[1] || text || res.statusText;
      throw new Error(msg.replace(/<[^>]+>/g, "").trim() || `Request failed (${res.status})`);
    }
    return res.status === 204 ? null : res.json();
  }

  let toastTimer;
  function toast(msg, err = false) {
    const el = $("#toast");
    el.textContent = msg;
    el.className = "toast" + (err ? " err" : "");
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (el.hidden = true), 4600);
  }

  /* ---------------------------------------------------------------- state */
  let jobs = [];
  let jobsLoading = false;
  let jobsLoaded = false;
  let view = "landing";
  let current = null;              // { id, job, loaded }
  let pollTimer = null;
  let jobActionsKey = "";

  let rows = [];
  let lines = [];
  let ga = null;
  let fps = 30;
  let raf = 0;
  let lastReadIdx = -1;
  let lastReadAt = 0;
  let lastNow = -1;
  let lastWin = -1;
  let feedHover = false;
  let feedKind = "all";
  let archiveFilter = "all";
  let zoneMapData = { zones: {}, total: 0 };

  let showcaseId = null;
  let showcaseLoading = false;

  const form = { mode: "full", nlp: "local", device: "cuda", len: "full", file: null, vw: 0, vh: 0 };
  const probeVideo = document.createElement("video");
  probeVideo.muted = true;
  probeVideo.playsInline = true;
  probeVideo.preload = "metadata";

  const STAGES = {
    full: [
      ["prepare", "Prepare", "Normalise 1080p"],
      ["detect", "Detect & track", "Players · pose · ball"],
      ["ball", "Ball track", "Re-detect and clean"],
      ["graphs", "Graph models", "GAT · HRN · activity"],
      ["dashboard", "Render", "Dashboard · commentary"],
      ["finalize", "Finalize", "Browser encode"],
    ],
    demo: [
      ["prepare", "Load sample", "Bundled detections"],
      ["dashboard", "Render", "Dashboard · commentary"],
      ["finalize", "Finalize", "Browser encode"],
    ],
  };
  const STAGE_TEXT = {
    queued: "Waiting in queue",
    prepare: "Preparing the footage",
    detect: "Detecting players, pose and the ball",
    ball: "Re-detecting and cleaning the ball track",
    graphs: "Building graph models and scoring attacks",
    dashboard: "Rendering the dashboard and commentary",
    finalize: "Encoding the result for the browser",
  };
  const ACTION_COLOR = { pass: "#60a5fa", set: "#2de2be", spike: "#fbbf24", winpoint: "#f43f5e" };
  const TEAM_COLOR = ["#f2c230", "#ef5a4a"];
  const ACTION_LABEL = { pass: "Pass", set: "Set", spike: "Spike", winpoint: "Win point" };
  const ZONES = [
    [4, 0, 0], [3, 1, 0], [2, 2, 0],
    [5, 0, 1], [6, 1, 1], [1, 2, 1],
  ];

  /* ---------------------------------------------------------------- system */
  async function loadSystem() {
    try {
      const s = await api("/api/system");
      const gpu = s.gpu ? String(s.gpu) : "";
      const gpuShort = shortGpu(gpu);
      const model = String(s.model || "commentary model").split("/").pop().replace("-Instruct", "");
      $("#sys-status").innerHTML = `<span class="dot ${gpu ? "ok" : "warn"}"></span><span>${gpu ? `GPU · ${esc(gpuShort)}` : "CPU mode"}</span>`;
      $("#sys-status").title = `${gpu ? `Compute: ${gpu}` : "Compute: CPU"} · Commentary model: ${model} ${s.model_cached ? "cached" : "not cached"}`;
      $("#foot-status").innerHTML = `<span class="dot ${gpu ? "ok" : "warn"}"></span><span>${gpu ? esc(gpuShort) : "CPU mode"} · ${esc(model)} ${s.model_cached ? "cached" : "not cached"}</span>`;
      $("#hero-compute").textContent = gpu ? `${gpuShort} available` : "CPU mode";
      $("#eng-device").textContent = gpuShort || "NVIDIA RTX (CUDA)";
      const optCuda = $("#opt-cuda") || document.querySelector("#device option[value='cuda']");
      if (optCuda) {
        optCuda.textContent = gpu ? `NVIDIA RTX (${gpuShort}) — CUDA` : "NVIDIA RTX (CUDA)";
      }
      if (gpu) {
        form.device = "cuda";
        const devSel = $("#device");
        if (devSel) devSel.value = "cuda";
      }
      $("#eng-engine").textContent = s.model_cached ? "Offline LLM cached" : "Offline LLM / template";
      if (!s.sample) {
        const demo = $('[data-mode="demo"]');
        if (demo) { demo.disabled = true; demo.title = "Sample assets are not available on this machine."; }
      }
    } catch {
      $("#sys-status").innerHTML = `<span class="dot bad"></span><span>Server offline</span>`;
      $("#foot-status").innerHTML = `<span class="dot bad"></span><span>Server offline</span>`;
    }
  }

  /* ---------------------------------------------------------------- jobs */
  const isActive = (j) => j.status === "running" || j.status === "queued";
  const modeLabel = (j) => (j.mode === "demo" ? "Sample render" : "Full pipeline");
  const jobTitle = (j) => j.title || j.id;
  const thumbOf = (j) => (j.status === "done" && j.snapshots?.length ? `/files/${j.id}/dashboard/${encodeURIComponent(j.snapshots[0])}` : "");
  const statusText = (j) => {
    if (j.status === "done") return "Complete";
    if (j.status === "failed") return "Failed";
    if (j.status === "running") return `${Math.round((j.progress || 0) * 100)}% running`;
    return "Queued";
  };

  async function loadJobs() {
    if (jobsLoading) return;
    jobsLoading = true;
    try {
      const out = await api("/api/jobs");
      jobs = (Array.isArray(out) ? out : []).slice().sort((a, b) => (b.created || 0) - (a.created || 0));
      jobsLoaded = true;
      renderArchiveNumbers();
      if (view === "runs") renderArchive();
      if (view === "landing") {
        renderLandingRuns();
        renderShowcase();
      }
    } catch {
      if (!jobsLoaded) {
        $("#archive-summary").innerHTML = `<div><b>—</b><span>Archive offline</span></div>`;
      }
    } finally {
      jobsLoading = false;
    }
  }

  function renderArchiveNumbers() {
    const done = jobs.filter((j) => j.status === "done").length;
    const active = jobs.filter(isActive).length;
    const failed = jobs.filter((j) => j.status === "failed").length;
    const total = jobs.length;
    $("#count-all").textContent = total;
    $("#count-done").textContent = done;
    $("#count-running").textContent = active;
    $("#count-failed").textContent = failed;
    $("#archive-summary").innerHTML = `
      <div><b>${total}</b><span>Total runs</span></div>
      <div><b>${done}</b><span>Complete</span></div>
      <div><b>${active}</b><span>In progress</span></div>`;
  }

  /* ---------------------------------------------------------------- landing */
  function renderLandingRuns() {
    const section = $("#archive-preview");
    const host = $("#landing-runs");
    const latest = jobs.slice(0, 4);
    section.hidden = latest.length === 0;
    if (!latest.length) return;
    host.innerHTML = latest.map((j) => {
      const t = thumbOf(j);
      return `<a class="preview-run" href="#/job/${encodeURIComponent(j.id)}">
        <span class="preview-thumb"${t ? ` style="background-image:url('${esc(t)}')"` : ""}></span>
        <span class="preview-main"><b>${esc(jobTitle(j))}</b><small>${esc(j.teams?.[0] ?? "—")} vs ${esc(j.teams?.[1] ?? "—")} · ${esc(ago(j.created))} · ${esc(modeLabel(j))}</small></span>
        <span class="preview-mode">${esc(statusText(j))}</span>
        <span class="preview-date">${esc((j.frames || 0) ? `${j.frames} frames` : "—")}</span>
        <span class="preview-arrow">↗</span>
      </a>`;
    }).join("");
  }

  async function renderShowcase() {
    if (showcaseLoading) return;
    const job = jobs.find((j) => j.status === "done");
    const video = $("#showcase-video");
    const empty = $("#showcase-empty");

    if (!job) {
      showcaseId = null;
      video.pause();
      video.hidden = true;
      video.removeAttribute("src");
      empty.hidden = false;
      $("#showcase-title").textContent = "Latest analysis";
      $("#showcase-mode").textContent = "—";
      $("#showcase-teams").textContent = "No completed run";
      $("#showcase-frames").textContent = "—";
      $("#showcase-fps").textContent = "—";
      $("#showcase-date").textContent = "—";
      $("#showcase-stats").innerHTML = `<div><span>Frames analysed</span><b>—</b></div><div><span>Attack chances</span><b>—</b></div><div><span>Peak point chance</span><b>—</b></div>`;
      $("#showcase-court").innerHTML = `<p class="muted" style="font-size:11px;line-height:1.5">Target zones appear after a completed analysis.</p>`;
      $("#showcase-commentary").innerHTML = `<p class="muted">Run a match to generate a tactical read.</p>`;
      $("#showcase-open").href = "#/runs";
      return;
    }

    if (showcaseId === job.id && !video.hidden) return;
    showcaseLoading = true;
    try {
      showcaseId = job.id;
      video.hidden = false;
      empty.hidden = true;
      video.muted = true;
      video.poster = thumbOf(job) || "";
      video.src = `/files/${job.id}/dashboard.mp4`;
      video.play().catch(() => {});
      $("#showcase-title").textContent = jobTitle(job);
      $("#showcase-mode").textContent = modeLabel(job);
      $("#showcase-teams").textContent = `${job.teams?.[0] ?? "—"} vs ${job.teams?.[1] ?? "—"}`;
      $("#showcase-frames").textContent = `${job.frames || "—"} frames`;
      $("#showcase-fps").textContent = `${job.fps || 30} fps`;
      $("#showcase-date").textContent = dateFmt(job.created);
      $("#showcase-open").href = `#/job/${encodeURIComponent(job.id)}`;

      const [ins, comm] = await Promise.all([
        api(`/api/jobs/${job.id}/insights`).catch(() => []),
        api(`/files/${job.id}/commentary.json`).catch(() => []),
      ]);
      if (showcaseId !== job.id) return;
      const data = Array.isArray(ins) ? ins : [];
      const attack = data.filter((r) => r.attack_chance);
      let chances = 0;
      data.forEach((r, i) => { if (r.attack_chance && !data[i - 1]?.attack_chance) chances++; });
      const peak = attack.length ? attack.reduce((m, r) => ((r.epv || 0) > (m.epv || 0) ? r : m)) : null;
      $("#showcase-stats").innerHTML = `
        <div><span>Frames analysed</span><b>${data.length || job.frames || "—"}</b></div>
        <div><span>Attack chances</span><b>${attack.length ? chances : "—"}</b></div>
        <div><span>Peak point chance</span><b>${peak ? pct(peak.epv) : "—"}</b></div>`;
      $("#showcase-court").innerHTML = data.length ? courtSvg(zoneStats(data).zones, zoneStats(data).total) : `<p class="muted" style="font-size:11px">No zone calls in this run.</p>`;
      const list = Array.isArray(comm) ? comm : [];
      $("#showcase-commentary").innerHTML = list.length
        ? list.slice(0, 2).map((l) => `<div><div class="rc-meta"><span>${clock(l.t)}</span><span>${esc(l.kind)}</span><span>${esc(l.source)}</span></div><p>${esc(l.text)}</p></div>`).join("")
        : `<p class="muted">No commentary lines were written for this run.</p>`;
    } catch {
      showcaseId = null;
    } finally {
      showcaseLoading = false;
    }
  }

  function zoneStats(data) {
    const zones = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0 };
    let total = 0;
    data.filter((r) => r.attack_chance).forEach((r) => {
      const z = Number(r.target_zone);
      if (zones[z] !== undefined) { zones[z]++; total++; }
    });
    return { zones, total };
  }

  function courtSvg(zones, total) {
    const W = 300, H = 236;
    const x = 18, y = 30, w = 264, h = 184;
    const cw = w / 3, rh = h / 2;
    const max = Math.max(1, ...Object.values(zones));
    const cells = ZONES.map(([z, c, r]) => {
      const v = zones[z] || 0;
      const share = total ? v / total : 0;
      const alpha = 0.025 + 0.42 * (v / max);
      const cx = x + c * cw, cy = y + r * rh;
      return `<g>
        <rect class="zone-cell" data-zone="${z}" x="${cx}" y="${cy}" width="${cw}" height="${rh}" fill="rgba(51,85,255,${alpha.toFixed(3)})" stroke="rgba(236,238,244,.12)" stroke-width="1"><title>Zone ${z} · ${pct(share)} of calls</title></rect>
        <text x="${cx + 8}" y="${cy + 16}" fill="rgba(242,240,233,.55)" font-family="JetBrains Mono, monospace" font-size="8" letter-spacing="1.4">Z${z}</text>
        <text x="${cx + cw - 8}" y="${cy + rh - 9}" text-anchor="end" fill="${alpha > 0.26 ? "#f2f0e9" : "rgba(242,240,233,.45)"}" font-family="JetBrains Mono, monospace" font-size="${alpha > 0.26 ? 13 : 9}">${v ? pct(share) : "—"}</text>
      </g>`;
    }).join("");
    return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Target zone distribution">
      <rect x="${x}" y="${y}" width="${w}" height="${h}" fill="none" stroke="rgba(236,238,244,.32)"/>
      <path d="M${x} ${y + rh}H${x + w}M${x + cw} ${y}V${y + h}M${x + 2 * cw} ${y}V${y + h}" fill="none" stroke="rgba(236,238,244,.14)"/>
      <path d="M${x - 8} ${y}H${x + w + 8}" stroke="rgba(242,240,233,.75)" stroke-width="2"/>
      <text x="${x + w + 8}" y="${y - 7}" text-anchor="end" fill="rgba(242,240,233,.42)" font-family="JetBrains Mono, monospace" font-size="7.5" letter-spacing="1.6">NET</text>
      <text x="${x}" y="${y + h + 17}" fill="rgba(242,240,233,.32)" font-family="JetBrains Mono, monospace" font-size="7.5" letter-spacing="1.2">18.0 M</text>
      <path d="M${x} ${y + h + 6}v4M${x + w} ${y + h + 6}v4H${x}" fill="none" stroke="rgba(236,238,244,.25)"/>
      ${cells}
    </svg>`;
  }

  /* ---------------------------------------------------------------- intake */
  function segBind(id, key, cb) {
    $$(`#${id} button`).forEach((b) => b.addEventListener("click", () => {
      if (b.disabled) return;
      $$(`#${id} button`).forEach((x) => x.classList.toggle("on", x === b));
      form[key] = b.dataset[key];
      cb?.();
      refreshForm();
    }));
  }

  function setMode(mode) {
    form.mode = mode;
    $$("#mode-seg button").forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
    $$("[data-show]").forEach((el) => (el.hidden = el.dataset.show !== mode));
    refreshForm();
  }

  function selectedDeviceLabel() {
    const v = $("#device").value;
    return { cuda: "NVIDIA RTX (CUDA)", cpu: "CPU mode", xpu: "Intel Arc (XPU)", auto: "Automatic" }[v] || v;
  }

  function refreshForm() {
    const demo = form.mode === "demo";
    const btn = $("#start-btn");
    const hint = $("#submit-hint");
    const title = $("#title").value.trim();
    const left = $("#team-left").value.trim().toUpperCase() || "BRA";
    const right = $("#team-right").value.trim().toUpperCase() || "JPN";

    $("#sum-source").textContent = demo ? "Bundled sample · BRA–JPN" : (form.file ? form.file.name : "No footage selected");
    $("#sum-mode").textContent = demo ? "Sample render" : "Full pipeline";
    $("#sum-teams").textContent = `${left} vs ${right}`;
    $("#sum-device").textContent = selectedDeviceLabel();
    $("#sum-commentary").textContent = form.nlp === "local" ? "Offline LLM" : "Templates";
    $("#sum-source").title = demo ? "Bundled sample" : (form.file?.name || "");
    $("#source-state").textContent = form.file ? "File ready" : "No file";
    $("#source-state").classList.toggle("live", Boolean(form.file));

    $("#mini-flow").innerHTML = (demo
      ? ["Load sample", "Render dashboard", "Finalize"]
      : ["Prepare", "Detect & track", "Ball track", "Graph models", "Render dashboard"])
      .map((s) => `<li>${s}</li>`).join("");

    if (demo) {
      btn.disabled = false;
      hint.textContent = "Runs from the bundled sample's existing detections.";
    } else {
      btn.disabled = !form.file;
      hint.textContent = form.file
        ? "Ready. Court corners come from the project's annotations.json."
        : "Choose a video to begin.";
    }

    const preview = form.len === "preview";
    $("#config-note").textContent = preview
      ? `Stops detection after ${$("#max-frames").value || 150} frames; graph models, EPV and the dashboard still run on that slice.`
      : "The full pipeline runs detection, ball cleanup, graph models, EPV and the rendered dashboard.";
    $("#title").placeholder = demo ? "Sample rally · BRA vs JPN" : (form.file ? form.file.name.replace(/\.[^.]+$/, "") : "Set 2 · rally 14");
    if (title && !demo && !form.file) { /* keep typed title */ }
  }

  function setFile(file) {
    if (!file) return;
    if (!file.type.startsWith("video/") && !/\.(mp4|mov|avi|mkv|m4v|webm)$/i.test(file.name)) {
      toast("That doesn't look like a video file.", true);
      return;
    }
    form.file = file;
    form.vw = 0; form.vh = 0;
    $("#drop").classList.add("has-file");
    $(".drop-empty").hidden = true;
    $(".drop-file").hidden = false;
    $("#file-name").textContent = file.name;
    $("#file-meta").textContent = `${(file.size / 1048576).toFixed(1)} MB · reading…`;
    $("#preview-name").textContent = file.name;
    $("#preview-res").textContent = "Reading…";
    $("#preview-dur").textContent = "Reading…";
    $("#source-preview").hidden = false;
    URL.revokeObjectURL(probeVideo.src);
    const url = URL.createObjectURL(file);
    probeVideo.src = url;
    $("#preview-video").src = url;
    $("#preview-video").load();
    if (!$("#title").value) $("#title").placeholder = file.name.replace(/\.[^.]+$/, "");
    refreshForm();
  }

  function initIntake() {
    segBind("mode-seg", "mode", () => {
      $$("[data-show]").forEach((el) => (el.hidden = el.dataset.show !== form.mode));
    });
    segBind("nlp-seg", "nlp");
    segBind("len-seg", "len", () => {
      $("#frames-wrap").hidden = form.len !== "preview";
    });

    $("#device").addEventListener("change", (e) => { form.device = e.target.value; refreshForm(); });
    ["#team-left", "#team-right", "#title", "#max-frames"].forEach((sel) => $(sel).addEventListener("input", refreshForm));

    probeVideo.addEventListener("loadedmetadata", () => {
      form.vw = probeVideo.videoWidth;
      form.vh = probeVideo.videoHeight;
      const meta = `${(form.file.size / 1048576).toFixed(1)} MB · ${form.vw}×${form.vh} · ${dur(probeVideo.duration)}`
        + (form.vh && form.vh !== 1080 ? " · scaled to 1080p" : "");
      $("#file-meta").textContent = meta;
      $("#preview-res").textContent = `${form.vw} × ${form.vh}`;
      $("#preview-dur").textContent = dur(probeVideo.duration);
      $("#preview-name").textContent = form.file.name;
      updateSteps(2);
    });
    probeVideo.addEventListener("error", () => {
      $("#file-meta").textContent = `${(form.file.size / 1048576).toFixed(1)} MB · preview unavailable`;
      $("#preview-res").textContent = "Unavailable";
      $("#preview-dur").textContent = "Unavailable";
    });

    const drop = $("#drop");
    $("#video-input").addEventListener("change", (e) => {
      setFile(e.target.files[0]);
      e.target.value = "";
      updateSteps(2);
    });
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => { setFile(e.dataTransfer.files[0]); updateSteps(2); });
    drop.addEventListener("click", (e) => { if (form.file) e.preventDefault(); });
    $("#file-clear").addEventListener("click", (e) => {
      e.preventDefault(); e.stopPropagation();
      $("#video-input").click();
    });

    $$(".step").forEach((step) => step.addEventListener("focusin", () => {
      const idx = Number(step.querySelector(".step-index")?.textContent || 0);
      if (idx) updateSteps(idx);
    }));

    $("#setup-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const btn = $("#start-btn");
      const mode = form.mode;
      if (mode === "full" && !form.file) { toast("Choose a video first.", true); return; }
      const fd = new FormData();
      fd.append("mode", mode);
      const base = form.file ? form.file.name.replace(/\.[^.]+$/, "") : "";
      fd.append("title", $("#title").value.trim() || (mode === "demo" ? "Sample rally · BRA vs JPN" : base || "Untitled analysis"));
      fd.append("team_left", ($("#team-left").value.trim() || "BRA").toUpperCase());
      fd.append("team_right", ($("#team-right").value.trim() || "JPN").toUpperCase());
      fd.append("nlp", form.nlp);
      fd.append("device", $("#device").value);
      if (form.len === "preview") fd.append("max_frames", String(Math.max(30, Number($("#max-frames").value) || 150)));
      if (mode === "full") fd.append("video", form.file);

      btn.disabled = true;
      const label = btn.querySelector("span");
      const old = label.textContent;
      label.textContent = mode === "full" ? "Uploading…" : "Starting…";
      try {
        const job = await api("/api/jobs", { method: "POST", body: fd });
        updateSteps(3);
        location.hash = `#/job/${job.id}`;
        loadJobs();
      } catch (err) {
        toast(err.message, true);
        label.textContent = old;
        refreshForm();
      } finally {
        label.textContent = old;
      }
    });

    // mode buttons: keep source preview in sync with the visible step
    $$("#mode-seg button").forEach((b) => b.addEventListener("click", () => {
      const demo = b.dataset.mode === "demo";
      $("#mode-seg").querySelector(".mode-note").textContent = demo ? "Bundled detections" : "One run at a time";
      updateSteps(1);
    }));
  }

  function updateSteps(n) {
    $$(".intake-steps li").forEach((li, i) => li.classList.toggle("on", i === n - 1));
  }

  /* ---------------------------------------------------------------- job */
  function stopJobView() {
    clearTimeout(pollTimer);
    pollTimer = null;
    cancelAnimationFrame(raf);
    raf = 0;
    const player = $("#player");
    if (player && !player.paused) player.pause();
    current = null;
    jobActionsKey = "";
    lastReadIdx = -1;
    lastNow = -1;
    lastWin = -1;
  }

  async function openJob(id) {
    clearTimeout(pollTimer);
    pollTimer = null;
    if (current?.id !== id) {
      current = { id, job: null, loaded: false };
      jobActionsKey = "";
      $("#proc").hidden = true;
      $("#results").hidden = true;
      $("#failed").hidden = true;
      $("#job-title").textContent = "Loading…";
      $("#job-meta").innerHTML = "";
      $("#job-actions").innerHTML = "";
      lastReadIdx = -1; lastNow = -1; lastWin = -1;
    }
    let job;
    try {
      job = await api(`/api/jobs/${encodeURIComponent(id)}`);
    } catch {
      $("#job-title").textContent = "Run not found";
      $("#job-crumb").textContent = `Run / ${id}`;
      $("#failed").hidden = false;
      $("#err-msg").textContent = "This run is no longer on disk, or the server restarted without it.";
      $("#err-log").textContent = "";
      return;
    }
    if (current?.id !== id) return;
    current.job = job;
    renderJobHead(job);
    if (job.status === "done") {
      $("#proc").hidden = true;
      $("#failed").hidden = true;
      if (!current.loaded) {
        current.loaded = true;
        await renderResults(job);
        loadJobs();
      }
    } else if (job.status === "failed") {
      $("#proc").hidden = true;
      $("#results").hidden = true;
      $("#failed").hidden = false;
      $("#err-msg").textContent = job.error || "Unknown error";
      $("#err-log").textContent = (job.log || []).join("\n");
      loadJobs();
    } else {
      $("#proc").hidden = false;
      $("#results").hidden = true;
      $("#failed").hidden = true;
      renderProgress(job);
      pollTimer = setTimeout(() => {
        if (view === "job" && current?.id === id) openJob(id);
      }, 1600);
      loadJobs();
    }
  }

  function renderJobHead(j) {
    $("#job-crumb").textContent = `Run / ${j.id}`;
    $("#job-title").textContent = jobTitle(j);
    const bits = [
      `<span class="team-tag"><i class="t0"></i><b>${esc(j.teams?.[0] ?? "—")}</b> vs <b>${esc(j.teams?.[1] ?? "—")}</b><i class="t1"></i></span>`,
      `<span>${esc(modeLabel(j))}</span>`,
      `<span>${j.nlp === "local" ? "Offline LLM" : "Templates"}</span>`,
      `<span>${esc(j.device || "auto")}</span>`,
      `<span>${esc(dateFmt(j.created))}</span>`,
    ];
    if (j.frames) bits.push(`<span>${j.frames} frames</span>`);
    if (j.fps) bits.push(`<span>${j.fps} fps</span>`);
    if (j.started && j.finished) bits.push(`<span>Processed in ${dur(j.finished - j.started)}</span>`);
    $("#job-meta").innerHTML = bits.join("");

    const acts = [];
    if (j.status === "done") {
      acts.push(`<a class="btn btn-line" href="/files/${j.id}/dashboard.mp4?download=1">Video ↓</a>`);
      acts.push(`<a class="btn btn-line" href="/files/${j.id}/dashboard/commentary.txt?download=1">Commentary ↓</a>`);
      acts.push(`<a class="btn btn-line" href="/files/${j.id}/dashboard/dashboard_insights.json?download=1">Insights ↓</a>`);
    }
    if (j.status === "done" || j.status === "failed") acts.push(`<button class="btn btn-line danger" id="del-btn" type="button">Delete</button>`);
    else acts.push(`<button class="btn btn-line danger" id="cancel-btn" type="button">Cancel run</button>`);

    const key = `${j.status}|${j.id}`;
    if (jobActionsKey === key) return;
    jobActionsKey = key;
    $("#job-actions").innerHTML = acts.join("");
    $("#cancel-btn")?.addEventListener("click", async () => {
      if (!window.confirm("Stop this run?")) return;
      try { await api(`/api/jobs/${j.id}/cancel`, { method: "POST" }); toast("Run cancelled"); openJob(j.id); }
      catch (err) { toast(err.message, true); }
    });
    $("#del-btn")?.addEventListener("click", async () => {
      if (!window.confirm("Delete this run and its generated files?")) return;
      try {
        await api(`/api/jobs/${j.id}`, { method: "DELETE" });
        stopJobView();
        location.hash = "#/runs";
        loadJobs();
      } catch (err) { toast(err.message, true); }
    });
  }

  function renderProgress(j) {
    const p = clamp(j.progress || 0, 0, 1);
    $("#pct").textContent = String(Math.floor(p * 100));
    $("#progress-fill").style.width = `${(p * 100).toFixed(1)}%`;
    $("#proc-status").textContent = j.status === "queued" ? "Queued" : "Processing";
    $("#proc-stage").textContent = STAGE_TEXT[j.stage] || "Working";
    if (j.started) {
      const elapsed = Date.now() / 1000 - j.started;
      $("#proc-times").textContent = `Elapsed ${dur(elapsed)}${j.frames ? ` · ${j.frames} frames` : ""}${j.engine ? ` · commentary: ${j.engine}` : ""}`;
    } else {
      $("#proc-times").textContent = "Another run is in progress. This one starts automatically.";
    }
    const stages = STAGES[j.mode] || STAGES.full;
    const idx = stages.findIndex(([k]) => k === j.stage);
    $("#stepper").innerHTML = stages.map(([, label, desc], i) => `
      <li class="${i < idx ? "done" : i === idx ? "active" : ""}">
        <span class="si"></span>${esc(label)}<span class="stage-desc">${esc(desc)}</span>
      </li>`).join("");
    const log = $("#log");
    const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
    log.textContent = (j.log || []).join("\n");
    if (atBottom) log.scrollTop = log.scrollHeight;
    $("#log-count").textContent = `${(j.log || []).length} lines`;
  }

  /* ---------------------------------------------------------------- results */
  async function renderResults(j) {
    $("#results").hidden = false;
    $("#proc").hidden = true;
    $("#failed").hidden = true;
    $("#job-title").textContent = jobTitle(j);
    $("#video-time").textContent = "00:00.0";
    fps = j.fps || 30;
    const player = $("#player");
    player.poster = thumbOf(j) || "";
    player.src = `/files/${j.id}/dashboard.mp4`;
    player.load();

    const [ins, comm, group] = await Promise.all([
      api(`/api/jobs/${j.id}/insights`).catch(() => []),
      api(`/files/${j.id}/commentary.json`).catch(() => []),
      api(`/files/${j.id}/dashboard/group_activity.json`).catch(() => null),
    ]);
    rows = Array.isArray(ins) ? ins : [];
    lines = Array.isArray(comm) ? comm : [];
    ga = group && Array.isArray(group.windows) && group.windows.length ? group : null;

    feedKind = "all";
    $$("#feed-filters button").forEach((b) => b.classList.toggle("on", b.dataset.kind === "all"));
    $("#engine-tag").textContent = j.engine ? `· ${j.engine}` : "";
    $("#commentary-count").textContent = lines.length;

    renderStats();
    renderFeed();
    renderChart();
    renderBreakdowns(j);
    renderTeamShape(j);
    renderCourtMap();
    renderRally(j);
    renderSnaps(j);
    renderFrameTable(j);
    updateReadout(0);
    switchTab("commentary");
    syncToVideo();

// phases and snapshots are persistent cards
  }

  function renderStats() {
    const attack = rows.filter((r) => r.attack_chance);
    let chances = 0;
    rows.forEach((r, i) => { if (r.attack_chance && !rows[i - 1]?.attack_chance) chances++; });
    const peak = attack.length ? attack.reduce((m, r) => ((r.epv || 0) > (m.epv || 0) ? r : m)) : null;
    const avg = attack.length ? attack.reduce((s, r) => s + (r.epv || 0), 0) / attack.length : 0;
    const swaps = rows.filter((r, i) => i && r.attack_team != null && rows[i - 1].attack_team != null && r.attack_team !== rows[i - 1].attack_team).length;
    const tiles = [
      ["Footage analysed", rows.length ? dur(rows.length / fps) : "—", rows.length ? `${rows.length} frames` : "no frames", ""],
      ["Attack chances", attack.length ? String(chances) : "—", attack.length ? `${dur(attack.length / fps)} in total` : "none detected", ""],
      ["Peak point chance", peak ? pct(peak.epv) : "—", peak ? `at ${clock(peak.frame / fps)} · zone ${peak.target_zone ?? "—"}` : "no attack chances", "accent"],
      ["Average point chance", attack.length ? pct(avg) : "—", "during attack chances", ""],
      ["Possession changes", rows.length ? String(swaps) : "—", "attack-team swaps", ""],
      ["Commentary lines", lines.length ? String(lines.length) : "—", "time-stamped reads", ""],
    ];
    $("#stats").innerHTML = tiles.map(([k, v, s, c]) => `<div class="stat ${c}"><small>${k}</small><b>${v}</b><span>${s}</span></div>`).join("");
  }

  function renderFeed() {
    const items = lines.map((l, i) => ({ l, i })).filter(({ l }) => feedKind === "all" || l.kind === feedKind);
    $("#feed").innerHTML = items.length ? items.map(({ l, i }) => `
      <li class="k-${esc(l.kind)}" data-i="${i}">
        <div class="fh"><span class="ft">${clock(l.t)}</span><span class="fk">${esc(l.kind)}</span><span class="fs">${esc(l.source)}</span></div>
        <p>${esc(l.text)}</p>
      </li>`).join("") : `<li class="empty" style="opacity:.6;cursor:default">No commentary lines for this filter.</li>`;
    $$("#feed li[data-i]").forEach((li) => li.addEventListener("click", () => seek(lines[Number(li.dataset.i)].t)));
    lastNow = -1;
  }

  function seek(t) {
    const p = $("#player");
    if (!p) return;
    const max = Number.isFinite(p.duration) ? p.duration : t;
    p.currentTime = clamp(t, 0, Math.max(0, max - 0.05));
    if (p.paused) p.play().catch(() => {});
  }

  function renderChart() {
    const host = $("#chart");
    if (!host) return;
    const W = Math.max(host.clientWidth || 760, 280);
    const H = host.clientHeight || 192;
    const m = { l: 36, r: 8, t: 10, b: 22 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    if (!rows.length) {
      host.innerHTML = `<div class="empty" style="padding:52px 0;text-align:center;color:var(--dim);font:500 9px var(--mono);letter-spacing:.1em;text-transform:uppercase">No insight rows for this run.</div>`;
      return;
    }
    const n = rows.length;
    const f0 = rows[0].frame, f1 = rows[n - 1].frame;
    const xf = (f) => m.l + ((f - f0) / Math.max(f1 - f0, 1)) * iw;
    const x = (i) => xf(rows[i].frame);
    const y = (v) => m.t + (1 - clamp(v || 0, 0, 1)) * ih;
    const pts = rows.map((r, i) => `${x(i).toFixed(1)},${y(r.epv).toFixed(1)}`);

    let bands = "", segs = "", start = null;
    rows.forEach((r, i) => {
      if (r.attack_chance && start === null) start = i;
      if ((!r.attack_chance || i === n - 1) && start !== null) {
        const end = r.attack_chance ? i : i - 1;
        const seg = pts.slice(start, end + 1);
        bands += `<rect class="band" x="${x(start)}" y="${m.t}" width="${Math.max(x(end) - x(start), 2)}" height="${ih}"/>`;
        if (seg.length > 1) segs += `<path class="area" d="M${x(start)},${y(0)} L${seg.join(" L")} L${x(end)},${y(0)} Z"/><path class="line" d="M${seg.join(" L")}"/>`;
        start = null;
      }
    });

    const ticksY = [0, .25, .5, .75, 1];
    const total = (f1 - f0) / fps;
    const step = total > 60 ? 10 : total > 20 ? 5 : total > 8 ? 2 : 1;
    const ticksX = [];
    for (let s = 0; s <= total + 1e-6; s += step) ticksX.push(s);
    const commentTicks = lines.map((l) => {
      const t = l.t <= total ? l.t : null;
      if (t === null) return "";
      const px = xf(f0 + t * fps);
      return `<line class="comment-tick" x1="${px}" x2="${px}" y1="${m.t}" y2="${m.t + 7}"/>`;
    }).join("");

    host.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Point chance over time">
        <defs><linearGradient id="epvFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="#6d8bff" stop-opacity=".30"/><stop offset="1" stop-color="#6d8bff" stop-opacity="0"/>
        </linearGradient></defs>
        <g class="grid">${ticksY.map((v) => `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/>`).join("")}</g>
        <g class="axis">${ticksY.map((v) => `<text x="${m.l - 8}" y="${y(v) + 3}" text-anchor="end">${Math.round(v * 100)}%</text>`).join("")}
          ${ticksX.map((s) => `<text x="${xf(f0 + s * fps)}" y="${H - 4}" text-anchor="middle">${s}s</text>`).join("")}</g>
        ${commentTicks}
        ${bands}
        <path class="line-dim" d="M${pts.join(" L")}"/>
        ${segs}
        <line class="cross" id="cross" y1="${m.t}" y2="${m.t + ih}" visibility="hidden"/>
        <circle class="marker" id="marker" r="4" visibility="hidden"/>
        <line class="playhead" id="playhead" y1="${m.t}" y2="${m.t + ih}"/>
        <rect class="hit" x="${m.l}" y="0" width="${iw}" height="${H}"/>
      </svg><div class="tip" hidden></div>`;

    const svg = host.querySelector("svg"), tip = host.querySelector(".tip");
    const idxAt = (e) => {
      const r = svg.getBoundingClientRect();
      const px = (e.clientX - r.left) / r.width * W;
      const f = f0 + clamp((px - m.l) / iw, 0, 1) * (f1 - f0);
      let best = 0;
      rows.forEach((row, i) => { if (Math.abs(row.frame - f) < Math.abs(rows[best].frame - f)) best = i; });
      return best;
    };
    host.querySelector(".hit").addEventListener("mousemove", (e) => {
      const i = idxAt(e), r = rows[i], box = svg.getBoundingClientRect();
      const cx = x(i), cy = y(r.epv);
      const cross = host.querySelector("#cross"), marker = host.querySelector("#marker");
      cross.setAttribute("x1", cx); cross.setAttribute("x2", cx); cross.setAttribute("visibility", "visible");
      marker.setAttribute("cx", cx); marker.setAttribute("cy", cy); marker.setAttribute("visibility", "visible");
      tip.hidden = false;
      tip.style.left = `${clamp(cx / W * box.width, 90, box.width - 90)}px`;
      tip.style.top = `${cy / H * box.height}px`;
      const team = r.attack_team != null ? current?.job?.teams?.[r.attack_team] : null;
      tip.innerHTML = `<span class="tt">${clock(r.frame / fps)} · frame ${r.frame}</span>${r.attack_chance
        ? `<b>${pct(r.epv)}</b> point chance<br>${esc(team || "")} · ${esc(String(r.move || "").toLowerCase())} → zone ${r.target_zone ?? "—"}`
        : `<b>—</b> no attack chance<br>${esc(r.phase || "")}`}`;
    });
    host.querySelector(".hit").addEventListener("mouseleave", () => {
      tip.hidden = true;
      host.querySelector("#cross").setAttribute("visibility", "hidden");
      host.querySelector("#marker").setAttribute("visibility", "hidden");
    });
    host.querySelector(".hit").addEventListener("click", (e) => seek(rows[idxAt(e)].frame / fps));
    host._x = (t) => xf(clamp(f0 + t * fps, f0, f1));
    renderTable();
  }

  function renderTable() {
    const every = Math.max(1, Math.round(fps / 2));
    $("#table-wrap").innerHTML = `<table><thead><tr><th>Time</th><th>Attacking</th><th>Point chance</th><th>Block risk</th><th>Gap</th><th>Play</th><th>Zone</th></tr></thead><tbody>
      ${rows.filter((_, i) => i % every === 0).map((r) => `<tr><td>${clock(r.frame / fps)}</td><td>${esc(r.attack_team != null ? current?.job?.teams?.[r.attack_team] : "—")}</td>
        <td>${r.attack_chance ? pct(r.epv) : "—"}</td><td>${r.attack_chance ? pct(r.block_prob) : "—"}</td><td>${r.attack_chance ? `${(r.gap_m || 0).toFixed(1)} m` : "—"}</td>
        <td>${esc(r.move || "—")}</td><td>${r.target_zone ?? "—"}</td></tr>`).join("")}</tbody></table>`;
  }

  function bar(label, value, max, text, color = "") {
    const isClass = color && (color === "red" || color === "amber" || color === "green");
    const colorStyle = color && !isClass ? `background:${color};` : "";
    const classAttr = isClass ? ` ${color}` : "";
    return `<div class="bar-row"><span class="bl">${label}</span><span class="bv">${text}</span>
      <span class="track"><span class="fill${classAttr}" style="width:${clamp(value / Math.max(max, 1e-9), 0, 1) * 100}%;${colorStyle}"></span></span></div>`;
  }

  const ZONE_ORDER = [4, 3, 2, 5, 6, 1];

  function renderBreakdowns(j) {
    const attack = rows.filter((r) => r.attack_chance);
    const att = attack.length ? attack : rows;

    // 1. Recommended plays
    const moves = {};
    att.forEach((r) => { if (r.move) moves[r.move] = (moves[r.move] || 0) + 1; });
    const mv = Object.entries(moves).sort((a, b) => b[1] - a[1]).slice(0, 6);
    const movesEl = $("#moves");
    if (movesEl) {
      movesEl.innerHTML = mv.length
        ? mv.map(([k, v]) => bar(esc(k.toLowerCase().replace(/^./, (c) => c.toUpperCase())), v, mv[0][1], pct(v / att.length))).join("")
        : `<p class="muted" style="font-size:11px">No recommended plays.</p>`;
    }

    // 2. Key defenders
    const def = {};
    att.forEach((r) => (r.matchups || []).forEach((mu) => {
      const key = `${mu.defender}|${r.attack_team}`;
      (def[key] ||= { sum: 0, n: 0 }).sum += (mu.share != null ? mu.share : 0);
      def[key].n++;
    }));
    const dv = Object.entries(def).map(([k, v]) => {
      const [d, at] = k.split("|");
      const defTeam = j.teams ? (j.teams[1 - Number(at)] || "") : "";
      return { label: `#${d}${defTeam ? ` · ${defTeam}` : ""}`, v: v.sum / v.n, n: v.n };
    }).filter((d) => d.n >= fps / 2).sort((a, b) => b.v - a.v).slice(0, 6);
    const defendersEl = $("#defenders");
    if (defendersEl) {
      defendersEl.innerHTML = dv.length
        ? dv.map((d) => bar(esc(d.label), d.v, dv[0].v, pct(d.v))).join("")
        : `<p class="muted" style="font-size:11px">No matchups.</p>`;
    }

    // 3. Target zones (6-cell court with NET)
    const zones = {};
    att.forEach((r) => { if (r.target_zone) zones[r.target_zone] = (zones[r.target_zone] || 0) + 1; });
    const zmax = Math.max(1, ...Object.values(zones));
    const ztotal = Object.values(zones).reduce((a, b) => a + b, 0) || 1;
    const zonesEl = $("#zones");
    if (zonesEl) {
      zonesEl.innerHTML = ZONE_ORDER.map((z) => {
        const v = zones[z] || 0, a = 0.06 + 0.84 * (v / zmax);
        return `<div class="zone ${a > 0.55 ? "top" : ""}" style="background:rgba(45,226,190,${a.toFixed(2)})" title="Zone ${z}: ${pct(v / ztotal)}">
          <span class="zn">Zone ${z}</span><span class="zv" style="${a > 0.55 ? "color:#04241e" : ""}">${pct(v / ztotal)}</span></div>`;
      }).join("");
    }
  }

  function renderTeamShape(j) {
    const shape = {};
    rows.forEach((r) => {
      if (!r.team_shape) return;
      Object.entries(r.team_shape).forEach(([team, v]) => {
        (shape[team] ||= { coverage: 0, balance: 0, n: 0 });
        shape[team].coverage += v.coverage || 0;
        shape[team].balance += v.balance || 0;
        shape[team].n++;
      });
    });
    const teams = [0, 1].map((idx) => {
      const s = shape[idx] || shape[String(idx)];
      return s && s.n ? { idx, coverage: s.coverage / s.n, balance: s.balance / s.n } : null;
    }).filter(Boolean);
    const html = teams.length ? teams.map((t) => bar(
      `<i style="display:inline-block;width:6px;height:6px;background:${TEAM_COLOR[t.idx]};margin-right:6px"></i>${esc(j.teams?.[t.idx] ?? "")} <small>coverage</small>`,
      t.coverage, 1, pct(t.coverage), t.idx === 0 ? "" : "red"
    )).join("") : `<p class="muted" style="font-size:11px">No team-shape data.</p>`;
    const teamShapeEl = $("#team-shape");
    if (teamShapeEl) {
      teamShapeEl.innerHTML = `<div class="shape-bars">${teams.length ? teams.map((t) => `
        <section><h4><i style="background:${TEAM_COLOR[t.idx]}"></i>${esc(j.teams?.[t.idx] ?? "")}</h4>
        <div class="shape-row"><span class="sr-team">Coverage</span><span class="track"><span style="width:${(t.coverage * 100).toFixed(0)}%;background:${TEAM_COLOR[t.idx]}"></span></span><span class="sr-val">${pct(t.coverage)}</span></div>
        <div class="shape-row"><span class="sr-team">Balance</span><span class="track"><span style="width:${(t.balance * 100).toFixed(0)}%;background:rgba(242,240,233,.5)"></span></span><span class="sr-val">${pct(t.balance)}</span></div></section>`).join("") : `<p class="muted" style="font-size:11px">No team-shape data.</p>`}</div>`;
    }
    // right rail (compact coverage)
    const shapeEl = $("#shape");
    if (shapeEl) {
      shapeEl.innerHTML = `<div class="panel-label"><span>Team shape</span><span>Coverage</span></div>
        ${html}`;
    }
  }

  function renderCourtMap() {
    zoneMapData = zoneStats(rows);
    const zm = $("#zone-map");
    if (zm) zm.innerHTML = courtSvg(zoneMapData.zones, zoneMapData.total);
    const cn = $("#court-now");
    if (cn) cn.textContent = zoneMapData.total ? "Aggregate" : "—";
  }

  function updateCourtNow(zone) {
    $$("#zone-map .zone-cell").forEach((el) => el.classList.toggle("now", Number(el.dataset.zone) === Number(zone)));
    if (zone && zoneMapData.zones[zone] !== undefined) {
      $("#court-now").textContent = `Zone ${zone} · ${zoneMapData.total ? pct(zoneMapData.zones[zone] / zoneMapData.total) : "—"}`;
    } else {
      $("#court-now").textContent = "No call";
    }
  }

  function groupInfo(g, j) {
    if (!g?.label) return null;
    const side = String(g.label)[0] === "l" ? 0 : 1;
    const action = ACTION_LABEL[String(g.label).slice(2)] || String(g.label).slice(2);
    return { team: j.teams?.[side], action, score: g.score, side };
  }

  function updateReadout(t) {
    if (!rows.length || !current?.job) return;
    const frame = t * fps;
    let best = 0, bd = Infinity;
    for (let i = 0; i < rows.length; i++) {
      const d = Math.abs(rows[i].frame - frame);
      if (d < bd) { bd = d; best = i; }
    }
    const now = performance.now();
    if (best === lastReadIdx && now - lastReadAt < 350) return;
    lastReadIdx = best;
    lastReadAt = now;

    const r = rows[best];
    const j = current.job;
    const atk = r.attack_team;
    const team = atk != null ? (j.teams?.[atk] ?? "—") : "—";
    const gaInfo = groupInfo(r.group_activity, j);
    const timing = clock(r.frame / fps);
    $("#readout-clock").textContent = timing;
    $("#video-time").textContent = timing;

    const phase = cap(String(r.phase || "waiting").replace(/_/g, " "));
    const shape = atk != null ? r.team_shape?.[atk] : null;
    const cov = shape?.coverage != null ? pct(shape.coverage) : "—";
    const gaHtml = gaInfo
      ? `<span class="ga-tag"><i style="background:${gaInfo.side === 0 ? TEAM_COLOR[0] : TEAM_COLOR[1]}"></i>${esc(gaInfo.team)} · ${esc(gaInfo.action)}</span>`
      : `<span class="ga-tag">No activity label</span>`;

    if (r.attack_chance) {
      const move = cap(String(r.move || "read").toLowerCase());
      const mu = (r.matchups || []).slice(0, 2).map((m) => `<div><span>#${esc(m.defender)} · ${esc(j.teams?.[1 - atk] ?? "")}</span><b>${pct(m.share)}</b></div>`).join("");
      $("#readout").innerHTML = `
        <div class="readout-top"><span class="phase-tag">${esc(phase)}</span>${gaHtml}</div>
        <div class="readout-call"><b>${esc(team)}</b><span>→ zone ${r.target_zone ?? "—"}</span></div>
        <p class="readout-sub">${esc(move)}${r.key_player ? ` · key player #${esc(r.key_player)}` : ""}</p>
        <div class="readout-grid">
          <div><span>Point chance</span><b>${pct(r.epv)}</b></div>
          <div><span>Block risk</span><b>${pct(r.block_prob)}</b></div>
          <div><span>Gap</span><b>${(r.gap_m || 0).toFixed(1)} m</b></div>
          <div><span>Coverage</span><b>${cov}</b></div>
        </div>
        <div class="readout-move"><span>Recommended play</span><b>${esc(move)} → zone ${r.target_zone ?? "—"}</b>
          <small>${r.gap_m ? `${(r.gap_m).toFixed(1)} m gap to the target.` : ""} ${r.key_player ? `Key player #${esc(r.key_player)}.` : ""}</small></div>
        ${mu ? `<div class="readout-matchups">${mu}</div>` : ""}`;
      updateCourtNow(r.target_zone);
    } else {
      $("#readout").innerHTML = `
        <div class="readout-top"><span class="phase-tag">${esc(phase)}</span>${gaHtml}</div>
        <div class="readout-call"><b>READ</b><span>no attack chance</span></div>
        <p class="readout-sub">The model has not flagged a target call on this frame.</p>
        <div class="readout-grid">
          <div><span>Point chance</span><b>—</b></div>
          <div><span>Block risk</span><b>—</b></div>
          <div><span>Gap</span><b>—</b></div>
          <div><span>Coverage</span><b>${cov}</b></div>
        </div>
        <div class="readout-move"><span>Current state</span><b>${esc(phase)}</b>
          <small>Court mapping and tracking continue; no target is being recommended.</small></div>`;
      updateCourtNow(null);
    }
  }

  function renderRally(j) {
    const card = $("#rally-card");
    if (!card) return;
    card.hidden = !ga?.windows?.length;
    if (card.hidden) return;
    const n = Math.max(ga.frames - 1, 1);
    const rallyEl = $("#rally");
    if (rallyEl) {
      rallyEl.innerHTML = ga.windows.map((w, i) => {
        const [team, action] = splitLabel(w.label);
        return `<button class="seg" data-i="${i}" type="button"
          style="--c:${ACTION_COLOR[action] || "#8694a8"};--t:${TEAM_COLOR[team]}"
          title="${clock(w.start_s)}–${clock(w.end_s)} · ${esc(j.teams[team])} ${esc(ACTION_LABEL[action] || action)} · ${pct(w.score)}"></button>`;
      }).join("") + `<span class="rally-head" id="rally-head"></span>`;
      $$("#rally .seg").forEach((b) => b.addEventListener("click", () => seek(ga.windows[Number(b.dataset.i)].start_s)));
    }
    const legendEl = $("#rally-legend");
    if (legendEl) {
      legendEl.innerHTML = Object.entries(ACTION_COLOR)
        .map(([k, c]) => `<span><i style="background:${c}"></i>${esc(ACTION_LABEL[k] || k)}</span>`).join("")
        + j.teams.map((t, i) => `<span><i class="edge" style="background:${TEAM_COLOR[i]}"></i>${esc(t)} Side</span>`).join("");
    }
    const teamsEl = $("#rally-teams");
    if (teamsEl) {
      teamsEl.innerHTML = [0, 1].map((team) => {
        const counts = {};
        ga.windows.forEach((w) => { const [t, a] = splitLabel(w.label); if (t === team) counts[a] = (counts[a] || 0) + 1; });
        const total = Object.values(counts).reduce((a, b) => a + b, 0);
        const items = Object.keys(ACTION_COLOR).map((a) => [a, counts[a] || 0]);
        const max = Math.max(1, ...items.map((x) => x[1]));
        return `<div><h4><i style="background:${TEAM_COLOR[team]}"></i>${esc(j.teams[team])}<small class="muted">${dur(total * ga.stride / fps)} in play</small></h4>
          <div class="bars">${items.map(([a, v]) => bar(esc(ACTION_LABEL[a] || cap(a)), v, max,
            total ? pct(v / total) : "—", ACTION_COLOR[a])).join("")}</div></div>`;
      }).join("");
    }
    lastWin = -1;

    syncRally(0);
  }

  function splitLabel(label) {
    return [String(label)[0] === "l" ? 0 : 1, String(label).slice(2)];
  }

  function syncRally(t) {
    if (!ga?.windows?.length) return;
    const f = t * fps;
    const head = $("#rally-head");
    if (head) head.style.left = `${clamp(f / Math.max(ga.frames - 1, 1), 0, 1) * 100}%`;
    let i = ga.windows.findIndex((w) => f >= w.start_frame && f <= w.end_frame + 1);
    if (i < 0) i = ga.windows.length - 1;
    if (i === lastWin) return;
    lastWin = i;
    $$("#rally .seg").forEach((b, k) => b.classList.toggle("now", k === i));
    const w = ga.windows[i], [team, action] = splitLabel(w.label);
    const gaNow = $("#ga-now");
    if (gaNow) {
      gaNow.innerHTML = `<span class="ga-team" style="border-color:${TEAM_COLOR[team]};color:${TEAM_COLOR[team]}">${esc(current?.job?.teams?.[team] ?? "")}</span>
        <b style="color:${ACTION_COLOR[action] || "var(--paper)"}">${esc(ACTION_LABEL[action] || action)}</b><span class="muted">${pct(w.score)}</span>`;
    }
  }

  function renderSnaps(j) {
    const card = $("#snaps-card");
    const snaps = j.snapshots || [];
    if (card) card.hidden = !snaps.length;
    const snapsEl = $("#snaps");
    if (snapsEl) {
      snapsEl.innerHTML = snaps.length
        ? snaps.map((s) => `<img loading="lazy" src="/files/${j.id}/dashboard/${encodeURIComponent(s)}" alt="Rendered dashboard snapshot ${esc(s)}">`).join("")
        : `<p class="muted" style="font-size:11px">No snapshots were rendered.</p>`;
      $$("#snaps img").forEach((img) => img.addEventListener("click", () => {
        $("#lightbox img").src = img.src;
        $("#lightbox").hidden = false;
      }));
    }
  }

  function renderFrameTable(j) {
    const every = Math.max(1, Math.round(fps / 2));
    const sampled = rows.filter((_, i) => i % every === 0);
    const count = $("#frame-count");
    if (count) count.textContent = sampled.length ? `${sampled.length} rows · every ${every} frames` : "No rows";
    const ft = $("#frame-table");
    if (ft) {
      ft.innerHTML = sampled.length ? `<table><thead><tr><th>Time</th><th>Attacking</th><th>Zone</th><th>Point chance</th><th>Block risk</th><th>Gap</th><th>Play</th><th>Phase</th></tr></thead><tbody>
        ${sampled.map((r) => `<tr><td>${clock(r.frame / fps)}</td><td>${esc(r.attack_team != null ? j.teams?.[r.attack_team] : "—")}</td>
          <td>${r.target_zone ?? "—"}</td><td>${r.attack_chance ? pct(r.epv) : "—"}</td><td>${r.attack_chance ? pct(r.block_prob) : "—"}</td>
          <td>${r.attack_chance ? `${(r.gap_m || 0).toFixed(1)} m` : "—"}</td><td>${esc(r.move || "—")}</td><td>${esc(r.phase || "—")}</td></tr>`).join("")}</tbody></table>`
        : `<p class="muted" style="padding:18px;font-size:11px">No frame data.</p>`;
    }
  }

  let lastNowAt = 0;
  function syncToVideo() {
    cancelAnimationFrame(raf);
    const player = $("#player");
    const tick = () => {
      if (view !== "job" || $("#results").hidden) return;
      const t = player.currentTime || 0;
      const host = $("#chart");
      if (host._x) {
        const px = host._x(t);
        const ph = $("#playhead");
        ph?.setAttribute("x1", px);
        ph?.setAttribute("x2", px);
      }
      if (performance.now() - lastNowAt > 90) {
        lastNowAt = performance.now();
        updateReadout(t);
      }
      syncRally(t);
      const now = stateAt(t);
      if (now !== lastNow) {
        lastNow = now;
        $$("#feed li[data-i]").forEach((li) => {
          const i = Number(li.dataset.i);
          li.classList.toggle("past", i < now);
          li.classList.toggle("now", i === now);
        });
        const el = $(`#feed li[data-i="${now}"]`);
        if (el && !feedHover) {
          const feed = $("#feed");
          if (feed) feed.scrollTop = el.offsetTop - feed.clientHeight / 2 + el.clientHeight / 2;
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
  }

  function stateAt(t) {
    let now = -1;
    lines.forEach((l, i) => { if (l.t <= t + 0.05) now = i; });
    return now;
  }

  /* ---------------------------------------------------------------- archive */
  function renderArchive() {
    const filtered = jobs.filter((j) => archiveFilter === "all" || (archiveFilter === "running" ? isActive(j) : j.status === archiveFilter));
    const list = $("#archive-list");
    list.innerHTML = filtered.map((j, i) => runRow(j, i)).join("");
    $("#archive-empty").hidden = filtered.length > 0;

    // Attach row delete handlers
    list.querySelectorAll(".run-del-btn").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.preventDefault();
        e.stopPropagation();
        const jid = btn.dataset.del;
        const targetJob = jobs.find((x) => x.id === jid);
        const name = targetJob?.title || jid;
        if (!confirm(`Delete run "${name}"?\nThis will delete all its generated video, commentary and data.`)) return;
        try {
          await api(`/api/jobs/${encodeURIComponent(jid)}`, { method: "DELETE" });
          toast("Run deleted from archive");
          loadJobs();
        } catch (err) {
          toast(err.message, true);
        }
      });
    });
  }

  function runRow(j, idx) {
    const t = thumbOf(j);
    const active = isActive(j);
    return `<a class="run-row" href="#/job/${encodeURIComponent(j.id)}">
      <span class="run-thumb ${t ? "" : "blank"}"${t ? ` style="background-image:url('${esc(t)}')"` : ""}>
        ${t ? "" : `<svg viewBox="0 0 24 24"><rect x="2.5" y="5.5" width="19" height="13"/><path d="m10 9 5 3-5 3z"/></svg>`}
        <span class="run-index">${String(idx + 1).padStart(2, "0")}</span>
      </span>
      <span class="run-main"><b>${esc(jobTitle(j))}</b><small>${esc(j.teams?.[0] ?? "—")} vs ${esc(j.teams?.[1] ?? "—")} · ${esc(ago(j.created))}</small></span>
      <span class="run-mode">${esc(modeLabel(j))}</span>
      <span class="run-frames">${j.frames ? `${j.frames} frames` : "—"}</span>
      <span class="run-created">${esc(new Date(j.created * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" }))}</span>
      <span class="run-status ${esc(j.status)}">${esc(statusText(j))}</span>
      <span class="run-actions-cell">
        ${!active ? `<button type="button" class="run-del-btn" data-del="${esc(j.id)}" title="Delete run ${esc(j.id)}">✕</button>` : ""}
        <span class="run-open">↗</span>
      </span>
      ${active ? `<span class="run-progress"><span style="width:${Math.round((j.progress || 0) * 100)}%"></span></span>` : ""}
    </a>`;
  }

  /* ---------------------------------------------------------------- tabs */
  function switchTab(name) {
    $$("#intel-tabs button").forEach((b) => {
      const on = b.dataset.tab === name;
      b.classList.toggle("on", on);
      b.setAttribute("aria-selected", on ? "true" : "false");
    });
    $$(".tab-panel").forEach((p) => (p.hidden = p.dataset.panel !== name));
  }

  /* ---------------------------------------------------------------- router */
  let currentView = "landing";
  let pendingScroll = null;

  function parseHash() {
    const h = location.hash || "#/";
    const raw = h.replace(/^#\/?/, "");
    const [path, query] = raw.split("?");
    const parts = path.split("/").filter(Boolean);
    if (parts[0] === "analyze") return { view: "analyze" };
    if (parts[0] === "runs") return { view: "runs" };
    if (parts[0] === "job" && parts[1]) return { view: "job", id: decodeURIComponent(parts[1]), query };
    return { view: "landing" };
  }

  function route() {
    const r = parseHash();
    currentView = r.view;
    view = r.view;
    $$(".view").forEach((v) => (v.hidden = v.id !== `view-${r.view}`));
    document.body.dataset.view = r.view;
    closeMenu();
    if (r.view !== "job") stopJobView();
    $$("[data-nav]").forEach((a) => a.classList.toggle("on", a.dataset.nav === (r.view === "runs" || r.view === "job" ? "runs" : "")));
    if (r.view === "landing") {
      loadJobs();
      setNavForScroll();
    } else if (r.view === "runs") {
      loadJobs();
    } else if (r.view === "analyze") {
      refreshForm();
    } else if (r.view === "job") {
      openJob(r.id);
    }
    window.scrollTo({ top: 0, behavior: "instant" });
    if (pendingScroll) {
      const target = pendingScroll;
      pendingScroll = null;
      setTimeout(() => scrollToSection(target), 80);
    }
    onScroll();
  }

  function scrollToSection(id) {
    const el = document.getElementById(id);
    if (!el) return;
    const top = el.getBoundingClientRect().top + window.scrollY - 84;
    window.scrollTo({ top, behavior: "smooth" });
  }

  function onScroll() {
    const head = $("#site-head");
    head.classList.toggle("scrolled", window.scrollY > 8);
    if (currentView === "landing") setNavForScroll();
  }

  let spySections = [];
  function setNavForScroll() {
    if (currentView !== "landing") return;
    if (!spySections.length) {
      spySections = ["platform", "capabilities", "method"].map((id) => document.getElementById(id)).filter(Boolean);
    }
    const mark = window.scrollY + window.innerHeight * 0.32;
    let active = "";
    spySections.forEach((s) => { if (s.offsetTop <= mark) active = s.id; });
    $$("[data-nav]").forEach((a) => {
      if (a.dataset.nav === "runs") return;
      a.classList.toggle("on", a.dataset.nav === active);
    });
  }

  function closeMenu() {
    $("#head-nav").classList.remove("open");
    $("#menu-btn").setAttribute("aria-expanded", "false");
    document.body.classList.remove("nav-open");
  }

  /* ---------------------------------------------------------------- init */
  function init() {
    $("#foot-year").textContent = new Date().getFullYear();

    // header
    $("#menu-btn").addEventListener("click", () => {
      const open = !$("#head-nav").classList.contains("open");
      $("#head-nav").classList.toggle("open", open);
      $("#menu-btn").setAttribute("aria-expanded", String(open));
      document.body.classList.toggle("nav-open", open);
    });
    document.addEventListener("click", (e) => {
      const link = e.target.closest("[data-scroll]");
      if (link) {
        e.preventDefault();
        const id = link.dataset.scroll;
        closeMenu();
        if (currentView !== "landing") {
          pendingScroll = id;
          if (location.hash === "#/" || location.hash === "") route();
          else location.hash = "#/";
        } else {
          scrollToSection(id);
        }
        return;
      }
      if (!e.target.closest("#head-nav") && !e.target.closest("#menu-btn")) closeMenu();
    });
    window.addEventListener("scroll", onScroll, { passive: true });

    // intake
    initIntake();

    // filters
    $("#archive-filters").addEventListener("click", (e) => {
      const b = e.target.closest("[data-filter]");
      if (!b) return;
      archiveFilter = b.dataset.filter;
      $$("#archive-filters button").forEach((x) => x.classList.toggle("on", x === b));
      renderArchive();
    });

    // Clear all archive button
    $("#btn-clear-archive")?.addEventListener("click", async () => {
      const removable = jobs.filter((j) => !isActive(j));
      if (!removable.length) {
        toast("No completed or failed runs to clear.");
        return;
      }
      if (!confirm(`Clear all ${removable.length} runs from the archive cache?\nAll video files and analysis data will be removed. Active runs will be preserved.`)) return;
      try {
        const res = await api("/api/jobs", { method: "DELETE" });
        toast(`Cleared ${res?.deleted ?? removable.length} runs from archive`);
        loadJobs();
      } catch (err) {
        toast(err.message, true);
      }
    });

    // tabs
    $("#intel-tabs").addEventListener("click", (e) => {
      const b = e.target.closest("[data-tab]");
      if (!b || b.hidden) return;
      switchTab(b.dataset.tab);
      if (b.dataset.tab === "frames" || b.dataset.tab === "matchups") {
        // tables already built; no lazy work required
      }
    });

    // feed filters
    $("#feed-filters").addEventListener("click", (e) => {
      const b = e.target.closest("[data-kind]");
      if (!b) return;
      feedKind = b.dataset.kind;
      $$("#feed-filters button").forEach((x) => x.classList.toggle("on", x === b));
      renderFeed();
    });
    $("#feed").addEventListener("mouseenter", () => (feedHover = true));
    $("#feed").addEventListener("mouseleave", () => (feedHover = false));

    // table toggle
    $("#table-toggle").addEventListener("click", (e) => {
      const w = $("#table-wrap");
      w.hidden = !w.hidden;
      e.currentTarget.textContent = w.hidden ? "Frame table" : "Hide table";
    });

    // lightbox
    $("#lightbox").addEventListener("click", (e) => {
      if (e.target.id === "lightbox" || e.target.closest(".lightbox-close")) $("#lightbox").hidden = true;
    });
    $("#lightbox img").addEventListener("click", (e) => e.stopPropagation());
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") { $("#lightbox").hidden = true; closeMenu(); } });

    // showcase media
    const showVideo = $("#showcase-video");
    showVideo.addEventListener("timeupdate", () => {
      const total = Number.isFinite(showVideo.duration) && showVideo.duration > 0 ? clock(showVideo.duration) : "—";
      $("#showcase-timecode").textContent = `${clock(showVideo.currentTime || 0)} / ${total}`;
    });

    // resize chart
    let resizeTimer;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => { if (view === "job" && !$("#results").hidden && rows.length) renderChart(); }, 160);
    });

    window.addEventListener("hashchange", route);
    loadSystem();
    refreshForm();
    route();
    setInterval(() => { if ((view === "landing" || view === "runs") && !jobsLoading) loadJobs(); }, 5000);
  }

  init();
})();
