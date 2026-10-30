/* GRITSENSE web app front end - no build step, no dependencies. */
(() => {
  "use strict";

  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v, d = 0) => `${(v * 100).toFixed(d)}%`;
  const clock = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${(s % 60).toFixed(1).padStart(4, "0")}`;
  const dur = (s) => (s < 60 ? `${Math.round(s)}s` : `${Math.floor(s / 60)}m ${String(Math.round(s % 60)).padStart(2, "0")}s`);
  const ago = (t) => {
    const s = Date.now() / 1000 - t;
    if (s < 60) return "just now";
    if (s < 3600) return `${Math.floor(s / 60)} min ago`;
    if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
    return new Date(t * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" });
  };
  const ZONE_ORDER = [4, 3, 2, 5, 6, 1]; // net at the top: front row 4-3-2, back row 5-6-1
  const STAGES = {
    full: [["prepare", "Prepare video"], ["detect", "Detect & track"], ["ball", "Ball track"],
           ["graphs", "Graph models"], ["dashboard", "Render dashboard"], ["finalize", "Finalize"]],
    demo: [["prepare", "Load sample"], ["dashboard", "Render dashboard"], ["finalize", "Finalize"]],
  };
  const STAGE_TEXT = {
    queued: "Waiting in queue", prepare: "Preparing video", detect: "Detecting players, pose and ball",
    ball: "Re-detecting and cleaning the ball track", graphs: "Building graphs · GAT → HRN · group-activity GAT",
    dashboard: "Rendering tactical dashboard and commentary", finalize: "Encoding for the browser",
  };

  async function api(path, opts) {
    const res = await fetch(path, opts);
    if (!res.ok) {
      const text = await res.text();
      const msg = (text.match(/<p>(.*?)<\/p>/s) || [])[1] || text || res.statusText;
      throw new Error(msg.replace(/<[^>]+>/g, "").trim());
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
    toastTimer = setTimeout(() => (el.hidden = true), 4200);
  }

  /* ---------------- System status ---------------- */
  async function loadSystem() {
    try {
      const s = await api("/api/system");
      const model = s.model.split("/").pop().replace("-Instruct", "");
      $("#sys").innerHTML = `
        <span class="chip"><span class="dot ${s.gpu ? "ok" : "warn"}"></span>${s.gpu ? `<b>GPU</b> ${esc(s.gpu)}` : "<b>CPU</b> mode"}</span>
        <span class="chip" title="${s.model_cached ? "Model cached - runs offline" : "Downloads once on first run"}">
          <span class="dot ${s.model_cached ? "ok" : "warn"}"></span><b>${esc(model)}</b> ${s.model_cached ? "offline" : "not cached"}</span>`;
      if (!s.sample) $('[data-mode="demo"]').disabled = true;
    } catch {
      $("#sys").innerHTML = `<span class="chip"><span class="dot"></span>Server offline</span>`;
    }
  }

  /* ---------------- Runs ---------------- */
  let jobs = [];
  function thumb(j) {
    if (j.status === "done" && j.snapshots?.length) {
      return `<span class="run-thumb" style="background-image:url('/files/${j.id}/dashboard/${j.snapshots[0]}')"></span>`;
    }
    return `<span class="run-thumb"><svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="m10 9 5 3-5 3z"/></svg></span>`;
  }
  function runItem(j) {
    const label = j.status === "running" ? `${Math.round(j.progress * 100)}%` : j.status;
    return `<li><a href="#/job/${j.id}">${thumb(j)}
      <span class="run-main"><b>${esc(j.title || j.id)}</b>
      <small>${esc(j.teams[0])} vs ${esc(j.teams[1])} · ${j.mode === "demo" ? "sample" : "full pipeline"} · ${ago(j.created)}</small></span>
      <span class="status ${j.status}">${label}</span></a></li>`;
  }
  async function loadJobs() {
    try { jobs = await api("/api/jobs"); } catch { return; }
    $("#runs-count").textContent = jobs.length;
    const empty = `<li class="empty">No runs yet. Your analyses will appear here.</li>`;
    $("#recent-list").innerHTML = jobs.slice(0, 6).map(runItem).join("") || empty;
    $("#all-runs").innerHTML = jobs.map(runItem).join("") || empty;
  }

  /* ---------------- New analysis form ---------------- */
  const form = { mode: "full", nlp: "local", len: "full", file: null, vw: 0, vh: 0 };
  // Only reads the video's size and length; court corners come from annotations.json.
  const probeVideo = document.createElement("video");
  probeVideo.muted = true;
  probeVideo.playsInline = true;
  probeVideo.preload = "metadata";

  function seg(id, key, cb) {
    $$(`#${id} button`).forEach((b) => b.addEventListener("click", () => {
      $$(`#${id} button`).forEach((x) => x.classList.toggle("on", x === b));
      form[key] = b.dataset[key];
      cb?.();
      refreshForm();
    }));
  }
  seg("mode-seg", "mode", () => {
    $$("[data-show]").forEach((el) => (el.hidden = el.dataset.show !== form.mode));
    $("#settings-num").textContent = form.mode === "full" ? "2" : "1";
  });
  seg("nlp-seg", "nlp");
  seg("len-seg", "len", () => ($("#frames-wrap").hidden = form.len !== "preview"));

  function refreshForm() {
    const btn = $("#start-btn");
    const hint = $("#submit-hint");
    if (form.mode === "demo") {
      btn.disabled = false;
      hint.textContent = "Renders the dashboard from the bundled sample detections.";
      return;
    }
    btn.disabled = !form.file;
    hint.textContent = !form.file ? "Choose a video to begin."
      : "Ready. Court corners come from the project's annotations.json. Full pipeline: detection, ball tracking, graph models, then the dashboard.";
  }

  function setFile(file) {
    if (!file) return;
    if (!file.type.startsWith("video/") && !/\.(mp4|mov|avi|mkv|m4v|webm)$/i.test(file.name)) {
      toast("That doesn't look like a video file.", true);
      return;
    }
    form.file = file;
    $("#drop").classList.add("has-file");
    $(".drop-empty").hidden = true;
    $(".drop-file").hidden = false;
    $("#file-name").textContent = file.name;
    $("#file-meta").textContent = `${(file.size / 1048576).toFixed(1)} MB · reading…`;
    URL.revokeObjectURL(probeVideo.src);
    probeVideo.src = URL.createObjectURL(file);
    refreshForm();
    if (!$("#title").value) $("#title").placeholder = file.name.replace(/\.[^.]+$/, "");
  }

  probeVideo.addEventListener("loadedmetadata", () => {
    form.vw = probeVideo.videoWidth;
    form.vh = probeVideo.videoHeight;
    $("#file-meta").textContent = `${(form.file.size / 1048576).toFixed(1)} MB · ${form.vw}×${form.vh} · ${dur(probeVideo.duration)}`
      + (form.vh !== 1080 ? " · will be scaled to 1080p" : "");
  });
  probeVideo.addEventListener("error", () => {
    $("#file-meta").textContent = `${(form.file.size / 1048576).toFixed(1)} MB · preview unavailable`;
  });

  const drop = $("#drop");
  $("#video-input").addEventListener("change", (e) => setFile(e.target.files[0]));
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
  drop.addEventListener("click", (e) => { if (form.file) e.preventDefault(); });
  $("#file-clear").addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    $("#video-input").value = "";
    $("#video-input").click();
  });

  $("#setup-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = $("#start-btn");
    const fd = new FormData();
    fd.append("mode", form.mode);
    fd.append("title", $("#title").value || $("#title").placeholder.replace(/^e\.g\. .*/, ""));
    fd.append("team_left", $("#team-left").value);
    fd.append("team_right", $("#team-right").value);
    fd.append("nlp", form.nlp);
    if (form.len === "preview") fd.append("max_frames", $("#max-frames").value);
    if (form.mode === "full") {
      fd.append("video", form.file);
    }
    btn.disabled = true;
    btn.querySelector("span").textContent = form.mode === "full" ? "Uploading…" : "Starting…";
    try {
      const job = await api("/api/jobs", { method: "POST", body: fd });
      location.hash = `#/job/${job.id}`;
      loadJobs();
    } catch (err) {
      toast(err.message, true);
    } finally {
      btn.querySelector("span").textContent = "Run analysis";
      refreshForm();
    }
  });

  /* ---------------- Job page ---------------- */
  let pollTimer = null;
  let current = null; // { id, job, loaded }

  async function openJob(id) {
    clearTimeout(pollTimer);
    if (current?.id !== id) {
      current = { id, job: null, loaded: false };
      $("#proc").hidden = $("#results").hidden = $("#failed").hidden = true;
      $("#job-title").textContent = "Loading…";
      $("#job-meta").innerHTML = "";
      $("#job-actions").innerHTML = "";
    }
    let job;
    try { job = await api(`/api/jobs/${id}`); } catch {
      $("#job-title").textContent = "Run not found";
      return;
    }
    if (current?.id !== id) return;
    current.job = job;
    renderJobHead(job);
    if (job.status === "done") {
      $("#proc").hidden = true;
      if (!current.loaded) { current.loaded = true; await renderResults(job); loadJobs(); }
    } else if (job.status === "failed") {
      $("#proc").hidden = true;
      $("#failed").hidden = false;
      $("#err-msg").textContent = job.error || "Unknown error";
      $("#err-log").textContent = (job.log || []).join("\n");
      loadJobs();
    } else {
      $("#proc").hidden = false;
      renderProgress(job);
      pollTimer = setTimeout(() => location.hash === `#/job/${id}` && openJob(id), 1500);
    }
  }

  function renderJobHead(j) {
    $("#job-title").textContent = j.title || j.id;
    const bits = [
      `<span class="teams-tag"><span class="sw t0" style="width:8px;height:8px;border-radius:2px;display:inline-block"></span><b>${esc(j.teams[0])}</b> vs <b>${esc(j.teams[1])}</b><span class="sw t1" style="width:8px;height:8px;border-radius:2px;display:inline-block"></span></span>`,
      `<span>${j.mode === "demo" ? "Sample match" : "Full pipeline"}</span>`,
      `<span>${j.nlp === "local" ? "Offline LLM commentary" : "Template commentary"}</span>`,
      `<span>${new Date(j.created * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}</span>`,
    ];
    if (j.finished && j.started) bits.push(`<span>Processed in ${dur(j.finished - j.started)}</span>`);
    $("#job-meta").innerHTML = bits.join("");
    const acts = [];
    if (j.status === "done") {
      acts.push(`<a class="btn ghost sm" href="/files/${j.id}/dashboard.mp4?download=1">Download video</a>`,
                `<a class="btn ghost sm" href="/files/${j.id}/dashboard/commentary.txt?download=1">Commentary</a>`,
                `<a class="btn ghost sm" href="/files/${j.id}/dashboard/dashboard_insights.json?download=1">Insights JSON</a>`);
    }
    if (j.status === "done" || j.status === "failed") acts.push(`<button class="btn ghost sm danger" id="del-btn">Delete</button>`);
    else acts.push(`<button class="btn ghost sm danger" id="cancel-btn">Cancel run</button>`);
    const html = acts.join("");
    if ($("#job-actions").dataset.html !== html) {
      $("#job-actions").innerHTML = html;
      $("#job-actions").dataset.html = html;
      $("#cancel-btn")?.addEventListener("click", async () => {
        if (!confirm("Stop this run?")) return;
        try { await api(`/api/jobs/${j.id}/cancel`, { method: "POST" }); toast("Run cancelled"); openJob(j.id); }
        catch (err) { toast(err.message, true); }
      });
      $("#del-btn")?.addEventListener("click", async () => {
        if (!confirm("Delete this run and its files?")) return;
        try { await api(`/api/jobs/${j.id}`, { method: "DELETE" }); location.hash = "#/runs"; loadJobs(); }
        catch (err) { toast(err.message, true); }
      });
    }
  }

  function renderProgress(j) {
    const p = Math.max(0, Math.min(1, j.progress || 0));
    $("#pct").textContent = Math.floor(p * 100);
    $("#ring-fg").style.strokeDashoffset = String(326.7 * (1 - p));
    $("#proc-status").textContent = j.status === "queued" ? "Queued" : "Processing";
    $("#proc-stage").textContent = STAGE_TEXT[j.stage] || "Working";
    if (j.started) {
      const el = Date.now() / 1000 - j.started;
      const eta = p > 0.06 ? ` · about ${dur(el / p * (1 - p))} left` : "";
      $("#proc-times").textContent = `Elapsed ${dur(el)}${eta}${j.engine ? ` · commentary: ${j.engine}` : ""}`;
    } else {
      $("#proc-times").textContent = "Another run is in progress. This one starts automatically.";
    }
    const stages = STAGES[j.mode];
    const idx = stages.findIndex(([k]) => k === j.stage);
    $("#stepper").innerHTML = stages.map(([, label], i) =>
      `<li class="${i < idx ? "done" : i === idx ? "active" : ""}"><span class="si"></span>${label}</li>`).join("");
    const log = $("#log");
    const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
    log.textContent = (j.log || []).join("\n");
    if (atBottom) log.scrollTop = log.scrollHeight;
  }

  /* ---------------- Results ---------------- */
  let rows = [], lines = [], fps = 30, raf = 0, ga = null;
  const ACTION_COLOR = { pass: "#60a5fa", set: "#2de2be", spike: "#fbbf24", winpoint: "#f43f5e" };
  const TEAM_COLOR = ["#facc15", "#ef4444"];
  const splitLabel = (l) => [l[0] === "l" ? 0 : 1, l.slice(2)]; // "l-spike" -> [0, "spike"]

  async function renderResults(j) {
    $("#results").hidden = false;
    fps = j.fps || 30;
    const player = $("#player");
    player.src = `/files/${j.id}/dashboard.mp4`;
    [rows, lines, ga] = await Promise.all([
      api(`/api/jobs/${j.id}/insights`).catch(() => []),
      api(`/files/${j.id}/commentary.json`).catch(() => []),
      api(`/files/${j.id}/dashboard/group_activity.json`).catch(() => null),
    ]);
    renderRally(j);
    $("#engine-tag").textContent = j.engine ? `by ${j.engine}` : "";
    renderStats();
    renderFeed();
    renderChart();
    renderBreakdowns(j);
    renderSnaps(j);
    syncToVideo();
  }

  function attackRows() {
    const a = rows.filter((r) => r.attack_chance);
    return a.length ? a : rows;
  }

  function renderStats() {
    const att = rows.filter((r) => r.attack_chance);
    let chances = 0;
    rows.forEach((r, i) => { if (r.attack_chance && !rows[i - 1]?.attack_chance) chances++; });
    const peak = att.length ? att.reduce((m, r) => (r.epv > m.epv ? r : m)) : null;
    const avg = att.length ? att.reduce((s, r) => s + r.epv, 0) / att.length : 0;
    const swaps = rows.filter((r, i) => i && r.attack_team != null && rows[i - 1].attack_team != null && r.attack_team !== rows[i - 1].attack_team).length;
    const tiles = [
      ["Footage analysed", dur(rows.length / fps), `${rows.length} frames`],
      ["Attack chances", chances, `${dur(att.length / fps)} in total`],
      ["Peak point chance", peak ? pct(peak.epv) : "—", peak ? `at ${clock(peak.frame / fps)} · Zone ${peak.target_zone ?? "—"}` : "no attack chances", "accent"],
      ["Avg point chance", att.length ? pct(avg) : "—", "during attack chances"],
      ["Commentary", lines.length, `${swaps} possession changes`],
    ];
    $("#stats").innerHTML = tiles.map(([k, v, s, c]) => `<div class="stat ${c || ""}"><small>${k}</small><b>${v}</b><span>${s}</span></div>`).join("");
  }

  function renderFeed() {
    $("#feed").innerHTML = lines.map((l, i) => `
      <li class="k-${esc(l.kind)}" data-i="${i}">
        <div class="fh"><span class="ft">${clock(l.t)}</span><span class="fk">${esc(l.kind)}</span><span class="fs">${esc(l.source)}</span></div>
        <p>${esc(l.text)}</p>
      </li>`).join("") || `<li class="empty">No commentary lines.</li>`;
    $$("#feed li[data-i]").forEach((li) => li.addEventListener("click", () => seek(lines[li.dataset.i].t)));
  }

  function seek(t) {
    const p = $("#player");
    p.currentTime = Math.max(0, t - 0.05);
    if (p.paused) p.play().catch(() => {});
  }

  /* point-chance chart (single series: no legend; title names it) */
  function renderChart() {
    const host = $("#chart");
    const W = host.clientWidth || 800, H = host.clientHeight || 180;
    const m = { l: 36, r: 8, t: 8, b: 22 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    if (!rows.length) { host.innerHTML = `<div class="empty">No insights available.</div>`; return; }
    const n = rows.length;
    const f0 = rows[0].frame, f1 = rows[n - 1].frame;
    // x is the video frame, so gaps (frames without a court homography) stay in place.
    const xf = (f) => m.l + ((f - f0) / Math.max(f1 - f0, 1)) * iw;
    const x = (i) => xf(rows[i].frame);
    const y = (v) => m.t + (1 - v) * ih;
    const pts = rows.map((r, i) => `${x(i).toFixed(1)},${y(r.epv || 0).toFixed(1)}`);
    // Bright line + fill only during attack chances (when the dashboard shows "where to hit").
    let bands = "", segs = "", start = null;
    rows.forEach((r, i) => {
      if (r.attack_chance && start === null) start = i;
      if ((!r.attack_chance || i === n - 1) && start !== null) {
        const end = r.attack_chance ? i : i - 1;
        const seg = pts.slice(start, end + 1);
        bands += `<rect class="band" x="${x(start)}" y="${m.t}" width="${Math.max(x(end) - x(start), 2)}" height="${ih}" rx="3"/>`;
        segs += `<path class="area" d="M${x(start)},${y(0)} L${seg.join(" L")} L${x(end)},${y(0)} Z"/><path class="line" d="M${seg.join(" L")}"/>`;
        start = null;
      }
    });
    const ticksY = [0, 0.25, 0.5, 0.75, 1];
    const total = (f1 - f0) / fps;
    const step = total > 60 ? 10 : total > 20 ? 5 : total > 8 ? 2 : 1;
    const ticksX = [];
    for (let s = 0; s <= total + 1e-6; s += step) ticksX.push(s);
    host.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Point chance over time">
        <defs><linearGradient id="epvFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="#2de2be" stop-opacity=".32"/><stop offset="1" stop-color="#2de2be" stop-opacity="0"/></linearGradient></defs>
        <g class="grid">${ticksY.map((v) => `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/>`).join("")}</g>
        <g class="axis">${ticksY.map((v) => `<text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text>`).join("")}
          ${ticksX.map((s) => `<text x="${xf(f0 + s * fps)}" y="${H - 4}" text-anchor="middle">${s}s</text>`).join("")}</g>
        ${bands}
        <path class="line-dim" d="M${pts.join(" L")}"/>
        ${segs}
        <line class="cross" id="cross" y1="${m.t}" y2="${m.t + ih}" visibility="hidden"/>
        <circle class="marker" id="marker" r="5" visibility="hidden"/>
        <line class="playhead" id="playhead" y1="${m.t}" y2="${m.t + ih}"/>
        <rect class="hit" x="${m.l}" y="0" width="${iw}" height="${H}"/>
      </svg><div class="tip" hidden></div>`;
    const svg = $("svg", host), tip = $(".tip", host);
    const idxAt = (e) => {
      const r = svg.getBoundingClientRect();
      const px = (e.clientX - r.left) / r.width * W;
      const f = f0 + Math.max(0, Math.min(1, (px - m.l) / iw)) * (f1 - f0);
      let best = 0;
      rows.forEach((r, i) => { if (Math.abs(r.frame - f) < Math.abs(rows[best].frame - f)) best = i; });
      return best;
    };
    $(".hit", host).addEventListener("mousemove", (e) => {
      const i = idxAt(e), r = rows[i], r0 = svg.getBoundingClientRect();
      const cx = x(i), cy = y(r.epv || 0);
      $("#cross").setAttribute("x1", cx); $("#cross").setAttribute("x2", cx); $("#cross").setAttribute("visibility", "visible");
      $("#marker").setAttribute("cx", cx); $("#marker").setAttribute("cy", cy); $("#marker").setAttribute("visibility", "visible");
      tip.hidden = false;
      tip.style.left = `${Math.max(90, Math.min(r0.width - 90, cx / W * r0.width))}px`;
      tip.style.top = `${cy / H * r0.height}px`;
      const team = current.job.teams[r.attack_team] ?? "—";
      tip.innerHTML = `<span class="tt">${clock(r.frame / fps)} · frame ${r.frame}</span>${r.attack_chance ? `<b>${pct(r.epv || 0)}</b> point chance<br>` : `<b>—</b> no attack chance<br>`}
        ${r.attack_chance ? `${esc(team)} · ${esc((r.move || "").toLowerCase())} → Zone ${r.target_zone ?? "—"}` : esc(r.phase || "")}`;
    });
    $(".hit", host).addEventListener("mouseleave", () => {
      tip.hidden = true;
      $("#cross").setAttribute("visibility", "hidden");
      $("#marker").setAttribute("visibility", "hidden");
    });
    $(".hit", host).addEventListener("click", (e) => seek(rows[idxAt(e)].frame / fps));
    host._x = (t) => xf(Math.min(f0 + t * fps, f1));
    renderTable();
  }

  function renderTable() {
    const every = Math.max(1, Math.round(fps / 2));
    $("#table-wrap").innerHTML = `<table><thead><tr><th>Time</th><th>Attacking</th><th>Point chance</th><th>Block risk</th><th>Play</th><th>Zone</th></tr></thead><tbody>
      ${rows.filter((_, i) => i % every === 0).map((r) => `<tr><td>${clock(r.frame / fps)}</td><td>${esc(current.job.teams[r.attack_team] ?? "—")}</td>
        <td>${pct(r.epv || 0)}</td><td>${pct(r.block_prob || 0)}</td><td>${esc(r.move || "")}</td><td>${r.target_zone ?? "—"}</td></tr>`).join("")}</tbody></table>`;
  }
  $("#table-toggle").addEventListener("click", (e) => {
    const w = $("#table-wrap");
    w.hidden = !w.hidden;
    e.target.textContent = w.hidden ? "Show table" : "Hide table";
  });

  function bar(label, value, max, text, color) {
    return `<div class="bar-row"><span class="bl">${label}</span><span class="bv">${text}</span>
      <div class="track"><div class="fill" style="width:${(value / max) * 100}%${color ? `;background:${color}` : ""}"></div></div></div>`;
  }

  function renderBreakdowns(j) {
    const att = attackRows();
    const moves = {};
    att.forEach((r) => { if (r.move) moves[r.move] = (moves[r.move] || 0) + 1; });
    const mv = Object.entries(moves).sort((a, b) => b[1] - a[1]).slice(0, 6);
    $("#moves").innerHTML = mv.map(([k, v]) => bar(esc(k.toLowerCase().replace(/^./, (c) => c.toUpperCase())), v, mv[0][1], pct(v / att.length)))
      .join("") || `<div class="empty">No recommendations.</div>`;

    const def = {};
    att.forEach((r) => (r.matchups || []).forEach((mu) => {
      const key = `${mu.defender}|${r.attack_team}`;
      (def[key] ||= { sum: 0, n: 0 }).sum += mu.share;
      def[key].n++;
    }));
    const dv = Object.entries(def).map(([k, v]) => {
      const [d, at] = k.split("|");
      return [`#${esc(d)} <span class="muted">· ${esc(j.teams[1 - Number(at)] ?? "")}</span>`, v.sum / v.n, v.n];
    }).filter((d) => d[2] >= fps / 2).sort((a, b) => b[1] - a[1]).slice(0, 6);
    $("#defenders").innerHTML = dv.map(([k, v]) => bar(k, v, dv[0][1], pct(v))).join("") || `<div class="empty">No matchups.</div>`;

    const zones = {};
    att.forEach((r) => { if (r.target_zone) zones[r.target_zone] = (zones[r.target_zone] || 0) + 1; });
    const zmax = Math.max(1, ...Object.values(zones));
    const ztotal = Object.values(zones).reduce((a, b) => a + b, 0) || 1;
    $("#zones").innerHTML = ZONE_ORDER.map((z) => {
      const v = zones[z] || 0, a = 0.06 + 0.84 * (v / zmax), top = v === zmax && v > 0;
      return `<div class="zone ${a > 0.55 ? "top" : ""}" style="background:rgba(45,226,190,${a.toFixed(2)})" title="Zone ${z}: ${pct(v / ztotal)}">
        <span class="zn">Zone ${z}</span><span class="zv" style="${a > 0.55 ? "color:#04241e" : ""}">${pct(v / ztotal)}</span></div>`;
    }).join("");
  }

  /* pretrained group-activity GAT: rally timeline + action share per team */
  function renderRally(j) {
    const card = $("#rally-card");
    card.hidden = !ga?.windows?.length;
    if (card.hidden) return;
    const n = Math.max(ga.frames - 1, 1);
    $("#rally").innerHTML = ga.windows.map((w, i) => {
      const [team, action] = splitLabel(w.label);
      return `<button class="seg" data-i="${i}" type="button"
        style="left:${(w.start_frame / n) * 100}%;width:${((w.end_frame - w.start_frame + 1) / n) * 100}%;--c:${ACTION_COLOR[action] || "#8694a8"};--t:${TEAM_COLOR[team]}"
        title="${clock(w.start_s)}–${clock(w.end_s)} · ${esc(j.teams[team])} ${esc(action)} · ${pct(w.score)}"></button>`;
    }).join("") + `<span class="rally-head" id="rally-head"></span>`;
    $$("#rally .seg").forEach((b) => b.addEventListener("click", () => seek(ga.windows[b.dataset.i].start_s)));
    $("#rally-legend").innerHTML = Object.entries(ACTION_COLOR)
      .map(([k, c]) => `<span><i style="background:${c}"></i>${k}</span>`).join("")
      + j.teams.map((t, i) => `<span><i class="edge" style="background:${TEAM_COLOR[i]}"></i>${esc(t)} side</span>`).join("");
    $("#rally-teams").innerHTML = [0, 1].map((team) => {
      const counts = {};
      ga.windows.forEach((w) => { const [t, a] = splitLabel(w.label); if (t === team) counts[a] = (counts[a] || 0) + 1; });
      const total = Object.values(counts).reduce((a, b) => a + b, 0);
      const items = Object.keys(ACTION_COLOR).map((a) => [a, counts[a] || 0]);
      const max = Math.max(1, ...items.map((x) => x[1]));
      return `<div><h4><i style="background:${TEAM_COLOR[team]}"></i>${esc(j.teams[team])}<small class="muted">${dur(total * ga.stride / fps)} in play</small></h4>
        <div class="bars">${items.map(([a, v]) => bar(a.replace(/^./, (c) => c.toUpperCase()), v, max,
          total ? pct(v / total) : "—", ACTION_COLOR[a])).join("")}</div></div>`;
    }).join("");
    lastWin = -1;
  }

  let lastWin = -1;
  function syncRally(t) {
    if (!ga?.windows?.length) return;
    const f = t * fps;
    const head = $("#rally-head");
    if (head) head.style.left = `${Math.min(f / Math.max(ga.frames - 1, 1), 1) * 100}%`;
    let i = ga.windows.findIndex((w) => f >= w.start_frame && f <= w.end_frame + 1);
    if (i < 0) i = ga.windows.length - 1;
    if (i === lastWin) return;
    lastWin = i;
    $$("#rally .seg").forEach((b, k) => b.classList.toggle("now", k === i));
    const w = ga.windows[i], [team, action] = splitLabel(w.label);
    $("#ga-now").innerHTML = `<span class="ga-team" style="border-color:${TEAM_COLOR[team]}">${esc(current.job.teams[team])}</span>
      <b style="color:${ACTION_COLOR[action]}">${esc(action)}</b><span class="muted">${pct(w.score)}</span>`;
  }

  function renderSnaps(j) {
    const snaps = j.snapshots || [];
    $("#snaps-card").hidden = !snaps.length;
    $("#snaps").innerHTML = snaps.map((s) => `<img loading="lazy" src="/files/${j.id}/dashboard/${s}" alt="Dashboard snapshot ${esc(s)}">`).join("");
    $$("#snaps img").forEach((img) => img.addEventListener("click", () => {
      $("#lightbox img").src = img.src;
      $("#lightbox").hidden = false;
    }));
  }
  $("#lightbox").addEventListener("click", () => ($("#lightbox").hidden = true));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#lightbox").hidden = true; });

  let lastNow = -1, feedHover = false;
  $("#feed").addEventListener("mouseenter", () => (feedHover = true));
  $("#feed").addEventListener("mouseleave", () => (feedHover = false));
  function syncToVideo() {
    cancelAnimationFrame(raf);
    const player = $("#player");
    const tick = () => {
      const t = player.currentTime || 0;
      const host = $("#chart");
      if (host._x && rows.length) {
        const px = host._x(t);
        const ph = $("#playhead");
        ph?.setAttribute("x1", px);
        ph?.setAttribute("x2", px);
      }
      syncRally(t);
      let now = -1;
      lines.forEach((l, i) => { if (l.t <= t + 0.05) now = i; });
      if (now !== lastNow) {
        lastNow = now;
        $$("#feed li[data-i]").forEach((li, i) => {
          li.classList.toggle("past", i < now);
          li.classList.toggle("now", i === now);
        });
        const el = $(`#feed li[data-i="${now}"]`);
        if (el && !feedHover) {
          const feed = $("#feed");
          feed.scrollTop = el.offsetTop - feed.clientHeight / 2 + el.clientHeight / 2;
        }
      }
      if (!$("#results").hidden) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
  }

  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (!$("#results").hidden && rows.length) renderChart(); }, 120);
  });

  /* ---------------- Router ---------------- */
  function route() {
    const h = location.hash || "#/";
    const [, page, id] = h.split("/");
    const view = page === "job" && id ? "job" : page === "runs" ? "runs" : "home";
    $$(".view").forEach((v) => (v.hidden = v.id !== `view-${view}`));
    $$("[data-nav]").forEach((a) => a.classList.toggle("on", a.dataset.nav === view));
    if (view !== "job") {
      clearTimeout(pollTimer);
      cancelAnimationFrame(raf);
      $("#player").pause();
      current = null;
      loadJobs();
    } else {
      openJob(id);
    }
    window.scrollTo(0, 0);
  }
  window.addEventListener("hashchange", route);

  loadSystem();
  refreshForm();
  route();
  setInterval(() => { if (!location.hash.startsWith("#/job")) loadJobs(); }, 5000);
})();
