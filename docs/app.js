// No backend, no build step -- fetches the JSON Phase 6 exported and renders
// with Chart.js. Must be served over http(s), not opened as a file:// path
// (fetch() of local files is blocked by the browser in that mode).

const PILLAR_ORDER = ["economic", "external", "fiscal", "institutional", "monetary"];
const PILLAR_COLORS = { economic: "#5fb3ff", external: "#9b7bd6", fiscal: "#e0a85f",
                        institutional: "#4caf78", monetary: "#e0667a" };

let countries = [];
let dsa = {};
let sortState = { key: "model_a_divergence", dir: "desc" };
let chartPillars, chartNotch, chartDsa;

async function main() {
  const [summary, countriesData, dsaData] = await Promise.all([
    fetchJSON("data/summary.json"),
    fetchJSON("data/countries.json"),
    fetchJSON("data/dsa.json"),
  ]);
  countries = countriesData;
  dsa = dsaData;

  document.getElementById("generated-at").textContent =
    `Data generated ${summary.generated_at} · ${summary.n_countries} sovereigns ` +
    `(${summary.n_ig} investment grade) · ${(summary.coverage_overall_fill_rate * 100).toFixed(0)}% ` +
    `indicator coverage.`;

  renderSummary(summary);
  renderTable();

  const preselect = new URLSearchParams(location.search).get("select");
  if (preselect) selectCountry(preselect);
  document.querySelectorAll("#country-table th").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.key;
      sortState = key === sortState.key
        ? { key, dir: sortState.dir === "desc" ? "asc" : "desc" }
        : { key, dir: "desc" };
      renderTable();
    });
  });
}

async function fetchJSON(path) {
  const resp = await fetch(path);
  if (!resp.ok) throw new Error(`Failed to fetch ${path}: ${resp.status}`);
  return resp.json();
}

function renderSummary(summary) {
  const cards = [
    { label: "Model A LOOCV MAE", value: `${summary.model_a.mae.toFixed(2)} notches` },
    { label: "Model A IG/HY mismatches", value: summary.model_a.ig_mismatch_count },
    { label: "Ordered logit exact band match", value: `${(summary.model_b_ordered_logit.exact_match_rate * 100).toFixed(0)}%` },
    { label: "Model B (GBM) LOOCV MAE", value: `${summary.model_b_gbm.mae.toFixed(2)} notches` },
    { label: "Models agree on divergence direction", value: `${summary.agreement_count} countries` },
    {
      label: "Market spread corroboration",
      value: summary.market_cross_check
        ? `${summary.market_cross_check.corroborated}/${summary.market_cross_check.n} (FRED subset)`
        : "no FRED data",
    },
  ];
  const grid = document.getElementById("summary-grid");
  grid.innerHTML = cards.map((c) =>
    `<div class="stat-card"><div class="label">${c.label}</div><div class="value">${c.value}</div></div>`
  ).join("");
}

function fmt(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}

function divergenceClass(v) {
  if (v === null || v === undefined) return "";
  return v > 0 ? "pos" : v < 0 ? "neg" : "";
}

function renderTable() {
  const sorted = [...countries].sort((a, b) => {
    let av = a[sortState.key], bv = b[sortState.key];
    if (typeof av === "string") { av = av ?? ""; bv = bv ?? ""; }
    if (av === null || av === undefined) av = -Infinity;
    if (bv === null || bv === undefined) bv = -Infinity;
    if (typeof av === "boolean") { av = av ? 1 : 0; bv = bv ? 1 : 0; }
    const cmp = av > bv ? 1 : av < bv ? -1 : 0;
    return sortState.dir === "desc" ? -cmp : cmp;
  });

  document.querySelectorAll("#country-table th").forEach((th) => {
    th.classList.toggle("sorted", th.dataset.key === sortState.key);
    th.classList.toggle("asc", th.dataset.key === sortState.key && sortState.dir === "asc");
  });

  const tbody = document.getElementById("country-tbody");
  tbody.innerHTML = sorted.map((c) => `
    <tr data-iso3="${c.iso3}">
      <td>${c.name}</td>
      <td>${c.actual_letter ?? "—"}</td>
      <td>${fmt(c.composite, 2)}</td>
      <td>${fmt(c.model_a_pred_notch)}</td>
      <td class="${divergenceClass(c.model_a_divergence)}">${fmt(c.model_a_divergence, 1)}</td>
      <td>${fmt(c.model_b_gbm_pred_notch)}</td>
      <td class="${divergenceClass(c.model_b_gbm_divergence)}">${fmt(c.model_b_gbm_divergence, 1)}</td>
      <td class="${c.models_agree_direction ? "agree" : ""}">${c.models_agree_direction ? "●" : ""}</td>
    </tr>
  `).join("");

  tbody.querySelectorAll("tr").forEach((tr) => {
    tr.addEventListener("click", () => selectCountry(tr.dataset.iso3));
  });
}

function selectCountry(iso3) {
  const c = countries.find((x) => x.iso3 === iso3);
  if (!c) return;

  document.querySelectorAll("#country-tbody tr").forEach((tr) => {
    tr.classList.toggle("selected", tr.dataset.iso3 === iso3);
  });
  document.getElementById("detail-empty").style.display = "none";
  document.getElementById("detail-charts").style.display = "grid";
  document.getElementById("detail-title").textContent = `Pillar scores — ${c.name}`;

  renderPillarChart(c);
  renderNotchChart(c);
  renderDsaChart(c);
}

function renderPillarChart(c) {
  const labels = PILLAR_ORDER.filter((p) => c.pillars[p] !== undefined && c.pillars[p] !== null);
  const data = labels.map((p) => c.pillars[p]);
  chartPillars?.destroy();
  chartPillars = new Chart(document.getElementById("chart-pillars"), {
    type: "bar",
    data: {
      labels,
      datasets: [{ data, backgroundColor: labels.map((p) => PILLAR_COLORS[p]) }],
    },
    options: {
      scales: { y: { min: 0, max: 1, title: { display: true, text: "percentile rank (signed)" } } },
      plugins: { legend: { display: false } },
    },
  });
}

function renderNotchChart(c) {
  chartNotch?.destroy();
  chartNotch = new Chart(document.getElementById("chart-notch"), {
    type: "bar",
    data: {
      labels: ["Actual (agencies)", "Model A", "Model B (GBM)"],
      datasets: [{
        data: [c.actual_notch, c.model_a_pred_notch, c.model_b_gbm_pred_notch],
        backgroundColor: ["#8a8f9c", "#5fb3ff", "#e0a85f"],
      }],
    },
    options: {
      indexAxis: "y",
      scales: { x: { min: 1, max: 21, title: { display: true, text: "notch (1=D .. 21=AAA)" } } },
      plugins: { legend: { display: false } },
    },
  });
}

function renderDsaChart(c) {
  const d = dsa[c.iso3];
  chartDsa?.destroy();
  if (!d) {
    chartDsa = null;
    return;
  }
  const years = d.years;
  const datasets = [
    { label: "Fan p10–p90", data: d.fan.p90, borderWidth: 0, backgroundColor: "rgba(95,179,255,0.12)", fill: "+1", pointRadius: 0 },
    { label: "_p10", data: d.fan.p10, borderWidth: 0, backgroundColor: "rgba(95,179,255,0.12)", fill: false, pointRadius: 0 },
    { label: "Median (Monte Carlo)", data: d.fan.p50, borderColor: "#5fb3ff", borderDash: [4, 3], pointRadius: 0 },
    { label: "Baseline", data: d.scenarios.baseline, borderColor: "#4caf78", pointRadius: 0, borderWidth: 2 },
    { label: "Combined adverse", data: d.scenarios.combined_adverse, borderColor: "#e0667a", pointRadius: 0, borderWidth: 2 },
  ];
  chartDsa = new Chart(document.getElementById("chart-dsa"), {
    type: "line",
    data: { labels: years.map((y) => `+${y}y`), datasets },
    options: {
      scales: { y: { title: { display: true, text: "debt / GDP (%)" } } },
      plugins: {
        legend: {
          labels: { filter: (item) => !item.text.startsWith("_") },
        },
      },
    },
  });
}

main().catch((err) => {
  console.error(err);
  document.body.insertAdjacentHTML("afterbegin",
    `<div style="background:#e0667a;color:#fff;padding:1rem;font-family:monospace">
       Failed to load dashboard data: ${err.message}. Are you serving this over http://
       (not file://)? Has <code>python -m shadowrating.cli phase6</code> been run?
     </div>`);
});
