let primaryChartInstance = null;
let secondaryChartInstance = null;

document.addEventListener("DOMContentLoaded", () => {
  const urlParams = new URLSearchParams(window.location.search);
  const slug = urlParams.get("slug");
  const targetOverride = urlParams.get("target") || null;
  const modeParam = urlParams.get("mode") || "classification";

  const modeSelect = document.getElementById("dashboardTaskModeSelect");
  modeSelect.value = modeParam;

  if (slug) {
    document.getElementById("datasetTitle").innerText = slug;
    loadDashboardData(slug, targetOverride, modeParam);
  }

  modeSelect.addEventListener("change", (e) => {
    const selectedMode = e.target.value;
    const targetSelect = document.getElementById("dashboardTargetSelect");
    targetSelect.style.display = (selectedMode === "unsupervised") ? "none" : "inline-block";
    loadDashboardData(slug, targetSelect.value, selectedMode);
  });

  document.getElementById("dashboardTargetSelect").addEventListener("change", (e) => {
    const selectedMode = document.getElementById("dashboardTaskModeSelect").value;
    loadDashboardData(slug, e.target.value, selectedMode);
  });

  document.getElementById("refreshBtn").addEventListener("click", () => {
    const currentTarget = document.getElementById("dashboardTargetSelect").value;
    const selectedMode = document.getElementById("dashboardTaskModeSelect").value;
    loadDashboardData(slug, currentTarget, selectedMode);
  });
});

async function loadDashboardData(slug, targetOverride = null, taskMode = "classification") {
  const authData = await chrome.storage.local.get(["kaggleUsername", "kaggleKey"]);
  const authPayload = (authData.kaggleUsername && authData.kaggleKey)
    ? { username: authData.kaggleUsername, key: authData.kaggleKey }
    : null;

  const payload = {
    dataset_slug: slug,
    target_column: targetOverride,
    task_mode: taskMode,
    auth: authPayload,
    max_sample_rows: 5000
  };

  try {
    let response;
    try {
      response = await fetch("http://localhost:8000/api/v1/profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
    } catch (e) {
      response = await fetch("http://127.0.0.1:8000/api/v1/profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
    }

    if (!response.ok) throw new Error("Could not fetch analytics.");
    const data = await response.json();
    renderCompleteStudio(data, taskMode);
  } catch (err) {
    console.error("Dashboard Load Error:", err);
  }
}

function renderCompleteStudio(data, activeMode) {
  // 1. Target Selector Visibility
  const targetSelect = document.getElementById("dashboardTargetSelect");
  targetSelect.style.display = (activeMode === "unsupervised") ? "none" : "inline-block";
  
  targetSelect.innerHTML = "";
  (data.target_detection.available_columns || []).forEach(col => {
    const opt = document.createElement("option");
    opt.value = col;
    opt.innerText = col;
    if (col === data.target_detection.target_column) opt.selected = true;
    targetSelect.appendChild(opt);
  });

  // 2. Docked Nutrition Card
  document.getElementById("dockedHealthScore").innerText = `${data.health_score}/100`;
  document.getElementById("dockedRows").innerText = data.vitals.total_rows.toLocaleString();
  document.getElementById("dockedCols").innerText =
    `${data.vitals.numeric_column_count} Num / ${data.vitals.categorical_column_count} Cat`;
  document.getElementById("dockedMissing").innerText = `${data.vitals.missing_cell_pct}%`;
  document.getElementById("dockedDuplicates").innerText = `${data.vitals.duplicate_row_pct}%`;

  // Risk Badges
  const badgesBox = document.getElementById("dockedRiskBadges");
  badgesBox.innerHTML = "";
  if (data.risk_flags.has_severe_imbalance) badgesBox.innerHTML += `<span class="badge red">Severe Imbalance</span>`;
  if (data.risk_flags.has_moderate_imbalance) badgesBox.innerHTML += `<span class="badge amber">Moderate Imbalance</span>`;
  if (data.risk_flags.has_high_skew) badgesBox.innerHTML += `<span class="badge amber">High Target Skew</span>`;
  if (data.risk_flags.has_leakage_suspect) badgesBox.innerHTML += `<span class="badge red">Leakage Risk</span>`;
  if (data.risk_flags.high_cardinality_columns && data.risk_flags.high_cardinality_columns.length > 0)
    badgesBox.innerHTML += `<span class="badge amber">Cardinality (${data.risk_flags.high_cardinality_columns.length})</span>`;
  if (data.risk_flags.collinear_pairs && data.risk_flags.collinear_pairs.length > 0)
    badgesBox.innerHTML += `<span class="badge amber">Collinear (${data.risk_flags.collinear_pairs.length})</span>`;
  if (badgesBox.children.length === 0) badgesBox.innerHTML = `<span class="badge green">Healthy Structure</span>`;

  // Task-Specific Side Panel (Distribution / Skewness)
  const distBox = document.getElementById("classDistContainer");
  distBox.innerHTML = "";
  
  if (activeMode === "classification" && Object.keys(data.class_distribution || {}).length > 0) {
    for (const [cls, pct] of Object.entries(data.class_distribution)) {
      distBox.innerHTML += `<div class="dist-item"><span>Class ${cls}</span><b>${pct}%</b></div>`;
    }
  } else if (activeMode === "regression" && data.regression_stats) {
    distBox.innerHTML = `
      <div class="dist-item"><span>Target Skew:</span><b>${data.regression_stats.skewness ?? 0}</b></div>
      <div class="dist-item"><span>Kurtosis:</span><b>${data.regression_stats.kurtosis ?? 0}</b></div>
    `;
  } else {
    distBox.innerHTML = `<div class="dist-item"><span>Space:</span><b>Unsupervised (No Target)</b></div>`;
  }

  // Model Snapshot
  const benchmark = data.stress_test || {};
  if (benchmark.model_snapshot) {
    document.getElementById("dockedLinear").innerText = benchmark.model_snapshot.linear_model_stability;
    document.getElementById("dockedTree").innerText = benchmark.model_snapshot.tree_model_stability;
  } else if (benchmark.unsupervised_snapshot) {
    document.getElementById("dockedLinear").innerText = benchmark.unsupervised_snapshot.scale_disparity ? "Scale Alert" : "Balanced";
    document.getElementById("dockedTree").innerText = `${benchmark.unsupervised_snapshot.components_for_80_pct} PCs (80% Var)`;
  }

  // 3. Render Dynamic Primary & Secondary Charts
  renderDynamicPrimaryChart(benchmark, activeMode);
  renderDynamicSecondaryChart(data, activeMode);

  // 4. Collinearities & Leakage
  renderCollinearities(data.risk_flags.collinear_pairs);
  renderLeakage(data.risk_flags.leakage_columns, activeMode);

  // 5. Remediation Recipes
  renderRecipes(data.remediation_recipes);
}

function renderDynamicPrimaryChart(benchmark, activeMode) {
  const ctx = document.getElementById("degradationChart").getContext("2d");
  const headerElem = document.querySelector("#degradationChart").closest(".chart-card").querySelector(".chart-header");

  if (primaryChartInstance) primaryChartInstance.destroy();

  if (activeMode === "unsupervised" && benchmark.pca_scree) {
    headerElem.querySelector("h4").innerText = "PCA Cumulative Explained Variance";
    headerElem.querySelector(".subtext").innerText = "Dimensionality capture curve (Scree Plot)";

    primaryChartInstance = new Chart(ctx, {
      type: "line",
      data: {
        labels: benchmark.pca_scree.labels,
        datasets: [
          {
            label: "Cumulative Variance (%)",
            data: benchmark.pca_scree.cumulative_variance,
            borderColor: "#3b82f6",
            backgroundColor: "rgba(59, 130, 246, 0.15)",
            fill: true,
            tension: 0.2
          },
          {
            type: "bar",
            label: "Individual Variance (%)",
            data: benchmark.pca_scree.individual_variance,
            backgroundColor: "#94a3b8"
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { min: 0, max: 100, title: { display: true, text: "Variance %" } } }
      }
    });
  } else if (benchmark.degradation_curve) {
    const isRegression = activeMode === "regression";
    const metricLabel = isRegression ? "R² Score" : "F1-Score";

    headerElem.querySelector("h4").innerText = isRegression ? "Regression Degradation Stress-Test" : "Algorithmic Degradation Stress-Test";
    headerElem.querySelector(".subtext").innerText = `${metricLabel} drop across induced missingness`;

    const labels = benchmark.degradation_curve.missingness_levels.map(l => `${l * 100}% Missing`);
    const models = benchmark.degradation_curve.models;

    const datasets = [];
    const colors = ["#ef4444", "#10b981", "#3b82f6"];
    let idx = 0;
    for (const [name, vals] of Object.entries(models)) {
      datasets.push({
        label: name,
        data: vals,
        borderColor: colors[idx % colors.length],
        tension: 0.2
      });
      idx++;
    }

    primaryChartInstance = new Chart(ctx, {
      type: "line",
      data: { labels: labels, datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { min: 0, max: 1.0, title: { display: true, text: metricLabel } } }
      }
    });
  }
}

function renderDynamicSecondaryChart(data, activeMode) {
  const ctx = document.getElementById("entropyChart").getContext("2d");
  const headerElem = document.querySelector("#entropyChart").closest(".chart-card").querySelector(".chart-header");

  if (secondaryChartInstance) secondaryChartInstance.destroy();

  // Mode 1: Regression (Target Distribution Histogram)
  if (activeMode === "regression" && data.target_histogram) {
    headerElem.querySelector("h4").innerText = "Target Distribution & Normality";
    headerElem.querySelector(".subtext").innerText = "Binned target frequencies (verifying skew & kurtosis)";

    secondaryChartInstance = new Chart(ctx, {
      type: "bar",
      data: {
        labels: data.target_histogram.labels,
        datasets: [{ label: "Target Frequency", data: data.target_histogram.counts, backgroundColor: "#10b981" }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { title: { display: true, text: "Sample Count" } } }
      }
    });
  }
  // Mode 2: Unsupervised (Feature Variance / Scale Disparity)
  else if (activeMode === "unsupervised") {
    headerElem.querySelector("h4").innerText = "Feature Variance & Magnitude Spread";
    headerElem.querySelector(".subtext").innerText = "Scale disparities that distort Euclidean distances";

    const numCols = data.column_profiles.filter(c => c.variance > 0);
    const labels = numCols.map(c => c.name);
    const variances = numCols.map(c => c.variance);

    secondaryChartInstance = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{ label: "Variance", data: variances, backgroundColor: "#f59e0b" }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { title: { display: true, text: "Variance" } } }
      }
    });
  }
  // Mode 3: Classification (Shannon Entropy)
  else {
    headerElem.querySelector("h4").innerText = "Information Content (Shannon Entropy)";
    headerElem.querySelector(".subtext").innerText = "Per-column entropy (H(X) in bits)";

    const labels = data.column_profiles.map(c => c.name);
    const entropyData = data.column_profiles.map(c => c.shannon_entropy);

    secondaryChartInstance = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{ label: "Entropy (bits)", data: entropyData, backgroundColor: "#6366f1" }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { title: { display: true, text: "Bits" } } }
      }
    });
  }
}

function renderCollinearities(pairs) {
  const container = document.getElementById("collinearListContainer");
  if (!pairs || pairs.length === 0) {
    container.innerHTML = `<p class="empty-state">No extreme collinear pairs identified (r < 0.85).</p>`;
    return;
  }
  let html = `<table style="width:100%; border-collapse: collapse;"><tr><th align="left">Feature 1</th><th align="left">Feature 2</th><th align="right">Pearson r</th></tr>`;
  pairs.forEach(p => {
    html += `<tr><td>${p.col1}</td><td>${p.col2}</td><td align="right"><b>${p.correlation}</b></td></tr>`;
  });
  html += `</table>`;
  container.innerHTML = html;
}

function renderLeakage(cols, activeMode) {
  const container = document.getElementById("leakageListContainer");
  const headerElem = container.closest(".data-card").querySelector(".chart-header");

  if (activeMode === "unsupervised") {
    headerElem.querySelector("h4").innerText = "Dimensional Sparsity Diagnostics";
    headerElem.querySelector(".subtext").innerText = "Evaluation of feature space sparsity";
    container.innerHTML = `<p class="empty-state">No high-sparsity feature distortions detected.</p>`;
    return;
  }

  headerElem.querySelector("h4").innerText = "Target Leakage Diagnostics";
  headerElem.querySelector(".subtext").innerText = "Features with suspicious correlation (r &ge; 0.98)";

  if (!cols || cols.length === 0) {
    container.innerHTML = `<p class="empty-state">No direct target leakage suspects detected (r < 0.98).</p>`;
    return;
  }
  container.innerHTML = `<p style="color:#ef4444; font-weight:600;">Suspect columns: ${cols.join(", ")}</p>`;
}

function renderRecipes(recipes) {
  const container = document.getElementById("recipesContainer");
  container.innerHTML = "";
  (recipes || []).forEach((r, idx) => {
    const box = document.createElement("div");
    box.className = "recipe-box";
    box.innerHTML = `
      <div class="recipe-box-header">
        <b>${r.title}</b>
        <button class="copy-btn" data-idx="${idx}">Copy</button>
      </div>
      <p class="recipe-desc">${r.description}</p>
      <pre><code>${r.code}</code></pre>
    `;
    box.querySelector(".copy-btn").addEventListener("click", () => {
      navigator.clipboard.writeText(r.code);
      const btn = box.querySelector(".copy-btn");
      btn.innerText = "Copied!";
      setTimeout(() => (btn.innerText = "Copy"), 1500);
    });
    container.appendChild(box);
  });
}