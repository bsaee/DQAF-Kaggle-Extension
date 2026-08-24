let degradationChartInstance = null;
let entropyChartInstance = null;

document.addEventListener("DOMContentLoaded", () => {
  const urlParams = new URLSearchParams(window.location.search);
  const slug = urlParams.get("slug");
  const targetOverride = urlParams.get("target") || null;
  const modeParam = urlParams.get("mode") || null;

  if (slug) {
    document.getElementById("datasetTitle").innerText = slug;
    loadDashboardData(slug, targetOverride, modeParam);
  }

  document.getElementById("dashboardTaskModeSelect").addEventListener("change", (e) => {
    const currentTarget = document.getElementById("dashboardTargetSelect").value;
    const mode = e.target.value;
    document.getElementById("dashboardTargetSelect").style.display = (mode === "unsupervised") ? "none" : "inline-block";
    loadDashboardData(slug, currentTarget, mode);
  });

  document.getElementById("dashboardTargetSelect").addEventListener("change", (e) => {
    const currentMode = document.getElementById("dashboardTaskModeSelect").value;
    loadDashboardData(slug, e.target.value, currentMode);
  });

  document.getElementById("refreshBtn").addEventListener("click", () => {
    const currentTarget = document.getElementById("dashboardTargetSelect").value;
    const currentMode = document.getElementById("dashboardTaskModeSelect").value;
    loadDashboardData(slug, currentTarget, currentMode);
  });
});

async function loadDashboardData(slug, targetOverride = null, taskMode = null) {
  const authData = await chrome.storage.local.get(["kaggleUsername", "kaggleKey"]);
  const authPayload = (authData.kaggleUsername && authData.kaggleKey)
    ? { username: authData.kaggleUsername, key: authData.kaggleKey }
    : null;

  const payload = {
    dataset_slug: slug,
    target_column: targetOverride,
    task_mode: taskMode || "auto",
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
    renderCompleteStudio(data);
  } catch (err) {
    console.error("Dashboard Load Error:", err);
  }
}

function renderCompleteStudio(data) {
  // 1. Sync Task Dropdown with Backend Decision
  const modeSelect = document.getElementById("dashboardTaskModeSelect");
  const rawTask = (data.target_detection.task_type || "").toLowerCase();
  if (rawTask.includes("classification")) {
    modeSelect.value = "classification";
  } else if (rawTask.includes("regression")) {
    modeSelect.value = "regression";
  } else {
    modeSelect.value = "unsupervised";
  }

  const targetSelect = document.getElementById("dashboardTargetSelect");
  targetSelect.style.display = (modeSelect.value === "unsupervised") ? "none" : "inline-block";

  // 2. Populate Target Selector
  targetSelect.innerHTML = "";
  (data.target_detection.available_columns || []).forEach(col => {
    const opt = document.createElement("option");
    opt.value = col;
    opt.innerText = col;
    if (col === data.target_detection.target_column) opt.selected = true;
    targetSelect.appendChild(opt);
  });

  // 3. Render Docked Nutrition Card
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
  if (data.risk_flags.has_leakage_suspect) badgesBox.innerHTML += `<span class="badge red">Leakage Risk</span>`;
  if (data.risk_flags.high_cardinality_columns.length > 0)
    badgesBox.innerHTML += `<span class="badge amber">Cardinality (${data.risk_flags.high_cardinality_columns.length})</span>`;
  if (data.risk_flags.collinear_pairs.length > 0)
    badgesBox.innerHTML += `<span class="badge amber">Collinear (${data.risk_flags.collinear_pairs.length})</span>`;
  if (badgesBox.children.length === 0) badgesBox.innerHTML = `<span class="badge green">Healthy Structure</span>`;

  // Class Distribution
  const distBox = document.getElementById("classDistContainer");
  distBox.innerHTML = "";
  for (const [cls, pct] of Object.entries(data.class_distribution || {})) {
    distBox.innerHTML += `<div class="dist-item"><span>Class ${cls}</span><b>${pct}%</b></div>`;
  }
  if (Object.keys(data.class_distribution || {}).length === 0) {
    distBox.innerHTML = `<span style="font-size:11px; color:#64748b;">N/A (Non-classification)</span>`;
  }

  // Model Snapshot
  if (data.stress_test && data.stress_test.model_snapshot) {
    document.getElementById("dockedLinear").innerText = data.stress_test.model_snapshot.linear_model_stability;
    document.getElementById("dockedTree").innerText = data.stress_test.model_snapshot.tree_model_stability;
  } else {
    document.getElementById("dockedLinear").innerText = "N/A";
    document.getElementById("dockedTree").innerText = "N/A";
  }

  // 4. Render Degradation Chart
  renderDegradationChart(data.stress_test);

  // 5. Render Entropy Chart
  renderEntropyChart(data.column_profiles);

  // 6. Render Collinearities & Leakage tables
  renderCollinearities(data.risk_flags.collinear_pairs);
  renderLeakage(data.risk_flags.leakage_columns);

  // 7. Render Remediation Recipes
  renderRecipes(data.remediation_recipes);
}

function renderDegradationChart(stressData) {
  const ctx = document.getElementById("degradationChart").getContext("2d");
  if (degradationChartInstance) degradationChartInstance.destroy();

  if (!stressData || !stressData.degradation_curve) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    return;
  }

  const labels = stressData.degradation_curve.missingness_levels.map(l => `${l * 100}% Missing`);
  const models = stressData.degradation_curve.models;

  degradationChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        { label: "Logistic Regression", data: models["Logistic Regression"], borderColor: "#ef4444", tension: 0.2 },
        { label: "Random Forest", data: models["Random Forest"], borderColor: "#10b981", tension: 0.2 },
        { label: "Gradient Boosting", data: models["Gradient Boosting"], borderColor: "#3b82f6", tension: 0.2 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { min: 0, max: 1.0, title: { display: true, text: "F1-Score" } } }
    }
  });
}

function renderEntropyChart(columns) {
  if (!columns) return;
  const ctx = document.getElementById("entropyChart").getContext("2d");
  if (entropyChartInstance) entropyChartInstance.destroy();

  const labels = columns.map(c => c.name);
  const data = columns.map(c => c.shannon_entropy);

  entropyChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{ label: "Entropy (bits)", data: data, backgroundColor: "#6366f1" }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { title: { display: true, text: "Bits" } } }
    }
  });
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

function renderLeakage(cols) {
  const container = document.getElementById("leakageListContainer");
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