let currentSlug = null;
let cachedProfile = null;

document.addEventListener("DOMContentLoaded", async () => {
  initAuthUI();
  await checkActiveTabAndLoad();

  document.getElementById("taskModeSelect").addEventListener("change", (e) => {
    const mode = e.target.value;
    const targetGroup = document.getElementById("targetGroup");
    targetGroup.style.display = (mode === "unsupervised") ? "none" : "flex";
    fetchProfile(currentSlug, document.getElementById("targetSelect").value, mode);
  });

  document.getElementById("targetSelect").addEventListener("change", (e) => {
    const mode = document.getElementById("taskModeSelect").value;
    fetchProfile(currentSlug, e.target.value, mode);
  });

  document.getElementById("openDashboardBtn").addEventListener("click", () => {
    const targetCol = document.getElementById("targetSelect").value;
    const mode = document.getElementById("taskModeSelect").value;
    chrome.runtime.sendMessage({
      action: "OPEN_DASHBOARD",
      slug: currentSlug,
      targetColumn: targetCol,
      taskMode: mode
    });
  });
});

function initAuthUI() {
  const toggleBtn = document.getElementById("toggleAuthBtn");
  const authInputs = document.getElementById("authInputs");
  const saveBtn = document.getElementById("saveAuthBtn");

  chrome.storage.local.get(["kaggleUsername", "kaggleKey"], (res) => {
    if (res.kaggleUsername && res.kaggleKey) {
      document.getElementById("usernameInput").value = res.kaggleUsername;
      document.getElementById("keyInput").value = res.kaggleKey;
      authInputs.classList.add("hidden");
      toggleBtn.innerText = "Edit";
    } else {
      authInputs.classList.remove("hidden");
      toggleBtn.innerText = "Hide";
    }
  });

  toggleBtn.addEventListener("click", () => {
    authInputs.classList.toggle("hidden");
    toggleBtn.innerText = authInputs.classList.contains("hidden") ? "Edit" : "Hide";
  });

  saveBtn.addEventListener("click", () => {
    const u = document.getElementById("usernameInput").value.trim();
    const k = document.getElementById("keyInput").value.trim();
    chrome.storage.local.set({ kaggleUsername: u, kaggleKey: k }, () => {
      authInputs.classList.add("hidden");
      toggleBtn.innerText = "Edit";
      if (currentSlug) fetchProfile(currentSlug);
    });
  });
}

async function checkActiveTabAndLoad() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || !tab.url) return;

  try {
    const url = new URL(tab.url);
    const parts = url.pathname.split("/").filter(Boolean);
    if (url.hostname.includes("kaggle.com") && parts[0] === "datasets" && parts.length >= 3) {
      currentSlug = `${parts[1]}/${parts[2]}`;
      document.getElementById("datasetSlugText").innerText = currentSlug;
      fetchProfile(currentSlug);
    } else {
      showError("Please open a specific Kaggle dataset page to view nutrition facts.");
    }
  } catch (err) {
    showError("Could not inspect current tab URL.");
  }
}

async function fetchProfile(slug, targetOverride = null, taskMode = null) {
  showLoading(true);
  showError(null);

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

    if (!response.ok) {
      const errJson = await response.json().catch(() => ({}));
      throw new Error(errJson.detail || `Server responded with status ${response.status}`);
    }

    cachedProfile = await response.json();
    renderNutritionCard(cachedProfile);
  } catch (err) {
    console.error("DQAF Fetch Error:", err);
    showError(err.message || "Failed to reach DQAF backend.");
  } finally {
    showLoading(false);
  }
}

function renderNutritionCard(data) {
  document.getElementById("nutritionContent").classList.remove("hidden");
  document.getElementById("healthScoreBadge").innerText = `${data.health_score}/100`;

  // 1. Sync Motive Dropdown to Backend's Inferred / Active Task
  const taskSelect = document.getElementById("taskModeSelect");
  const rawTask = (data.target_detection.task_type || "").toLowerCase();
  if (rawTask.includes("classification")) {
    taskSelect.value = "classification";
  } else if (rawTask.includes("regression")) {
    taskSelect.value = "regression";
  } else {
    taskSelect.value = "unsupervised";
  }

  const targetGroup = document.getElementById("targetGroup");
  targetGroup.style.display = (taskSelect.value === "unsupervised") ? "none" : "flex";

  // 2. Populate Target Dropdown
  const select = document.getElementById("targetSelect");
  select.innerHTML = "";
  const cols = data.target_detection.available_columns || [];
  cols.forEach(col => {
    const opt = document.createElement("option");
    opt.value = col;
    opt.innerText = col;
    if (col === data.target_detection.target_column) opt.selected = true;
    select.appendChild(opt);
  });

  // 3. Vitals
  document.getElementById("totalRows").innerText = data.vitals.total_rows.toLocaleString();
  document.getElementById("colDistribution").innerText =
    `${data.vitals.numeric_column_count} Num / ${data.vitals.categorical_column_count} Cat`;
  document.getElementById("missingPct").innerText = `${data.vitals.missing_cell_pct}%`;
  document.getElementById("duplicatePct").innerText = `${data.vitals.duplicate_row_pct}%`;

  // 4. Risk Badges
  const badgesContainer = document.getElementById("riskBadgesContainer");
  badgesContainer.innerHTML = "";
  const flags = data.risk_flags;

  if (flags.has_severe_imbalance) {
    badgesContainer.innerHTML += `<span class="risk-badge red">Severe Imbalance</span>`;
  } else if (flags.has_moderate_imbalance) {
    badgesContainer.innerHTML += `<span class="risk-badge amber">Moderate Imbalance</span>`;
  }

  if (flags.has_leakage_suspect) {
    badgesContainer.innerHTML += `<span class="risk-badge red">Leakage Risk</span>`;
  }
  if (flags.high_cardinality_columns && flags.high_cardinality_columns.length > 0) {
    badgesContainer.innerHTML += `<span class="risk-badge amber">Cardinality (${flags.high_cardinality_columns.length})</span>`;
  }
  if (flags.collinear_pairs && flags.collinear_pairs.length > 0) {
    badgesContainer.innerHTML += `<span class="risk-badge amber">Collinear (${flags.collinear_pairs.length})</span>`;
  }
  if (badgesContainer.children.length === 0) {
    badgesContainer.innerHTML = `<span class="risk-badge green">Clean Structure</span>`;
  }

  // 5. Model Snapshot
  if (data.stress_test && data.stress_test.model_snapshot) {
    const snap = data.stress_test.model_snapshot;
    document.getElementById("linearStability").innerText = snap.linear_model_stability;
    document.getElementById("treeStability").innerText = snap.tree_model_stability;
  } else {
    document.getElementById("linearStability").innerText = "N/A (Non-classification)";
    document.getElementById("treeStability").innerText = "N/A (Non-classification)";
  }
}

function showLoading(show) {
  document.getElementById("loadingState").classList.toggle("hidden", !show);
  if (show) document.getElementById("nutritionContent").classList.add("hidden");
}

function showError(msg) {
  const errContainer = document.getElementById("errorState");
  if (msg) {
    errContainer.classList.remove("hidden");
    document.getElementById("errorMessage").innerText = msg;
  } else {
    errContainer.classList.add("hidden");
  }
}