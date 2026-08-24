let currentSlug = null;
let cachedProfile = null;

document.addEventListener("DOMContentLoaded", async () => {
  initAuthUI();
  await checkActiveTabAndLoad();

  document.getElementById("targetSelect").addEventListener("change", (e) => {
    fetchProfile(currentSlug, e.target.value);
  });

  document.getElementById("openDashboardBtn").addEventListener("click", () => {
    const targetCol = document.getElementById("targetSelect").value;
    chrome.runtime.sendMessage({
      action: "OPEN_DASHBOARD",
      slug: currentSlug,
      targetColumn: targetCol
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

async function fetchProfile(slug, targetOverride = null) {
  showLoading(true);
  showError(null);

  const authData = await chrome.storage.local.get(["kaggleUsername", "kaggleKey"]);
  const authPayload = (authData.kaggleUsername && authData.kaggleKey)
    ? { username: authData.kaggleUsername, key: authData.kaggleKey }
    : null;

  try {
    const res = await fetch("http://localhost:8000/api/v1/profile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_slug: slug,
        target_column: targetOverride,
        auth: authPayload,
        max_sample_rows: 5000
      })
    });

    if (!res.ok) {
      const errJson = await res.json();
      throw new Error(errJson.detail || "Failed to profile dataset.");
    }

    cachedProfile = await res.json();
    renderNutritionCard(cachedProfile);
  } catch (err) {
    showError(err.message);
  } finally {
    showLoading(false);
  }
}

function renderNutritionCard(data) {
  document.getElementById("nutritionContent").classList.remove("hidden");
  document.getElementById("healthScoreBadge").innerText = `${data.health_score}/100`;

  // Populate Target Dropdown
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

  // Vitals
  document.getElementById("totalRows").innerText = data.vitals.total_rows.toLocaleString();
  document.getElementById("colDistribution").innerText =
    `${data.vitals.numeric_column_count} Num / ${data.vitals.categorical_column_count} Cat`;
  document.getElementById("missingPct").innerText = `${data.vitals.missing_cell_pct}%`;
  document.getElementById("duplicatePct").innerText = `${data.vitals.duplicate_row_pct}%`;

  // Risk Badges
  const badgesContainer = document.getElementById("riskBadgesContainer");
  badgesContainer.innerHTML = "";
  const flags = data.risk_flags;

  if (flags.has_severe_imbalance) {
    badgesContainer.innerHTML += `<span class="risk-badge red">Severe Imbalance</span>`;
  }
  if (flags.has_leakage_suspect) {
    badgesContainer.innerHTML += `<span class="risk-badge red">Leakage Risk</span>`;
  }
  if (flags.high_cardinality_columns && flags.high_cardinality_columns.length > 0) {
    badgesContainer.innerHTML += `<span class="risk-badge amber">High Cardinality (${flags.high_cardinality_columns.length})</span>`;
  }
  if (flags.collinear_pairs && flags.collinear_pairs.length > 0) {
    badgesContainer.innerHTML += `<span class="risk-badge amber">Collinear Pairs (${flags.collinear_pairs.length})</span>`;
  }
  if (badgesContainer.children.length === 0) {
    badgesContainer.innerHTML = `<span class="risk-badge green">Clean Structure</span>`;
  }

  // Model Snapshot
  if (data.stress_test && data.stress_test.model_snapshot) {
    const snap = data.stress_test.model_snapshot;
    document.getElementById("linearStability").innerText = snap.linear_model_stability;
    document.getElementById("treeStability").innerText = snap.tree_model_stability;
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