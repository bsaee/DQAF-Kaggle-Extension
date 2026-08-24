function getDatasetSlug() {
  const parts = window.location.pathname.split("/").filter(Boolean);
  if (parts[0] === "datasets" && parts.length >= 3) {
    return `${parts[1]}/${parts[2]}`;
  }
  return null;
}

function injectBadge() {
  const slug = getDatasetSlug();
  if (!slug || document.getElementById("dqaf-injected-badge")) return;

  // Locate the header container containing the dataset title
  const titleElem = document.querySelector("h1");
  if (!titleElem) return;

  const headerContainer = titleElem.parentElement;
  if (!headerContainer) return;

  const badge = document.createElement("div");
  badge.id = "dqaf-injected-badge";
  badge.innerHTML = `
    <span class="dqaf-badge-pill">DQAF</span>
    <span>Dataset Nutrition & ML Impact</span>
    <span style="font-size: 14px;">↗</span>
  `;

  badge.addEventListener("click", () => {
    chrome.runtime.sendMessage({
      action: "OPEN_DASHBOARD",
      slug: slug
    });
  });

  titleElem.style.display = "inline-flex";
  titleElem.style.alignItems = "center";
  titleElem.appendChild(badge);
}

// Observe URL changes for Single Page Application (SPA) navigation in Kaggle
const observer = new MutationObserver(() => {
  injectBadge();
});

observer.observe(document.body, { childList: true, subtree: true });
injectBadge();