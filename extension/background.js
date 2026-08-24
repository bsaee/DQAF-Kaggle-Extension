chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "OPEN_DASHBOARD") {
    const slug = message.slug;
    const target = message.targetColumn || "";
    const mode = message.taskMode || "auto";
    const dashboardUrl = chrome.runtime.getURL(
      `dashboard/dashboard.html?slug=${encodeURIComponent(slug)}&target=${encodeURIComponent(target)}&mode=${encodeURIComponent(mode)}`
    );
    chrome.tabs.create({ url: dashboardUrl });
  }
});