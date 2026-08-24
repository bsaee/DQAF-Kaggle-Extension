chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "OPEN_DASHBOARD") {
    const slug = message.slug;
    const target = message.targetColumn || "";
    const dashboardUrl = chrome.runtime.getURL(
      `dashboard/dashboard.html?slug=${encodeURIComponent(slug)}&target=${encodeURIComponent(target)}`
    );
    chrome.tabs.create({ url: dashboardUrl });
  }
});