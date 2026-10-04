// Draws the dashboard charts from JSON embedded in each page.

(function () {
  "use strict";

  const style = getComputedStyle(document.documentElement);
  const colour = (name) => style.getPropertyValue(name).trim();
  const palette = [colour("--accent"), "#d97706", "#7c3aed", "#0d9488"];

  Chart.defaults.font.family = colour("--font") || "system-ui, sans-serif";
  Chart.defaults.color = colour("--muted");
  Chart.defaults.borderColor = colour("--border");

  function readData(id) {
    const element = document.getElementById(id);
    return element ? JSON.parse(element.textContent) : null;
  }

  function formatDate(milliseconds) {
    return new Date(milliseconds).toLocaleDateString("en-GB", { month: "short", year: "numeric" });
  }

  function drawBankroll() {
    const data = readData("bankroll-data");
    const canvas = document.getElementById("bankroll-chart");
    if (!data || !data.length || !canvas) return;

    const datasets = data.map((series, index) => ({
      label: series.name,
      data: series.points,
      borderColor: palette[index % palette.length],
      backgroundColor: palette[index % palette.length],
      borderWidth: 2,
      pointRadius: 0,
      tension: 0,
    }));
    const times = data.flatMap((series) => series.points.map((point) => point.x));

    new Chart(canvas, {
      type: "line",
      data: { datasets },
      options: {
        maintainAspectRatio: false,
        interaction: { mode: "nearest", intersect: false },
        scales: {
          x: {
            type: "linear",
            min: Math.min(...times),
            max: Math.max(...times),
            ticks: { callback: formatDate, maxTicksLimit: 8 },
          },
          y: { title: { display: true, text: "Bankroll (units)" } },
        },
        plugins: {
          tooltip: {
            callbacks: {
              title: (items) => new Date(items[0].parsed.x).toLocaleDateString("en-GB"),
              label: (item) => `${item.dataset.label}: ${item.parsed.y.toFixed(2)}`,
            },
          },
        },
      },
    });
  }

  function drawCalibration() {
    const data = readData("calibration-data");
    const canvas = document.getElementById("calibration-chart");
    if (!data || !data.length || !canvas) return;

    const datasets = data.map((series, index) => ({
      label: series.name,
      data: series.points,
      borderColor: palette[index % palette.length],
      backgroundColor: palette[index % palette.length],
      borderWidth: 2,
      pointRadius: 4,
      showLine: true,
    }));
    datasets.push({
      label: "Perfect calibration",
      data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
      borderColor: colour("--muted"),
      borderDash: [6, 6],
      borderWidth: 1,
      pointRadius: 0,
      showLine: true,
    });

    const percent = (value) => `${Math.round(value * 100)}%`;
    new Chart(canvas, {
      type: "scatter",
      data: { datasets },
      options: {
        maintainAspectRatio: false,
        scales: {
          x: { min: 0, max: 1, title: { display: true, text: "Forecast probability" }, ticks: { callback: percent } },
          y: { min: 0, max: 1, title: { display: true, text: "How often it happened" }, ticks: { callback: percent } },
        },
        plugins: {
          tooltip: {
            filter: (item) => item.raw.count !== undefined,
            callbacks: {
              label: (item) =>
                `${item.dataset.label}: forecast ${(item.raw.x * 100).toFixed(1)}%, ` +
                `happened ${(item.raw.y * 100).toFixed(1)}% of ${item.raw.count}`,
            },
          },
        },
      },
    });
  }

  drawBankroll();
  drawCalibration();
})();
