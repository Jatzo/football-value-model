// Stake inputs for the paper bets table and the bet calculator on the fixtures page.
(function () {
  "use strict";

  function money(value) {
    return value.toLocaleString("en-GB", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function percent(value, signed) {
    const text = (value * 100).toFixed(1) + "%";
    return signed && value > 0 ? "+" + text : text;
  }

  function stakeFrom(input) {
    const value = parseFloat(input.value);
    return Number.isFinite(value) && value > 0 ? value : null;
  }

  function wirePickStakes() {
    document.querySelectorAll(".stake-input").forEach(function (input) {
      const cell = input.closest("tr").querySelector(".returns");
      const odds = parseFloat(input.dataset.odds);
      input.addEventListener("input", function () {
        const stake = stakeFrom(input);
        cell.textContent = stake === null ? "" : money(stake * odds);
      });
    });
  }

  function option(text, value) {
    const element = document.createElement("option");
    element.textContent = text;
    element.value = String(value);
    return element;
  }

  function wireCalculator() {
    const panel = document.getElementById("calculator");
    const source = document.getElementById("calculator-data");
    if (!panel || !source) {
      return;
    }
    const games = JSON.parse(source.textContent);
    const threshold = parseFloat(panel.dataset.threshold);
    const match = document.getElementById("calc-match");
    const outcome = document.getElementById("calc-outcome");
    const odds = document.getElementById("calc-odds");
    const stake = document.getElementById("calc-stake");
    const show = function (id, text) {
      document.getElementById(id).textContent = text;
    };

    let league = null;
    let group = null;
    games.forEach(function (game, index) {
      if (game.league !== league) {
        league = game.league;
        group = document.createElement("optgroup");
        group.label = league;
        match.appendChild(group);
      }
      group.appendChild(option(game.match, index));
    });

    function selected() {
      return games[Number(match.value)].outcomes[Number(outcome.value)];
    }

    function fillOutcomes() {
      outcome.replaceChildren();
      games[Number(match.value)].outcomes.forEach(function (item, index) {
        outcome.appendChild(option(item.name, index));
      });
      fillOdds();
    }

    function fillOdds() {
      const listed = selected().odds;
      odds.value = listed ? listed.toFixed(2) : "";
      update();
    }

    function update() {
      const chosen = selected();
      const price = parseFloat(odds.value);
      const amount = stakeFrom(stake);
      show("calc-chance", percent(chosen.chance, false));
      show("calc-fair", (1 / chosen.chance).toFixed(2));
      if (!(price > 1) || amount === null) {
        ["calc-returns", "calc-profit", "calc-edge", "calc-expected"].forEach(function (id) {
          show(id, "n/a");
        });
        show("calc-verdict", "Enter the odds and a stake to see the returns.");
        return;
      }
      const edge = chosen.chance * price - 1;
      show("calc-returns", money(amount * price));
      show("calc-profit", money(amount * (price - 1)));
      show("calc-edge", percent(edge, true));
      show("calc-expected", (edge >= 0 ? "+" : "") + money(amount * edge));
      const needed = ((1 + threshold) / chosen.chance).toFixed(2);
      show(
        "calc-verdict",
        edge >= threshold
          ? "A value bet by the model's rule: the odds beat its fair odds by at least " +
              percent(threshold, false) + "."
          : "Not a value bet: the model would want odds of at least " + needed + "."
      );
    }

    match.addEventListener("change", fillOutcomes);
    outcome.addEventListener("change", fillOdds);
    odds.addEventListener("input", update);
    stake.addEventListener("input", update);
    fillOutcomes();
  }

  wirePickStakes();
  wireCalculator();
})();
