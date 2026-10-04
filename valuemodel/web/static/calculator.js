// Stake inputs for the paper bets table and the bet slip on the fixtures page.
// The slip follows picks.accumulator: odds and the model's chances multiply,
// and only one leg is allowed per match.
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

  function show(id, text) {
    document.getElementById(id).textContent = text;
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

  function cell(text, className) {
    const element = document.createElement("td");
    element.textContent = text;
    if (className) {
      element.className = className;
    }
    return element;
  }

  function wireSlip() {
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
    const legsBody = document.getElementById("slip-legs");
    const legs = [];

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

    function selectedGame() {
      return games[Number(match.value)];
    }

    function selectedOutcome() {
      return selectedGame().outcomes[Number(outcome.value)];
    }

    function fillOutcomes() {
      outcome.replaceChildren();
      selectedGame().outcomes.forEach(function (item, index) {
        outcome.appendChild(option(item.name, index));
      });
      fillOdds();
    }

    function fillOdds() {
      const listed = selectedOutcome().odds;
      odds.value = listed ? listed.toFixed(2) : "";
      preview();
    }

    function preview() {
      const chosen = selectedOutcome();
      const price = parseFloat(odds.value);
      let text = "Model chance " + percent(chosen.chance, false) + ", fair odds " +
        (1 / chosen.chance).toFixed(2) + ".";
      if (price > 1) {
        const edge = chosen.chance * price - 1;
        text += " At " + price.toFixed(2) + " the edge is " + percent(edge, true) +
          (edge >= threshold ? ", a value bet." : ", not a value bet.");
      }
      show("calc-preview", text);
    }

    function hasOdds(leg) {
      return leg.odds > 1;
    }

    // Legs loaded from a likeliest slip arrive without odds, so only legs added
    // by hand must already have them.
    function addLeg(leg, needsOdds) {
      if (needsOdds && !hasOdds(leg)) {
        show("slip-message", "Enter decimal odds greater than 1 before adding the bet.");
        return;
      }
      if (legs.some(function (existing) { return existing.key === leg.key; })) {
        show("slip-message", leg.match.split(", ").pop() +
          " is already on the slip. Bets on the same match are linked, so the slip takes one per match.");
        return;
      }
      show("slip-message", "");
      legs.push(leg);
      render();
    }

    function edgeText(leg, element) {
      if (!hasOdds(leg)) {
        element.textContent = "n/a";
        element.className = "num";
        return;
      }
      const edge = leg.chance * leg.odds - 1;
      element.textContent = percent(edge, true);
      element.className = "num " + (edge >= 0 ? "positive" : "negative");
    }

    function oddsInput(leg, edgeCell) {
      const input = document.createElement("input");
      input.type = "number";
      input.min = "1.01";
      input.step = "0.01";
      input.inputMode = "decimal";
      input.className = "slip-odds";
      input.value = hasOdds(leg) ? leg.odds.toFixed(2) : "";
      input.setAttribute("aria-label", "Odds for " + leg.bet + ", " + leg.match);
      input.addEventListener("input", function () {
        leg.odds = parseFloat(input.value);
        edgeText(leg, edgeCell);
        summarise();
      });
      const wrapper = document.createElement("td");
      wrapper.className = "num";
      wrapper.appendChild(input);
      return wrapper;
    }

    function render() {
      legsBody.replaceChildren();
      legs.forEach(function (leg, index) {
        const row = document.createElement("tr");
        const edgeCell = cell("", "num");
        edgeText(leg, edgeCell);
        row.appendChild(cell(leg.match));
        row.appendChild(cell(leg.bet));
        row.appendChild(oddsInput(leg, edgeCell));
        row.appendChild(cell(percent(leg.chance, false), "num"));
        row.appendChild(edgeCell);
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "quiet";
        remove.textContent = "Remove";
        remove.setAttribute("aria-label", "Remove " + leg.bet + ", " + leg.match);
        remove.addEventListener("click", function () {
          legs.splice(index, 1);
          show("slip-message", "");
          render();
        });
        const action = document.createElement("td");
        action.appendChild(remove);
        row.appendChild(action);
        legsBody.appendChild(row);
      });
      document.getElementById("slip-empty").hidden = legs.length > 0;
      summarise();
    }

    function summarise() {
      const amount = stakeFrom(stake);
      ["calc-combined", "calc-returns", "calc-profit", "calc-chance", "calc-fair", "calc-edge",
        "calc-expected"].forEach(function (id) { show(id, "n/a"); });
      show(
        "slip-kind",
        legs.length > 1 ? legs.length + "-leg accumulator, odds" : legs.length ? "Single, odds" : "Bet"
      );
      if (legs.length === 0) {
        show("calc-verdict", "");
        return;
      }
      const chance = legs.reduce(function (total, leg) { return total * leg.chance; }, 1);
      const needed = ((1 + threshold) / chance).toFixed(2);
      show("calc-chance", percent(chance, false));
      show("calc-fair", (1 / chance).toFixed(2));
      if (!legs.every(hasOdds)) {
        show("calc-verdict", "Enter the odds for every leg to see the returns. The model would want combined odds of at least " +
          needed + " for a value bet.");
        return;
      }
      const combined = legs.reduce(function (total, leg) { return total * leg.odds; }, 1);
      const edge = chance * combined - 1;
      show("calc-combined", combined.toFixed(2));
      show("calc-edge", percent(edge, true));
      if (amount === null) {
        show("calc-verdict", "Enter a stake to see the returns.");
        return;
      }
      show("calc-returns", money(amount * combined));
      show("calc-profit", money(amount * (combined - 1)));
      show("calc-expected", (edge >= 0 ? "+" : "") + money(amount * edge));
      show(
        "calc-verdict",
        edge >= threshold
          ? "A value bet by the model's rule: the odds beat its fair odds by at least " +
              percent(threshold, false) + "."
          : "Not a value bet by the model's rule: it would want odds of at least " + needed + "."
      );
    }
    match.addEventListener("change", fillOutcomes);
    outcome.addEventListener("change", fillOdds);
    odds.addEventListener("input", preview);
    stake.addEventListener("input", summarise);
    document.getElementById("calc-add").addEventListener("click", function () {
      const game = selectedGame();
      const chosen = selectedOutcome();
      addLeg({
        key: game.key,
        match: game.match,
        bet: chosen.name,
        odds: parseFloat(odds.value),
        chance: chosen.chance
      }, true);
    });
    document.getElementById("slip-clear").addEventListener("click", function () {
      legs.length = 0;
      show("slip-message", "");
      render();
    });
    document.querySelectorAll(".add-to-slip").forEach(function (button) {
      button.addEventListener("click", function () {
        addLeg({
          key: button.dataset.key,
          match: button.dataset.match.replace(" ,", ","),
          bet: button.dataset.bet,
          odds: parseFloat(button.dataset.odds),
          chance: parseFloat(button.dataset.chance)
        }, true);
        panel.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    });
    document.querySelectorAll(".load-slip").forEach(function (button) {
      button.addEventListener("click", function () {
        legs.length = 0;
        show("slip-message", "");
        JSON.parse(button.dataset.legs).forEach(function (leg) {
          addLeg({ key: leg.key, match: leg.match, bet: leg.bet, odds: leg.odds, chance: leg.chance }, false);
        });
        render();
        panel.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    });
    fillOutcomes();
    render();
  }

  wirePickStakes();
  wireSlip();
})();
