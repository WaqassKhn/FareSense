const form = document.querySelector("#fare-form");

if (form) {
  const fieldNames = [
    "airline",
    "classs",
    "source_city",
    "destination_city",
    "departure_time",
    "arrival_time",
    "stops",
    "duration",
    "days_left",
  ];

  const fields = Object.fromEntries(
    fieldNames.map((name) => [name, form.elements.namedItem(name)]),
  );

  const text = (select) => {
    const option = select.options[select.selectedIndex];
    return select.value && option ? option.text : "";
  };

  const formatDuration = (value) => {
    const duration = Number.parseFloat(value);
    if (!Number.isFinite(duration)) return "";

    let hours = Math.floor(duration);
    let minutes = Math.round((duration - hours) * 60);
    if (minutes === 60) {
      hours += 1;
      minutes = 0;
    }

    return `${hours} hr ${minutes} min`;
  };

  const formatDepartureDate = (days) => {
    const count = Number.parseInt(days, 10);
    if (!Number.isInteger(count)) return "";

    const departure = new Date();
    departure.setHours(12, 0, 0, 0);
    departure.setDate(departure.getDate() + count);
    return departure.toLocaleDateString(undefined, {
      weekday: "short",
      day: "numeric",
      month: "short",
      year: "numeric",
    });
  };

  const updateRouteValidation = () => {
    const sameCity =
      fields.source_city.value &&
      fields.destination_city.value &&
      fields.source_city.value === fields.destination_city.value;
    const message = sameCity
      ? "Departure and destination cities must be different."
      : "";

    fields.destination_city.setCustomValidity(message);
    document.querySelector("#route-error").textContent = message;
    fields.destination_city.closest(".field").classList.toggle("has-error", sameCity);
    return !sameCity;
  };

  const updateProgress = () => {
    const completed = fieldNames.filter((name) => {
      const field = fields[name];
      return field.value !== "" && field.checkValidity();
    }).length;
    const percentage = Math.round((completed / fieldNames.length) * 100);

    document.querySelector("#progress-label").textContent =
      `${completed} of ${fieldNames.length} completed`;
    document.querySelector("#progress-bar").style.width = `${percentage}%`;

    fieldNames.forEach((name) => {
      const field = fields[name];
      field.classList.toggle("has-value", field.value !== "");
    });
  };

  const updateSummary = () => {
    const duration = formatDuration(fields.duration.value);
    const days = Number.parseInt(fields.days_left.value, 10);
    const departureDate = formatDepartureDate(fields.days_left.value);

    document.querySelector("#summary-source").textContent =
      text(fields.source_city) || "—";
    document.querySelector("#summary-destination").textContent =
      text(fields.destination_city) || "—";
    document.querySelector("#summary-airline").textContent =
      text(fields.airline) || "Not selected";
    document.querySelector("#summary-class").textContent =
      text(fields.classs) || "Not selected";
    document.querySelector("#summary-stops").textContent =
      text(fields.stops) || "Not selected";
    document.querySelector("#summary-duration").textContent =
      duration || "Not entered";
    document.querySelector("#summary-days").textContent =
      departureDate ? `${days} days · ${departureDate}` : "Not entered";

    const departureTime = text(fields.departure_time);
    const arrivalTime = text(fields.arrival_time);
    document.querySelector("#summary-schedule").textContent =
      departureTime || arrivalTime
        ? `${departureTime || "—"} → ${arrivalTime || "—"}`
        : "Not selected";

    document.querySelector("#duration-feedback").textContent = duration
      ? `That is approximately ${duration} total travel time.`
      : "Enter a duration to see hours and minutes.";

    document.querySelector("#days-feedback").textContent = departureDate
      ? `That is approximately ${departureDate} if counting from today.`
      : "The training data supports flights departing 1–49 days from now.";
  };

  const refreshInterface = () => {
    updateRouteValidation();
    updateProgress();
    updateSummary();
  };

  form.addEventListener("input", refreshInterface);
  form.addEventListener("change", refreshInterface);

  form.addEventListener("submit", (event) => {
    updateRouteValidation();
    if (!form.checkValidity()) {
      event.preventDefault();
      form.reportValidity();
      return;
    }

    const button = document.querySelector("#submit-button");
    button.classList.add("is-loading");
    button.querySelector("span").textContent = "Calculating your estimate…";
  });

  refreshInterface();
}
