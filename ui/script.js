const form = document.getElementById("command-form");
const output = document.getElementById("command-output");
const copyBtn = document.getElementById("copy-btn");
const copyStatus = document.getElementById("copy-status");

const buildCommand = () => {
  const data = new FormData(form);
  const parts = [
    "python3 midi_generator.py",
    `--artist \"${data.get("artist")}\"`,
    `--out ${data.get("out")}`,
    `--n ${data.get("n")}`,
    `--bars ${data.get("bars")}`,
    `--seed ${data.get("seed")}`,
  ];

  if (data.get("bass")) {
    parts.push("--bass");
  }
  if (data.get("topline")) {
    parts.push("--topline");
  }

  output.textContent = parts.join(" ");
};

form.addEventListener("input", buildCommand);
buildCommand();

copyBtn.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(output.textContent);
    copyStatus.textContent = "Kopiert!";
  } catch (error) {
    copyStatus.textContent = "Kopieren fehlgeschlagen.";
  }
  setTimeout(() => {
    copyStatus.textContent = "";
  }, 2000);
});
