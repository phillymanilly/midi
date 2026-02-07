# MIDI Generator

Ein erweiterter MIDI-Generator in Python, der stilisierte Akkordfolgen erzeugt und optional Bass- und Topline-Spuren hinzufügt.

## Voraussetzungen

- Python 3.9+
- Abhängigkeiten: `pretty_midi`

Installation:

```bash
pip install pretty_midi
```

## Nutzung

```bash
python3 midi_generator.py --artist "Drake" --out out --n 10 --bars 8 --bass --topline
```

### Optionen

- `--artist`: Künstlername zur Auswahl eines Stil-Profils (z. B. "Drake", "Gunna")
- `--out`: Zielordner für MIDI-Dateien (Standard: `out`)
- `--n`: Anzahl der zu erzeugenden Dateien
- `--bars`: Anzahl der Takte pro Datei
- `--seed`: Basis-Seed für reproduzierbare Ergebnisse
- `--bass`: Fügt eine Bass-Spur hinzu
- `--topline`: Fügt eine Topline-Spur hinzu

Die erzeugten Dateien liegen danach im angegebenen Ordner und können in einer DAW oder einem MIDI-Player geöffnet werden.

## UI (lokale Vorschau)

Unter `ui/index.html` gibt es eine einfache Web-Oberfläche, die dir passende CLI-Kommandos zusammenstellt. Öffne sie z. B. mit:

```bash
python3 -m http.server 8000 --directory ui
```

Dann im Browser `http://localhost:8000` öffnen.
