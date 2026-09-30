#!/usr/bin/env python3
"""
Evaluation der Metadaten-Extraktion aus Bauplänen (Title, Date, ID, Scale).

Zwei Sichten in einem Lauf:
  1. STRIKT : Normalisierter Vergleich mit der Groundtruth (kein LLM).
  2. FAIR   : Wie strikt, zusätzlich entscheidet ein LLM-Judge (Uni-API, Thinking maximal)
              bei Nichttreffern für Titel/ID. Datum & Maßstab werden deterministisch geprüft.
              Urteile: JA = 1 Punkt, TEILWEISE = 0.5, NEIN = 0.

Groundtruth (.xlsx, erstes Blatt, oder .csv) mit Spalten:
    File Name | Title | Date | ID | Scale | Datatype   (Datatype wird ignoriert)
  - Mehrere gültige Lösungen in einer Zelle mit "|" trennen (z.B. 1:100|1:50).
  - Nicht vorhanden: "Not found" oder leer.

Die API-Zugangsdaten stehen in der .env neben dem Skript:
    HTW_API_KEY=...
    HTW_BASE_URL=...

Aufruf:
    python evaluate_all.py
    python evaluate_all.py --results results_api.txt --gt groundtruth.xlsx

Benötigt: pip install openpyxl openai python-dotenv
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

FIELDS = ["Title", "Date", "ID", "Scale"]
JUDGE_FIELDS = {"Title"}  # nur hier ist ein LLM-Urteil sinnvoll (ID/Datum/Maßstab bleiben strikt)
KNOWN_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".pdf")
NOT_FOUND = {"", "not found", "notfound", "n/a", "na", "none", "null", "-", "--",
             "unknown", "nicht gefunden", "nicht vorhanden", "kein", "keine"}
GT_ALIASES = {
    "filename": "filename", "file name": "filename", "file": "filename", "datei": "filename", "dateiname": "filename",
    "title": "Title", "titel": "Title",
    "date": "Date", "datum": "Date",
    "id": "ID", "nummer": "ID", "plannummer": "ID",
    "scale": "Scale", "maßstab": "Scale", "massstab": "Scale", "masstab": "Scale",
}
MONTHS = {
    "januar": 1, "jan": 1, "january": 1, "februar": 2, "feb": 2, "february": 2,
    "märz": 3, "maerz": 3, "mär": 3, "mar": 3, "march": 3, "april": 4, "apr": 4,
    "mai": 5, "may": 5, "juni": 6, "jun": 6, "june": 6, "juli": 7, "jul": 7, "july": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "oktober": 10, "okt": 10,
    "oct": 10, "october": 10, "november": 11, "nov": 11, "dezember": 12, "dez": 12,
    "dec": 12, "december": 12,
}


# ============================================================
# Hilfsfunktionen
# ============================================================

def is_empty(value: str) -> bool:
    return value.strip().strip(".").strip().lower() in NOT_FOUND


def file_key(name: str) -> str:
    name = name.strip()
    if name.lower().endswith(KNOWN_EXT):
        name = Path(name).stem
    return name.lower()


# ============================================================
# Normalisierung
# ============================================================

def norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_id(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    return re.sub(r"[\s\-_/.,:;]", "", s)


def norm_date(s: str) -> str:
    """Gibt ISO-artigen String zurück (YYYY-MM-DD, YYYY-MM oder YYYY); sonst normalisierten Text."""
    t = unicodedata.normalize("NFKC", s).casefold().strip()

    def year(y: str) -> int:
        y_int = int(y)
        if len(y) == 2:
            return 2000 + y_int if y_int <= 30 else 1900 + y_int
        return y_int

    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", t)
    if m:
        return f"{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}"
    m = re.search(r"\b(\d{1,2})\s*[.\-/ ]\s*(\d{1,2})\s*[.\-/ ]\s*(\d{4}|\d{2})\b", t)
    if m:
        return f"{year(m[3]):04d}-{int(m[2]):02d}-{int(m[1]):02d}"
    m = re.search(r"\b(\d{1,2})\.?\s*([a-zäöü]{3,9})\.?\s*(\d{4}|\d{2})\b", t)
    if m and m[2] in MONTHS:
        return f"{year(m[3]):04d}-{MONTHS[m[2]]:02d}-{int(m[1]):02d}"
    m = re.search(r"\b([a-zäöü]{3,9})\.?\s*(\d{4})\b", t)
    if m and m[1] in MONTHS:
        return f"{int(m[2]):04d}-{MONTHS[m[1]]:02d}"
    m = re.search(r"\b(\d{1,2})\s*[./\-]\s*(\d{4})\b", t)
    if m and 1 <= int(m[1]) <= 12:
        return f"{int(m[2]):04d}-{int(m[1]):02d}"
    m = re.search(r"\b(1[6-9]\d\d|20\d\d)\b", t)
    if m:
        return m[1]
    return re.sub(r"\s+", "", t)


def norm_scale(s: str) -> frozenset:
    """Extrahiert alle Verhältnisse a:b bzw. a/b -> {'1:100', ...}."""
    t = unicodedata.normalize("NFKC", s)
    found = {f"{int(a)}:{int(b)}" for a, b in re.findall(r"(\d+)\s*[:/]\s*(\d+)", t)}
    if found:
        return frozenset(found)
    return frozenset({re.sub(r"\s+", "", t.casefold())})


# ============================================================
# Vergleich  ->  Score 0..1
# ============================================================

def sim_title(a: str, b: str) -> float:
    na, nb = norm_title(a), norm_title(b)
    if not na or not nb:
        return 0.0
    r1 = SequenceMatcher(None, na, nb).ratio()
    r2 = SequenceMatcher(None, " ".join(sorted(na.split())), " ".join(sorted(nb.split()))).ratio()
    return max(r1, r2)


def sim_id(a: str, b: str) -> float:
    na, nb = norm_id(a), norm_id(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def sim_date(a: str, b: str) -> float:
    return 1.0 if norm_date(a) == norm_date(b) else 0.0


def sim_scale(a: str, b: str) -> float:
    return 1.0 if norm_scale(a) & norm_scale(b) else 0.0


SIM = {"Title": sim_title, "Date": sim_date, "ID": sim_id, "Scale": sim_scale}


# ============================================================
# Einlesen
# ============================================================

def parse_results(path: Path):
    """Liest results_api.txt. Rückgabe: {file_key: {field: [werte pro Seite]}}, Statistiken."""
    text = path.read_text(encoding="utf-8", errors="replace")
    entry_re = re.compile(r"^FILE:\s*(.+?)\n-{5,}\n(.*?)\n={5,}", re.S | re.M)
    entries, dupes, errors = {}, 0, 0

    for m in entry_re.finditer(text):
        header, body = m.group(1).strip(), m.group(2)
        hm = re.match(r"^(.*?)(?:\s*\(Page\s+(\d+)/(\d+)\))?$", header)
        fname, page = hm.group(1).strip(), int(hm.group(2) or 1)
        key = (file_key(fname), page)

        if body.strip().lower().startswith("error"):
            errors += 1
        fields = {}
        for line in body.splitlines():
            line = line.replace("*", "").strip()
            fm = re.match(r"^(Title|Date|ID|Scale)\s*:\s*(.*)$", line, re.I)
            if fm:
                canon = {f.lower(): f for f in FIELDS}[fm.group(1).lower()]
                fields[canon] = fm.group(2).strip()
        if key in entries:
            dupes += 1  # Datei wurde mehrfach verarbeitet (Append-Modus) -> letzter Lauf gewinnt
        entries[key] = fields

    by_file = defaultdict(lambda: {f: [] for f in FIELDS})
    for (fk, _page), fields in sorted(entries.items()):
        for f in FIELDS:
            v = fields.get(f, "")
            if not is_empty(v):
                by_file[fk][f].append(v)
    return by_file, {"entries": len(entries), "duplicates": dupes, "errors": errors}


def _cell_to_str(v) -> str:
    """Excel-Zelle -> String (Datums-Zellen als TT.MM.JJJJ, ganze Zahlen ohne '.0')."""
    if v is None:
        return ""
    if isinstance(v, datetime):
        return v.strftime("%d.%m.%Y")
    if hasattr(v, "strftime"):  # datetime.date
        return v.strftime("%d.%m.%Y")
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def _read_gt_rows(path: Path):
    """Liest .xlsx/.xlsm (erstes Blatt) oder .csv -> (Spaltennamen, Liste von Zeilen-Dicts)."""
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        try:
            from openpyxl import load_workbook
        except ImportError:
            sys.exit("Für .xlsx wird openpyxl benötigt:  pip install openpyxl")
        wb = load_workbook(path, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            sys.exit("Groundtruth-Excel ist leer.")
        header = [_cell_to_str(h) for h in rows[0]]
        data = [dict(zip(header, [_cell_to_str(c) for c in r])) for r in rows[1:]]
        return header, data

    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    first = raw.splitlines()[0] if raw else ""
    delim = ";" if first.count(";") >= first.count(",") else ","
    reader = csv.DictReader(raw.splitlines(), delimiter=delim)
    return list(reader.fieldnames or []), list(reader)


def parse_groundtruth(path: Path):
    header, rows = _read_gt_rows(path)

    colmap = {}
    for col in header:
        key = re.sub(r"[\s_]+", " ", col.strip().lower())
        canon = GT_ALIASES.get(key)
        if canon and canon not in colmap:
            colmap[canon] = col  # Spalten wie "Datatype" werden ignoriert
    missing = [c for c in ["filename"] + FIELDS if c not in colmap]
    if missing:
        sys.exit(f"Groundtruth: Spalten fehlen oder sind nicht erkannt: {missing}. "
                 f"Gefunden: {header}")

    gt = {}
    for row in rows:
        fname = (row.get(colmap["filename"]) or "").strip()
        if not fname:
            continue
        entry = {}
        for f in FIELDS:
            cell = (row.get(colmap[f]) or "").strip()
            alts = [a.strip() for a in cell.split("|")]
            entry[f] = [a for a in alts if not is_empty(a)]
        gt[file_key(fname)] = entry
    return gt


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1



# ============================================================
# LLM-Judge & Auswertung
# ============================================================

JUDGE_MODEL = "qwen3.8-27b"   # Judge-Modell der Uni-API (per --model überschreibbar)

STRICT_THR = {"Title": 0.85, "ID": 1.0, "Date": 1.0, "Scale": 1.0}
VERDICT_RANK = {"NEIN": 0, "TEILWEISE": 1, "JA": 2}
VERDICT_SCORE = {"JA": 1.0, "TEILWEISE": 0.5, "NEIN": 0.0}

RULES = {
    "Title": (
        "JA: gleicher Plantitel bzw. gleiche Bedeutung. Unkritisch sind Schreibweise, "
        "Groß-/Kleinschreibung, Abkürzungen, Wortreihenfolge, kleine Tippfehler sowie "
        "fehlende oder zusätzliche Nebenzeilen (z.B. Untertitel, Firmenname). "
        "TEILWEISE: KI-Wert enthält nur einen Teil des Referenztitels (oder umgekehrt), "
        "es ist aber klar derselbe Plan. "
        "NEIN: anderes Objekt, anderer Ort oder anderer Inhalt."
    ),
    "ID": (
        "JA: gleiche Kennung. Unkritisch sind Trennzeichen, Leerzeichen, Groß-/Kleinschreibung "
        "und Präfixe wie 'Nr.', 'Bl.', 'Zeichnungs-Nr.'. "
        "TEILWEISE: gleiche Kennung, aber ein Zusatzteil fehlt oder ist überflüssig. "
        "NEIN: andere Ziffern oder Buchstaben in der Kennung."
    ),
    "Date": "JA: gleiches Datum, Format egal. NEIN: anderer Tag, Monat oder anderes Jahr.",
    "Scale": "JA: gleiches Maßstabsverhältnis. NEIN: anderes Verhältnis.",
}


# ============================================================
# Deterministische Regeln für Datum / Maßstab
# ============================================================

def det_verdict(field: str, gts: list, pred: str):
    """Liefert (verdict, reason) für Date/Scale ohne LLM; None wenn nicht zuständig."""
    if field == "Date":
        best = ("NEIN", "Datum weicht ab")
        pn = norm_date(pred)
        for g in gts:
            gn = norm_date(g)
            if gn == pn:
                return "JA", "Datum identisch"
            iso = re.compile(r"^\d{4}(-\d{2}){0,2}$")
            if iso.match(gn) and iso.match(pn) and (gn.startswith(pn) or pn.startswith(gn)):
                best = ("TEILWEISE", "Datum nur unterschiedlich genau (z.B. Monat/Jahr statt Tag)")
        return best
    if field == "Scale":
        ps = norm_scale(pred)
        for g in gts:
            if norm_scale(g) & ps:
                return "JA", "Maßstab identisch"
        return "NEIN", "Maßstab weicht ab"
    return None


# ============================================================
# LLM-Schiedsrichter
# ============================================================

def build_prompt(field: str, gts: list, pred: str) -> str:
    return (
        "Du bist ein fairer Prüfer für die automatische Extraktion von Metadaten aus "
        "Bauplänen. Ein Mensch hat die Referenz erstellt, eine KI hat den Wert aus dem "
        "(teils handschriftlichen) Plan gelesen. Entscheide, ob der KI-Wert inhaltlich mit "
        "einem der Referenzwerte übereinstimmt.\n\n"
        f"Feld: {field}\n"
        f"Regeln: {RULES[field]}\n"
        "Referenzwerte (jeder einzelne ist gültig):\n"
        + "\n".join(f"- {g}" for g in gts)
        + f"\nKI-Wert: {pred}\n\n"
        'Antworte ausschließlich als JSON: {"verdict": "JA" oder "TEILWEISE" oder "NEIN", '
        '"reason": "ein kurzer Satz"}'
    )


def parse_verdict(text: str):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            v = str(obj.get("verdict", "")).strip().upper()
            if v in VERDICT_RANK:
                return v, str(obj.get("reason", "")).strip()
        except json.JSONDecodeError:
            pass
    m = re.search(r"\b(TEILWEISE|NEIN|JA)\b", text.upper())
    if m:
        return m.group(1), text[:200].replace("\n", " ")
    return None, f"Antwort nicht lesbar: {text[:100]}"


def llm_verdict(client, model, field, gts, pred, retries=2):
    """Fragt den Judge im Thinking-Modus (maximale Leistung). Rückgabe: (verdict, reason)."""
    prompt = build_prompt(field, gts, pred)
    use_extras = True   # Thinking-Parameter; fallen automatisch weg, falls der Server sie ablehnt
    last_err = ""
    for _ in range(retries + 2):
        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6, top_p=0.95,   # Qwen3: im Thinking-Modus kein Greedy-Decoding
            max_tokens=16000,              # Reserve für den Denkblock
        )
        if use_extras:
            kwargs["reasoning_effort"] = "high"                                         # Ollama / OpenAI-Stil
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}  # vLLM / SGLang
        try:
            r = client.chat.completions.create(**kwargs)
            v, reason = parse_verdict(r.choices[0].message.content)
            if v:
                # Plausibilitätsschutz: Modelle sagen gern zu großzügig "JA"
                if field == "ID" and v != "NEIN" and max(SIM["ID"](g, pred) for g in gts) < 0.5:
                    return "NEIN", f"(Plausibilitätscheck: Zeichenähnlichkeit <0.5) {reason}"
                return v, reason
            last_err = reason
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            use_extras = False  # nächster Versuch ohne Spezialparameter
            time.sleep(1)
    return None, f"LLM-Fehler: {last_err}"


# ============================================================
# Auswertung
# ============================================================

def evaluate(gt, preds, client, model, judge_fields):
    cache = {}
    details = []
    n_llm_calls = 0

    for fk, g in gt.items():
        p = preds.get(fk)
        for f in FIELDS:
            gts = g[f]
            cands = p[f] if p else []
            strict, verdict, reason, cell = "", "", "", ""

            if not gts and not cands:
                cell, strict = "TN", "TN"
                reason = "beide leer"
            elif not gts and cands:
                cell, strict = "FP", "FP"
                reason = "KI liefert Wert, Referenz ist 'Not found'"
            elif gts and not cands:
                cell, strict = "FN", "FN"
                reason = "KI hat nichts erkannt"
            else:
                best_sim = max(SIM[f](a, c) for a in gts for c in cands)
                if best_sim >= STRICT_THR[f]:
                    cell, strict, reason = "MATCH_STRICT", "TP", "strikter Treffer"
                else:
                    strict = "FP+FN"
                    best_v, best_r = "NEIN", "kein Treffer"
                    for c in cands:
                        det = det_verdict(f, gts, c)
                        if det:
                            v, r = det
                        elif f in judge_fields:
                            key = (f, tuple(gts), c)
                            if key not in cache:
                                n_llm_calls += 1
                                cache[key] = llm_verdict(client, model, f, gts, c)
                            v, r = cache[key]
                            if v is None:  # LLM-Ausfall -> strikt bleiben
                                v, r = "NEIN", r
                        else:
                            v, r = "NEIN", "nicht vom LLM geprüft (Feld nicht in --judge-fields)"
                        if VERDICT_RANK[v] > VERDICT_RANK[best_v] or best_r == "kein Treffer":
                            best_v, best_r = v, r
                    verdict, reason = best_v, best_r
                    cell = {"JA": "MATCH_LLM", "TEILWEISE": "PARTIAL", "NEIN": "WRONG"}[verdict]

            details.append({
                "file": fk, "field": f,
                "groundtruth": " | ".join(gts) if gts else "Not found",
                "prediction": " || ".join(cands) if cands else "Not found",
                "strict": strict, "llm_verdict": verdict, "cell": cell, "reason": reason,
            })
        print(f"  [{len(set(d['file'] for d in details))}/{len(gt)}] {fk} fertig "
              f"(LLM-Aufrufe bisher: {n_llm_calls})", flush=True)
    return details, n_llm_calls


def aggregate(details):
    per_field = {f: {"TN": 0, "FP": 0, "FN": 0, "MATCH_STRICT": 0,
                     "MATCH_LLM": 0, "PARTIAL": 0, "WRONG": 0} for f in FIELDS}
    for d in details:
        per_field[d["field"]][d["cell"]] += 1
    return per_field


def field_metrics(c):
    """Kennzahlen aus den Zell-Kategorien (strikt und mit Judge)."""
    n = sum(c.values())
    strict_ok = c["TN"] + c["MATCH_STRICT"]
    lenient_ok = strict_ok + c["MATCH_LLM"]
    soft = lenient_ok + 0.5 * c["PARTIAL"]
    wrong_like = c["PARTIAL"] + c["WRONG"]

    # strikt: nur MATCH_STRICT ist TP; jeder andere Wert-Fehler zählt als FP und FN
    s_tp = c["MATCH_STRICT"]
    s_fp = c["FP"] + c["MATCH_LLM"] + wrong_like
    s_fn = c["FN"] + c["MATCH_LLM"] + wrong_like
    s_p, s_r, s_f1 = prf(s_tp, s_fp, s_fn)

    # fair: MATCH_LLM zählt zusätzlich als TP; TEILWEISE bleibt FP + FN
    tp = c["MATCH_STRICT"] + c["MATCH_LLM"]
    fp = c["FP"] + wrong_like
    fn = c["FN"] + wrong_like
    p, r, f1 = prf(tp, fp, fn)

    div = n if n else 1
    return {
        "N": n,
        "strict_acc": strict_ok / div, "s_precision": s_p, "s_recall": s_r, "s_f1": s_f1,
        "lenient_acc": lenient_ok / div, "soft_score": soft / div,
        "precision": p, "recall": r, "f1": f1,
    }


COLS = ["strict_acc", "s_precision", "s_recall", "s_f1",
        "lenient_acc", "soft_score", "precision", "recall", "f1"]
CSV_HEADER = ["Field", "N", "Strict_Acc", "Strict_Precision", "Strict_Recall", "Strict_F1",
              "Fair_Acc(JA)", "Fair_SoftScore", "Fair_Precision", "Fair_Recall", "Fair_F1"]


def main():
    ap = argparse.ArgumentParser(description="Evaluation: strikt + fairer LLM-Judge (Uni-API)")
    ap.add_argument("--results", default=None, help="Pfad zur Results-TXT (sonst Abfrage)")
    ap.add_argument("--gt", default=None, help="Pfad zur Groundtruth .xlsx/.csv (sonst Abfrage)")
    ap.add_argument("--model", default=JUDGE_MODEL, help=f"Judge-Modell (Standard: {JUDGE_MODEL})")
    ap.add_argument("--out-dir", default="eval_output")
    args = ap.parse_args()
    judge_fields = {"Title", "ID"}   # nur hier ist ein LLM-Urteil sinnvoll

    def ask_path(prompt):
        while True:
            val = input(prompt).strip().strip("\"'")
            if val:
                return val
            print("  Bitte einen Pfad eingeben.")

    results_path = Path((args.results or ask_path("Pfad zur Results-Datei (z.B. results_api.txt): ")).strip().strip("\"'"))
    gt_path = Path((args.gt or ask_path("Pfad zur Groundtruth-Datei (.xlsx/.csv): ")).strip().strip("\"'"))
    for p in (results_path, gt_path):
        if not p.exists():
            sys.exit(f"Datei nicht gefunden: {p}")

    # ---------- API-Client (Uni-API aus .env) ----------
    from openai import OpenAI
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent / ".env")
    except ImportError:
        pass
    api_key, base_url = os.getenv("HTW_API_KEY"), os.getenv("HTW_BASE_URL")
    if not api_key or not base_url:
        sys.exit("HTW_API_KEY / HTW_BASE_URL nicht gefunden (.env neben dem Skript?)")
    model_name = args.model
    client = OpenAI(base_url=base_url, api_key=api_key)

    print(f"Teste API-Verbindung ({model_name} @ {base_url}) ...")
    try:
        client.chat.completions.create(model=model_name, max_tokens=50,
                                       messages=[{"role": "user", "content": "Antworte mit OK."}])
    except Exception as e:  # noqa: BLE001
        sys.exit(f"API nicht erreichbar oder Modellname falsch: {e}")
    print("  OK")

    gt = parse_groundtruth(gt_path)
    preds, stats = parse_results(results_path)
    missing_in_results = [k for k in gt if k not in preds]
    not_in_gt = [k for k in preds if k not in gt]

    judge_txt = f"{model_name} @ {base_url} (Thinking: maximal)"
    print(f"\n{len(gt)} Dokumente | Judge: {judge_txt}")
    details, n_calls = evaluate(gt, preds, client, model_name, judge_fields)

    # ---------- Kennzahlen ----------
    per_field = aggregate(details)
    metrics = {f: field_metrics(per_field[f]) for f in FIELDS}
    tot = {k: sum(per_field[f][k] for f in FIELDS) for k in per_field[FIELDS[0]]}
    overall = field_metrics(tot)
    macro = {k: sum(metrics[f][k] for f in FIELDS) / len(FIELDS) for k in COLS}

    def row(name, m):
        return [name, m.get("N", "")] + [f"{m[k]:.4f}" for k in COLS]

    rows = [row(f, metrics[f]) for f in FIELDS]
    rows.append(row("OVERALL (micro)", overall))
    rows.append(row("MACRO (Ø Felder)", macro))

    docs_ok_strict = docs_ok_fair = 0
    by_file = defaultdict(list)
    for d in details:
        by_file[d["file"]].append(d["cell"])
    for cells in by_file.values():
        docs_ok_strict += all(c in ("TN", "MATCH_STRICT") for c in cells)
        docs_ok_fair += all(c in ("TN", "MATCH_STRICT", "MATCH_LLM") for c in cells)

    upgraded = sum(1 for d in details if d["cell"] in ("MATCH_LLM", "PARTIAL"))
    llm_errors = sum(1 for d in details if "LLM-Fehler" in d["reason"])

    # ---------- Ausgabe ----------
    out = []
    out.append("=" * 96)
    out.append(f"EVALUATION  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    out.append(f"Results: {results_path}   |   Groundtruth: {gt_path}")
    out.append(f"Judge: {judge_txt}")
    out.append("=" * 96)
    out.append(f"Dokumente in Groundtruth: {len(gt)} | Einträge in Results: {stats['entries']} "
               f"(Duplikate: {stats['duplicates']}, Fehlerzeilen: {stats['errors']})")
    if missing_in_results:
        out.append(f"WARNUNG: {len(missing_in_results)} GT-Dokumente fehlen in Results: "
                   f"{', '.join(missing_in_results[:10])}{' ...' if len(missing_in_results) > 10 else ''}")
    if not_in_gt:
        out.append(f"Hinweis: {len(not_in_gt)} Results-Dateien ohne GT-Eintrag (ignoriert)")
    out.append(f"LLM-Aufrufe: {n_calls} | durch Judge aufgewertet (JA/TEILWEISE): {upgraded} | "
               f"LLM-Fehler: {llm_errors}")
    out.append("")

    f1 = "{:<20}{:>5}{:>12}{:>11}{:>9}{:>8}"
    out.append("--- 1) STRIKT (ohne LLM) ---")
    out.append(f1.format("Feld", "N", "Accuracy", "Precision", "Recall", "F1"))
    for r_ in rows:
        out.append(f1.format(r_[0], r_[1], r_[2], r_[3], r_[4], r_[5]))
    out.append("")
    f2 = "{:<20}{:>5}{:>12}{:>12}{:>11}{:>9}{:>8}"
    out.append("--- 2) FAIR (mit LLM-Judge) ---")
    out.append(f2.format("Feld", "N", "Acc(JA)", "SoftScore", "Precision", "Recall", "F1"))
    for r_ in rows:
        out.append(f2.format(r_[0], r_[1], r_[6], r_[7], r_[8], r_[9], r_[10]))
    out.append("")
    out.append(f">>> GESAMT-ÜBEREINSTIMMUNG (Soft-Score, fair): {overall['soft_score'] * 100:.1f} %")
    out.append(f">>> Strikt (ohne LLM):                        {overall['strict_acc'] * 100:.1f} %")
    out.append(f">>> Fair, JA zählt voll:                      {overall['lenient_acc'] * 100:.1f} %")
    out.append(f">>> Pläne mit allen 4 Feldern korrekt:        strikt {docs_ok_strict}/{len(gt)}, "
               f"fair {docs_ok_fair}/{len(gt)}")
    out.append("")
    out.append("Legende: TN = korrektes 'Not found' | Strikt: nur normalisierte Treffer | Fair: zusätzlich "
               "Judge-JA | SoftScore: JA=1, TEILWEISE=0.5, NEIN=0.")
    out.append("Ein falscher Wert zählt als FP und FN. TEILWEISE zählt bei Precision/Recall/F1 als FP und FN. "
               "Begründungen je Fall: details_*.csv (Spalte 'reason').")
    summary = "\n".join(out)
    print("\n" + summary)

    # ---------- Speichern ----------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / f"metrics_{stamp}.txt").write_text(summary, encoding="utf-8")
    with open(out_dir / f"metrics_{stamp}.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(CSV_HEADER)
        w.writerows(rows)
    with open(out_dir / f"details_{stamp}.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(details[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(details)
    print(f"\nGespeichert in: {out_dir.resolve()}")
    print(f"  metrics_{stamp}.txt / .csv   (Zusammenfassung)")
    print(f"  details_{stamp}.csv          (jede Einzelentscheidung inkl. LLM-Begründung)")


if __name__ == "__main__":
    main()