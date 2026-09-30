#!/usr/bin/env python3
"""
Faire Bewertung der Metadaten-Extraktion: strikter Check + lokales LLM als Schiedsrichter.

Ablauf pro Dokument und Feld (Title, Date, ID, Scale):
  1. Strikter Vergleich (wie in evaluate_results.py). Treffer -> fertig, kein LLM-Aufruf.
  2. Kein Treffer, aber beide Seiten haben einen Wert -> Bewertung durch das LLM
     (Urteil: JA / TEILWEISE / NEIN + kurze Begründung).
     Datum und Maßstab werden deterministisch geprüft (LLMs sind bei Zahlen unzuverlässig).
  3. Nur Referenz leer / nur KI leer -> eindeutig (TN / FP / FN), kein LLM nötig.

Score: JA = 1, TEILWEISE = 0.5, NEIN = 0; korrektes "Not found" = 1.

Benötigt evaluate_results.py im selben Ordner.

Aufruf:
    python evaluate_llm_judge.py
    python evaluate_llm_judge.py --results results_api.txt --gt groundtruth.xlsx
    python evaluate_llm_judge.py --api htw                 # Uni-Modell (qwen3.8-27b) via .env
    python evaluate_llm_judge.py --api htw --model <name>  # anderes Uni-Modell
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_results import (  # noqa: E402
    FIELDS, SIM, norm_date, norm_scale, parse_results, parse_groundtruth, prf,
)

LOCAL_BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "qwen2.5vl:3b"      # Standard bei --api local
HTW_JUDGE_MODEL = "qwen3.8-27b"   # Standard bei --api htw

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
    prompt = build_prompt(field, gts, pred)
    last_err = ""
    for attempt in range(retries + 1):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2000,  # Reserve für <think>-Ausgabe von Qwen3
            )
            v, reason = parse_verdict(r.choices[0].message.content)
            if v:
                # Plausibilitätsschutz: kleine Modelle sagen gern zu großzügig "JA"
                if field == "ID" and v != "NEIN" and SIM["ID"](gts[0], pred) < 0.5 \
                        and max(SIM["ID"](g, pred) for g in gts) < 0.5:
                    return "NEIN", f"(Plausibilitätscheck: Zeichenähnlichkeit <0.5) {reason}"
                return v, reason
            last_err = reason
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
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
    n = sum(c.values())
    strict_ok = c["TN"] + c["MATCH_STRICT"]
    lenient_ok = strict_ok + c["MATCH_LLM"]
    soft = lenient_ok + 0.5 * c["PARTIAL"]
    tp = c["MATCH_STRICT"] + c["MATCH_LLM"]
    fp = c["FP"] + c["PARTIAL"] + c["WRONG"]
    fn = c["FN"] + c["PARTIAL"] + c["WRONG"]
    p, r, f1 = prf(tp, fp, fn)
    return {
        "N": n, "strict_acc": strict_ok / n if n else 0, "lenient_acc": lenient_ok / n if n else 0,
        "soft_score": soft / n if n else 0, "precision": p, "recall": r, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": c["TN"],
    }


def main():
    ap = argparse.ArgumentParser(description="Faire Evaluation mit lokalem LLM als Schiedsrichter")
    ap.add_argument("--results", default=None)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--api", choices=["local", "htw"], default="local",
                    help="local = Ollama, htw = Uni-API (HTW_API_KEY/HTW_BASE_URL aus .env)")
    ap.add_argument("--base-url", default=None, help="überschreibt die URL der gewählten API")
    ap.add_argument("--model", default=None, help="Judge-Modell (überschreibt den Standard)")
    ap.add_argument("--judge-fields", default="Title,ID",
                    help="Felder, die das LLM bewerten darf (Standard: Title,ID)")
    ap.add_argument("--out-dir", default="eval_output")
    args = ap.parse_args()

    def ask_path(prompt):
        while True:
            val = input(prompt).strip().strip("\"'")
            if val:
                return val
            print("  Bitte einen Pfad eingeben.")

    results_path = Path((args.results or ask_path("Pfad zur Results-Datei: ")).strip().strip("\"'"))
    gt_path = Path((args.gt or ask_path("Pfad zur Groundtruth-Datei (.xlsx/.csv): ")).strip().strip("\"'"))
    for p in (results_path, gt_path):
        if not p.exists():
            sys.exit(f"Datei nicht gefunden: {p}")

    judge_fields = {x.strip() for x in args.judge_fields.split(",") if x.strip()}
    unknown = judge_fields - set(FIELDS)
    if unknown:
        sys.exit(f"Unbekannte Felder in --judge-fields: {unknown}")

    from openai import OpenAI
    if args.api == "htw":
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parent / ".env")
        except ImportError:
            pass
        api_key, base_url = os.getenv("HTW_API_KEY"), args.base_url or os.getenv("HTW_BASE_URL")
        if not api_key or not base_url:
            sys.exit("HTW_API_KEY / HTW_BASE_URL nicht gefunden (.env neben dem Skript?)")
        args.model = args.model or HTW_JUDGE_MODEL
    else:
        api_key, base_url = "ollama", args.base_url or LOCAL_BASE_URL
        args.model = args.model or MODEL_NAME
    args.base_url = base_url
    client = OpenAI(base_url=base_url, api_key=api_key)

    gt = parse_groundtruth(gt_path)
    preds, stats = parse_results(results_path)

    print(f"\n{len(gt)} Dokumente, Judge-Modell: {args.model} @ {args.base_url}")
    details, n_calls = evaluate(gt, preds, client, args.model, judge_fields)

    per_field = aggregate(details)
    rows, tot = [], {k: 0 for k in next(iter(per_field.values()))}
    metrics = {}
    for f in FIELDS:
        metrics[f] = field_metrics(per_field[f])
        for k in tot:
            tot[k] += per_field[f][k]
    overall = field_metrics(tot)
    macro = {k: sum(metrics[f][k] for f in FIELDS) / len(FIELDS)
             for k in ("strict_acc", "lenient_acc", "soft_score", "precision", "recall", "f1")}

    header = ["Field", "N", "Strict_Acc", "Lenient_Acc", "Soft_Score", "Precision", "Recall", "F1"]

    def row(name, m):
        return [name, m.get("N", ""), f"{m['strict_acc']:.4f}", f"{m['lenient_acc']:.4f}",
                f"{m['soft_score']:.4f}", f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}"]

    rows = [row(f, metrics[f]) for f in FIELDS]
    rows.append(row("OVERALL (micro)", overall))
    rows.append(row("MACRO (Ø Felder)", macro))

    flipped = sum(1 for d in details if d["cell"] in ("MATCH_LLM", "PARTIAL"))
    llm_errors = sum(1 for d in details if "LLM-Fehler" in d["reason"])

    out = []
    out.append("=" * 90)
    out.append(f"FAIRE EVALUATION MIT LLM-JUDGE  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    out.append(f"Results: {results_path} | Groundtruth: {gt_path}")
    out.append(f"Judge: {args.model} | LLM geprüfte Felder: {', '.join(sorted(judge_fields))}")
    out.append("=" * 90)
    out.append(f"Dokumente: {len(gt)} | LLM-Aufrufe: {n_calls} | "
               f"durch Judge aufgewertet (JA/TEILWEISE): {flipped} | LLM-Fehler: {llm_errors}")
    out.append("")
    fmt = "{:<20}{:>5}{:>12}{:>13}{:>12}{:>11}{:>9}{:>8}"
    out.append(fmt.format(*header))
    out.append("-" * 90)
    for r in rows:
        out.append(fmt.format(*[str(x) for x in r]))
    out.append("")
    out.append(f">>> GESAMT-ÜBEREINSTIMMUNG (Soft-Score): {overall['soft_score'] * 100:.1f} %")
    out.append(f">>> Strikt (ohne LLM):                   {overall['strict_acc'] * 100:.1f} %")
    out.append(f">>> Großzügig (JA zählt voll):           {overall['lenient_acc'] * 100:.1f} %")
    out.append("")
    out.append("Strict_Acc: nur strikte Treffer + korrektes 'Not found'. Lenient_Acc: zusätzlich "
               "Judge-'JA'. Soft_Score: JA=1, TEILWEISE=0.5, NEIN=0.")
    out.append("Precision/Recall/F1: TEILWEISE zählt als FP und FN. Begründungen je Fall: "
               "siehe llm_details_*.csv (Spalte 'reason').")
    summary = "\n".join(out)
    print("\n" + summary)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / f"llm_metrics_{stamp}.txt").write_text(summary, encoding="utf-8")
    with open(out_dir / f"llm_metrics_{stamp}.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(header)
        w.writerows(rows)
    with open(out_dir / f"llm_details_{stamp}.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(details[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(details)
    print(f"\nGespeichert in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()