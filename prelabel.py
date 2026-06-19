"""
Pre-label + review toolkit for Milestone 3.
 
Two modes:
 
  1) PRE-LABEL (default)
     Reads the unlabeled comments CSV (from collect_reddit.py), asks an LLM to
     suggest ONE label per comment using your planning.md definitions, and
     writes a review CSV. The model's guess goes in `pre_label` (never edited);
     `label` is seeded with the same value for you to CONFIRM OR CORRECT by hand.
 
         python prelabel.py askphilosophy_comments.csv
 
  2) FINALIZE (--finalize)
     After you've reviewed every row's `label` in a spreadsheet, this validates
     the labels, prints the per-label distribution, enforces the checkpoint
     rules (>=200 rows, no label >70%), and writes a clean submission CSV with
     just text,label,notes.
 
         python prelabel.py reviewed.csv --finalize
 
Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY=...
 
The script is RESUMABLE: rerun it and it skips rows already pre-labeled, so you
can stop with Ctrl-C and continue later without paying for the same rows twice.
"""
 
import argparse
import csv
import os
import sys
import time
 
# Keep these in sync with planning.md. This block is sent to the model verbatim.
LABELS = ["argument", "assertion", "exegesis"]
 
DEFINITIONS = """\
You are labeling individual comments from r/askphilosophy by the FORM of the
discourse, NOT by whether the philosophy is correct. Assign exactly one label.
 
- argument: The comment advances a claim and supports it with explicit reasoning
  (premises, distinctions, conditionals, worked-through implications) that visibly
  connect to the conclusion. Test: strip the rhetoric and actual supporting
  reasons remain. The argument need not be correct, only present and visible.
 
- assertion: The comment states a position, judgment, or opinion WITHOUT supplying
  supporting reasons. May be confident or even correct. Name-dropping a thinker or
  using jargon does NOT make it an argument if no reasons are actually given.
 
- exegesis: The comment primarily explains or interprets what a specific text,
  thinker, or position SAYS, rather than arguing for a view of the commenter's own.
  Center of gravity is reporting/clarifying someone else's view.
 
EDGE-CASE RULES:
- Jargon-dressed assertion: if removing technical vocabulary and proper names
  leaves supporting reasons -> argument; if only the bare claim remains -> assertion.
- Exegesis vs argument: judge by center of gravity. Mostly reporting a thinker's
  view -> exegesis; clearly the commenter's own view with text as setup -> argument.
 
Respond with ONLY one word: argument, assertion, or exegesis. No punctuation, no
explanation."""
 
 
def detect_text_column(fieldnames):
    for cand in ("text", "comment_body", "body"):
        if cand in fieldnames:
            return cand
    raise SystemExit(f"No text column found. Looked for text/comment_body/body in {fieldnames}")
 
 
def classify(client, model, text):
    """Return one label string, defaulting to 'assertion' on an unparseable reply."""
    msg = client.messages.create(
        model=model,
        max_tokens=5,
        system=DEFINITIONS,
        messages=[{"role": "user", "content": f"Comment:\n\"\"\"\n{text}\n\"\"\""}],
    )
    reply = msg.content[0].text.strip().lower()
    for lab in LABELS:
        if lab in reply:
            return lab
    return "assertion"  # safe default; you'll review it anyway
 
 
def prelabel(infile, outfile, model):
    with open(infile, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("Input CSV is empty.")
    text_col = detect_text_column(rows[0].keys())
 
    # Resume: load any already-done pre_labels from an existing outfile.
    done = {}
    if os.path.exists(outfile):
        with open(outfile, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("pre_label"):
                    done[r["text"]] = r
 
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
 
    out_fields = ["text", "label", "pre_label", "pre_labeled",
                  "score", "permalink", "notes"]
    out_rows = []
    for i, r in enumerate(rows, 1):
        text = (r.get(text_col) or "").strip()
        if not text:
            continue
        if text in done:                       # already pre-labeled in a prior run
            out_rows.append(done[text])
            continue
        try:
            pred = classify(client, model, text)
        except Exception as e:
            print(f"  row {i}: API error ({e}); leaving blank to retry later",
                  file=sys.stderr)
            pred = ""
        out_rows.append({
            "text": text,
            "label": pred,            # seeded with the guess; YOU confirm/correct
            "pre_label": pred,        # immutable record of the model's suggestion
            "pre_labeled": "True",
            "score": r.get("score", ""),
            "permalink": r.get("permalink", ""),
            "notes": "",
        })
        if i % 10 == 0:
            print(f"  pre-labeled {i}/{len(rows)}", file=sys.stderr)
        time.sleep(0.3)               # gentle pacing
 
        # Write incrementally so a crash never loses progress.
        with open(outfile, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=out_fields)
            w.writeheader()
            w.writerows(out_rows)
 
    print(f"\nWrote {len(out_rows)} rows to {outfile}")
    print("NEXT: open it in a spreadsheet and review the `label` column for every "
          "row. Correct any the model got wrong. Then run with --finalize.")
 
 
def finalize(infile, outfile):
    with open(infile, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
 
    # Validate labels.
    bad = [i + 2 for i, r in enumerate(rows)            # +2 = header + 1-indexed
           if (r.get("label") or "").strip() not in LABELS]
    if bad:
        raise SystemExit(f"Rows with missing/invalid labels (must be one of "
                         f"{LABELS}): lines {bad[:20]}{' ...' if len(bad) > 20 else ''}")
 
    # Distribution + checkpoint rules.
    counts = {lab: 0 for lab in LABELS}
    for r in rows:
        counts[r["label"].strip()] += 1
    total = len(rows)
 
    print(f"\nTotal labeled examples: {total}")
    print("Distribution:")
    ok = True
    for lab in LABELS:
        pct = 100 * counts[lab] / total if total else 0
        flag = "  <-- OVER 70%!" if pct > 70 else ""
        print(f"  {lab:20s} {counts[lab]:4d}  ({pct:5.1f}%){flag}")
        if pct > 70:
            ok = False
    if total < 200:
        print(f"\n  WARNING: {total} < 200 required. Collect more.")
        ok = False
    n_overrides = sum(1 for r in rows
                      if r.get("pre_label") and r["pre_label"] != r["label"])
    if any(r.get("pre_label") for r in rows):
        print(f"\nYou overrode the model on {n_overrides}/{total} rows "
              f"({100*n_overrides/total:.1f}%). Disclose this in your AI usage section.")
 
    # Clean submission file: text, label, notes only.
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "notes"])
        w.writeheader()
        for r in rows:
            w.writerow({"text": r["text"], "label": r["label"].strip(),
                        "notes": r.get("notes", "")})
 
    print(f"\n{'PASS' if ok else 'NOT READY'}: wrote submission CSV to {outfile}")
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--finalize", action="store_true",
                    help="validate reviewed labels and emit submission CSV")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="claude-haiku-4-5-20251001",
                    help="cheap+fast is fine for pre-labeling")
    args = ap.parse_args()
 
    if args.finalize:
        finalize(args.infile, args.out or "dataset.csv")
    else:
        prelabel(args.infile, args.out or "reviewed.csv", args.model)
 
 
if __name__ == "__main__":
    main()
 