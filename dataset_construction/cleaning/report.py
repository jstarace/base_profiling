"""Render CleaningRecords into a human-readable Markdown report."""


def render_markdown(records):
    """`records` is a list of CleaningRecord dicts (the JSONL shape). Returns Markdown."""
    n = len(records)
    needs = sum(1 for r in records if r.get("panel_decision") == "needs_cleaning")
    all_no_action = sum(1 for r in records if r.get("all_no_action"))
    changed = sum(1 for r in records if r.get("original_text") != r.get("final_text"))
    cuts = reph = veto = dropped = 0
    for r in records:
        for d in r.get("final_decisions", []):
            cuts += d.get("action") == "cut"
            reph += d.get("action") == "rephrase"
            veto += d.get("action") == "veto"
        dropped += len(r.get("flag_mismatches", []))

    out = [
        "# Cleaning report",
        "",
        f"- **Rows:** {n}  ·  needs_cleaning: {needs}  ·  clean: {n - needs}  ·  all_no_action: {all_no_action}",
        f"- **Text changed:** {changed} of {n} rows",
        f"- **Final-judge actions:** {cuts} cut · {reph} rephrase · {veto} veto "
        f"(veto = panel overturned)",
        f"- **Flags dropped (quote didn't match its index):** {dropped}",
        "",
        "---",
    ]

    for i, r in enumerate(records):
        rephrase_by_idx = {rp["sentence_index"]: rp for rp in r.get("rephrases", [])}
        verdict = r.get("panel_decision")
        tag = verdict + (" · all_no_action" if r.get("all_no_action") else "")
        orig, final = r.get("original_text", ""), r.get("final_text", "")
        text_line = (f"{len(orig)} → {len(final)} chars ({len(final) - len(orig):+d})"
                     if orig != final else f"unchanged ({len(orig)} chars)")

        out += ["", f"## Row {i} — `{r.get('id')}`",
                f"**Verdict:** {tag}  ·  source: {r.get('source')}  ·  **text:** {text_line}"]

        decisions = r.get("final_decisions", [])
        if decisions:
            out += ["", "**Final judge decisions:**"]
            for d in decisions:
                idx, action = d["sentence_index"], d["action"]
                if action == "rephrase" and idx in rephrase_by_idx:
                    rp = rephrase_by_idx[idx]
                    out += ["",
                            f"- **[{idx}] REPHRASE**",
                            f"  - from: _{rp['original']}_",
                            f"  - to:   _{rp['rephrased']}_",
                            f"  - why:  {d.get('justification', '')}"]
                else:
                    out += ["",
                            f"- **[{idx}] {action.upper()}** — _{d.get('sentence', '')}_",
                            f"  - why: {d.get('justification', '')}"]
        elif verdict == "needs_cleaning":
            out += ["", "_Flagged overall, but no sentence reached majority — nothing sent to the final judge._"]

        if r.get("flag_mismatches"):
            out += ["", f"**Dropped flags (bad quote):** {len(r['flag_mismatches'])}"]

        out += ["", "---"]

    return "\n".join(out)
