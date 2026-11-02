# -*- coding: utf-8 -*-
"""ref_doi_audit.py -- 对 references.tex 全部条目用 Crossref 检索 DOI，
核实发表状态（期刊/会议）并生成可下载链接清单。

用法：python ref_doi_audit.py
输出：results/ref_doi_audit.csv 与 results/ref_doi_audit.json（供生成 HTML 使用）
只查询与写这两份结果文件，不改动任何论文源文件。
"""
import csv
import difflib
import json
import os
import re
import time
import urllib.parse
import urllib.request

BASE = "https://api.crossref.org/works"
SEL = ("DOI,title,container-title,volume,issue,page,published,published-print,"
       "issued,author,type,publisher,event")
UA = {"User-Agent": "ref-doi-audit/1.0 (mailto:research@local.invalid)"}

# 人工指定的条目：Crossref 未收录或检索到错配者。
# hehn2025 为 ICLR 2025 正式会议论文，由 OpenReview 提供开放获取全文，无 DOI。
MANUAL = {
    "hehn2025": {
        "doi": "",
        "cr_type": "proceedings-article",
        "cr_title": "Differentiable and Learnable Wireless Simulation with Geometric Transformers",
        "cr_venue": ("The Thirteenth International Conference on Learning "
                     "Representations (ICLR)"),
        "cr_year": "2025",
        "score": 1.0,
        "status": "OK_NO_DOI",
        "url": "https://openreview.net/forum?id=9TClCDZXeh",
        "url_pdf": "https://openreview.net/pdf?id=9TClCDZXeh",
    },
}

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "paper", "references.tex")
OUT_CSV = os.path.join(HERE, "..", "results", "ref_doi_audit.csv")
OUT_JSON = os.path.join(HERE, "..", "results", "ref_doi_audit.json")


def get(url, tries=3):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa
            last = e
            if i < tries - 1:
                time.sleep(2.5)
    raise last


def clean(s):
    """去掉 LaTeX 与标点，得到可比对的纯文本。"""
    s = re.sub(r"\$[^$]*\$", " ", s)
    s = s.replace("\\&", "&")
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = re.sub(r"[{}\\]", "", s)
    s = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", s)
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_refs(path):
    """逐行解析 \\bibitem，返回条目字典列表。"""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = re.match(r"\\bibitem\{([^}]+)\}\s*(.+)", line)
            if not m:
                continue
            key, body = m.group(1).strip(), m.group(2).strip()
            mm = re.match(r"^(.+?)\.\s+(.+?)\[([A-Z])\]", body)
            if mm:
                authors, title, mark = mm.group(1), mm.group(2), mm.group(3)
            else:
                authors, title, mark = "", body, "?"
            # 年份取「逗号后紧跟的四位年」，避免把页码（如 042001、2027-2032、
            # 文章号 2063）误当年份；无此形式时退回最后一个不超过当前年的候选。
            ym = re.findall(r",\s*((?:19|20)\d{2})\b", body)
            if ym:
                year = ym[-1]
            else:
                cand = [y for y in re.findall(r"(?:19|20)\d{2}", body) if int(y) <= 2026]
                year = cand[-1] if cand else ""
            # venue：标记之后到年份之前
            tail = body[body.find("[" + mark + "]"):] if mark != "?" else body
            venue = ""
            if mark == "J":
                vm = re.search(r"\[J\]\.\s*(.+?),\s*(?:19|20)\d{2}", tail)
                venue = vm.group(1).strip() if vm else ""
            elif mark == "C":
                vm = re.search(r"\[C\]//\s*(.+?)(?:,\s*)?(?:19|20)\d{2}", tail)
                venue = vm.group(1).strip() if vm else ""
            out.append({"key": key, "authors": authors, "title": title,
                        "mark": mark, "venue": venue, "year": year, "raw": body})
    return out


def cr_best(title, venue, year):
    """用题名加出处做 bibliographic 检索，返回最佳匹配原始记录。"""
    q = clean(title) + " " + clean(venue) + " " + str(year)
    url = BASE + "?" + urllib.parse.urlencode(
        {"query.bibliographic": q, "rows": 3, "select": SEL})
    items = get(url)["message"]["items"]
    if not items:
        return None, 0.0
    lt = clean(title).lower()
    best, bestsc = None, -1.0
    for m in items:
        ct = (m.get("title") or [""])[0]
        sc = difflib.SequenceMatcher(None, lt, clean(ct).lower()).ratio()
        cy = ""
        p = m.get("published") or m.get("published-print") or m.get("issued") or {}
        dp = p.get("date-parts") or [[None]]
        if dp and dp[0] and dp[0][0]:
            cy = str(dp[0][0])
        if cy and str(year) and cy == str(year):
            sc += 0.15
        if sc > bestsc:
            best, bestsc = m, sc
    return best, bestsc


def main():
    refs = parse_refs(REF)
    print("解析到条目数:", len(refs))
    rows = []
    for i, r in enumerate(refs, 1):
        rec = {"num": i, **r}
        if r["key"] in MANUAL:
            rec.update(MANUAL[r["key"]])
            print("  [%02d] %-16s 人工指定 %-18s %s" % (
                i, r["key"], rec["status"], rec["cr_title"][:60]))
            rows.append(rec)
            continue
        try:
            m, sc = cr_best(r["title"], r["venue"], r["year"])
        except Exception as e:  # noqa
            print("  [%02d] %-16s 检索失败: %s" % (i, r["key"], e))
            rec.update({"doi": "", "cr_type": "", "cr_title": "", "cr_venue": "",
                        "cr_year": "", "score": 0.0, "status": "QUERY_FAILED"})
            rows.append(rec)
            continue
        if not m:
            print("  [%02d] %-16s 未检到" % (i, r["key"]))
            rec.update({"doi": "", "cr_type": "", "cr_title": "", "cr_venue": "",
                        "cr_year": "", "score": 0.0, "status": "NOT_FOUND"})
            rows.append(rec)
            time.sleep(0.4)
            continue
        p = m.get("published") or m.get("published-print") or m.get("issued") or {}
        dp = p.get("date-parts") or [[None]]
        cr_year = str(dp[0][0]) if dp and dp[0] and dp[0][0] else ""
        ct = (m.get("title") or [""])[0]
        cv = (m.get("container-title") or [""])
        cv = cv[0] if cv else ""
        ratio = difflib.SequenceMatcher(None, clean(r["title"]).lower(), clean(ct).lower()).ratio()
        rec.update({"doi": m.get("DOI", ""), "cr_type": m.get("type", ""),
                    "cr_title": ct, "cr_venue": cv, "cr_year": cr_year,
                    "score": round(ratio, 3)})
        ok = ratio >= 0.90
        t = m.get("type", "")
        formal = t in ("journal-article", "proceedings-article", "book-chapter", "monograph")
        if not formal:
            rec["status"] = "TYPE_" + t.upper().replace("-", "_")
        elif not ok:
            rec["status"] = "LOW_MATCH"
        else:
            rec["status"] = "OK"
        print("  [%02d] %-16s sc=%.2f %-18s %s | %s" % (
            i, r["key"], sc, rec["status"], m.get("DOI", ""), ct[:60]))
        rows.append(rec)
        time.sleep(0.4)

    for r in rows:
        if not r.get("url"):
            r["url"] = ("https://doi.org/" + r["doi"]) if r.get("doi") else ""

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["num", "key", "mark", "authors", "title",
                                          "venue", "year", "doi", "cr_type",
                                          "cr_title", "cr_venue", "cr_year",
                                          "score", "status", "url"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)

    from collections import Counter
    c = Counter(r["status"] for r in rows)
    print("\n=== 状态统计 ===")
    for k, v in c.most_common():
        print(" %-18s %d" % (k, v))
    nofind = [r for r in rows if r["status"] in ("NOT_FOUND", "QUERY_FAILED", "LOW_MATCH")
              or r["status"].startswith("TYPE_")]
    print("\n=== 需人工处置 ===")
    for r in nofind:
        print(" [%02d] %-16s %s | %s" % (r["num"], r["key"], r["status"], r["title"][:70]))
    print("\n写出:", OUT_CSV)
    print("写出:", OUT_JSON)


if __name__ == "__main__":
    main()
