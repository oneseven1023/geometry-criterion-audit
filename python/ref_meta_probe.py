# -*- coding: utf-8 -*-
"""ref_meta_probe.py -- 用 Crossref 核实新增五篇文献的元数据（期刊名、卷、页、年、作者）

只查询与打印，不写任何文件。
"""
import json
import urllib.parse
import urllib.request

BASE = "https://api.crossref.org/works"
SEL = "DOI,title,container-title,volume,issue,page,published,published-print,author,type,short-container-title"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ref-meta-probe/1.0 (mailto:test@example.com)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def show(msg):
    au = msg.get("author", []) or []
    names = "; ".join((a.get("family", "") + " " + a.get("given", "")).strip() for a in au)
    ct = msg.get("container-title") or msg.get("short-container-title") or []
    print("  DOI      :", msg.get("DOI"))
    print("  type     :", msg.get("type"))
    print("  title    :", (msg.get("title") or [""])[0])
    print("  container:", ct[0] if ct else "")
    print("  volume   :", msg.get("volume"), " issue:", msg.get("issue"), " page:", msg.get("page"))
    print("  published:", msg.get("published", {}).get("date-parts"))
    print("  authors  :", names[:200])


def by_doi(doi):
    print("== DOI", doi)
    try:
        show(get(BASE + "/" + doi)["message"])
    except Exception as e:  # noqa
        print("  FAILED:", e)


def by_title(title, extra=""):
    print("== TITLE", title, extra)
    q = urllib.parse.urlencode({"query.bibliographic": title + " " + extra, "rows": 3, "select": SEL})
    try:
        for m in get(BASE + "?" + q)["message"]["items"]:
            show(m)
            print("  ---")
    except Exception as e:  # noqa
        print("  FAILED:", e)


if __name__ == "__main__":
    by_doi("10.1109/ISAES70982.2026.11669665")
    by_title("Robust A-Optimal Sensor Geometry for Target Localization")
    by_title("Reconstructing Spatial Localization Error Maps via Physics-Informed Tensor Completion")
    by_title("A study of localization metrics: Evaluation of position errors in wireless sensor networks", "Computer Networks")
