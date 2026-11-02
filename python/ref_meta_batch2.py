# -*- coding: utf-8 -*-
"""ref_meta_batch2.py -- 补查 ref_meta_batch.py 中未返回或被限流的文献元数据。只查询不写文件。"""
import json
import time
import urllib.parse
import urllib.request

BASE = "https://api.crossref.org/works"
SEL = "DOI,title,container-title,volume,issue,page,published,published-print,author,type"


def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ref-meta-batch/1.0 (mailto:test@example.com)"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa
            if i == tries - 1:
                raise
            time.sleep(3.0)


def show(m, indent="  "):
    au = m.get("author", []) or []
    names = "; ".join((a.get("family", "") + " " + a.get("given", "")).strip() for a in au)
    ct = m.get("container-title") or []
    pub = m.get("published") or m.get("published-print") or {}
    print(indent + "DOI      :", m.get("DOI"))
    print(indent + "type     :", m.get("type"))
    print(indent + "title    :", (m.get("title") or [""])[0])
    print(indent + "container:", ct[0] if ct else "")
    print(indent + "vol/iss/pg:", m.get("volume"), "/", m.get("issue"), "/", m.get("page"))
    print(indent + "published:", pub.get("date-parts"))
    print(indent + "authors  :", names[:240])


def by_title(key, title, extra="", rows=2):
    print("== KEY", key, "| QUERY:", title[:70])
    q = urllib.parse.urlencode({"query.bibliographic": title + " " + extra, "rows": rows, "select": SEL})
    try:
        for m in get(BASE + "?" + q)["message"]["items"]:
            show(m)
            print("  ---")
    except Exception as e:  # noqa
        print("  FAILED:", e)
    time.sleep(1.2)


CASES = [
    ("duggal2025", "Duggal Buehrer Dhillon Reed Diffraction-aided wireless positioning IEEE Transactions on Wireless Communications 2025"),
    ("wen2021", "Wen Kulmer Witrisal Wymeersch 5G positioning and mapping with diffuse multipath IEEE Transactions on Wireless Communications 2021"),
    ("han2016", "Han Shen Zhang Meng Win Performance limits and geometric properties of array localization IEEE Transactions on Information Theory 2016"),
    ("marengo2012", "Marengo Cramer-Rao bound study of multiple scattering effects in target localization International Journal of Antennas and Propagation 2012"),
    ("li2023", "Li Liao Li Valaee Automatic indoor radio map construction and localization via multipath fingerprint extrapolation IEEE Transactions on Wireless Communications 2023"),
    ("li2022", "Li Li Ma Zhang Array orientation adjustments subject to optimal direct position determination performance IEEE Signal Processing Letters 2022"),
    ("dogandzic2001", "Dogandzic Nehorai Cramer-Rao bounds for estimating range velocity and direction with an active array IEEE Transactions on Signal Processing 2001"),
    ("wielandt2017", "Wielandt De Strycker Indoor multipath assisted angle of arrival localization Sensors 2017"),
    ("naik2013", "Naik Bapat Access point height based location accuracy characterization in LOS and OLOS scenarios Wireless Personal Communications 2013"),
    ("ibrahim2024", "Ibrahim Chen Ye Ghazalian Kim Nilsson Wymeersch van de Beek Inferring direction and orientation from polarized signals feasibility and bounds IEEE Open Journal of the Communications Society 2024"),
    ("horodynski2021", "Horodynski Bouchet Rotter Invariance property of the Fisher information in scattering media Physical Review Letters 2021"),
]

if __name__ == "__main__":
    for key, q in CASES:
        by_title(key, q)
