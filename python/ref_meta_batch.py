# -*- coding: utf-8 -*-
"""ref_meta_batch.py -- 用 Crossref 批量核实新增文献的元数据（期刊名、卷、期、页、年、作者）

只查询与打印，不写任何文件。DOI 优先，无 DOI 者用 bibliographic 检索取前 2 条。
用法：python ref_meta_batch.py
"""
import json
import urllib.parse
import urllib.request

BASE = "https://api.crossref.org/works"
SEL = "DOI,title,container-title,volume,issue,page,published,published-print,author,type"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ref-meta-batch/1.0 (mailto:test@example.com)"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read().decode("utf-8"))


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


def by_doi(doi):
    print("== KEY", doi[0], "| DOI", doi[1])
    try:
        show(get(BASE + "/" + doi[1])["message"])
    except Exception as e:  # noqa
        print("  FAILED:", e)


def by_title(key, title, extra="", rows=2):
    print("== KEY", key, "| QUERY:", title[:70])
    q = urllib.parse.urlencode({"query.bibliographic": title + " " + extra, "rows": rows, "select": SEL})
    try:
        for m in get(BASE + "?" + q)["message"]["items"]:
            show(m)
            print("  ---")
    except Exception as e:  # noqa
        print("  FAILED:", e)


CASES = [
    ("li2014", "Li Krolik multipath exploitation passive radar IEEE Journal of Selected Topics in Signal Processing 2014"),
    ("setlur2013", "Setlur Devroye Multipath exploited Bayesian and Cramer-Rao bounds for single-sensor target localization EURASIP Journal on Advances in Signal Processing 2013"),
    ("rendas1991", "Rendas Moura Cramer-Rao bound for location systems in multipath environments IEEE Transactions on Signal Processing 1991"),
    ("liu2012", "Liu Jiang Yang PARALIND-based identifiability results for parameter estimation via uniform linear array EURASIP Journal on Advances in Signal Processing 2012"),
    ("geng2022", "Geng Shrestha Yajnanarayana Dahlman Behravan Joint scatterer localization and material identification using radio access technology EURASIP Journal on Wireless Communications and Networking 2022"),
    ("compagnoni2014", "Compagnoni Notari Antonacci Sarti A comprehensive analysis of the geometry of TDOA maps in localization problems Inverse Problems 2014"),
    ("denkovski2017", "Denkovski Geometric interpretation of theoretical bounds for RSS-based source localization with uncertain anchor positions Digital Signal Processing 2017"),
    ("duggal2025", "Duggal Buehrer Dhillon Reed Diffraction-aided wireless positioning IEEE Transactions on Wireless Communications 2025"),
    ("wen2021", "Wen Kulmer Witrisal Wymeersch 5G positioning and mapping with diffuse multipath IEEE Transactions on Wireless Communications 2021"),
    ("guo2024", "Guo Lv Zeng Angle-based positioning estimation leveraging diffuse scattering paths in millimeter-wave MIMO systems IEEE Sensors Journal 2024"),
    ("boman2001", "Boman Stoica Low angle estimation models methods and bounds Digital Signal Processing 2001"),
    ("wang2016", "Wang Cao Su Wang Target and reflecting surface height joint estimation in low-angle radar IET Radar Sonar and Navigation 2016"),
    ("ribot2016", "Ribot Botteron Farine Derivation of the Cramer-Rao bound in the GNSS-reflectometry context for static ground-based receivers in scenarios with coherent reflection Sensors 2016"),
    ("proszynski2015", "Proszynski Revisiting Baarda's concept of minimal detectable bias with regard to outlier identifiability Journal of Geodesy 2015"),
    ("wang2019", "Wang Joint spatiotemporal multipath mitigation in large-scale array localization IEEE Transactions on Signal Processing 2019"),
    ("gutenkunst2007", "Gutenkunst Waterfall Casey Brown Myers Sethna Universally sloppy parameter sensitivities in systems biology models PLoS Computational Biology 2007"),
    ("bishop2010", "Bishop Fidan Anderson Dogancay Pathirana Optimality analysis of sensor-target localization geometries Automatica 2010"),
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
