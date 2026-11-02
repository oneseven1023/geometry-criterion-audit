# -*- coding: utf-8 -*-
"""archive_refs.py -- 把 下载文献/ 中的 PDF 按论文引用编号归档进 参考文献.第1版/

命名沿用现有风格：[编号] 作者姓 年 出处简称 标题.pdf
先用首页文本与题名做词袋校验，再复制。默认干跑，加 --apply 才真正复制。

用法：
  python archive_refs.py           干跑，只打印匹配与校验
  python archive_refs.py --apply   执行复制
"""
import difflib
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_doi_audit import parse_refs, clean  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROJ = os.path.dirname(ROOT)
SRC = os.path.join(PROJ, "下载文献")
DST = os.path.join(ROOT, "参考文献.第1版")
REF = os.path.join(ROOT, "paper", "references.tex")

APPLY = "--apply" in sys.argv

# 源文件名片段到编号的对应，精确名优先，其次子串包含。
SRC_KEYS = [
    ("PhysRevD.77.042001", 1),
    ("sensors-25-05115", 3),
    ("s00190-015-0828-y", 5),
    ("A_Survey_on_the_Impact_of_Multipath", 7),
    ("High-Accuracy_Localization_for_Assisted_Living", 8),
    ("sensors-18-04249", 9),
    ("Evaluation_of_Position-Related_Information", 10),
    ("Asymptotic_Limits_of_3D_Localization", 11),
    ("AOA_Sensor_Placement", 12),
    ("Robust_A-Optimal_Sensor_Geometry", 13),
    ("electronics-15-02504", 14),
    ("Simultaneous_Target_and_Multipath_Positioning", 15),
    ("M3M3_Multipath_Assisted", 16),
    ("Automatic_Indoor_Radio_Map_Construction", 17),
    ("Performance_Limits_and_Geometric_Properties", 18),
    ("Cramer-Rao_bounds_for_estimating_range", 19),
    ("1687-6180-2013-53", 20),
    ("file.pdf", 21),
    ("Cramer-rao_bounds_for_power_delay_profile", 22),
    ("1687-6180-2012-154", 23),
    ("main.pdf", 24),
    ("5G_Positioning_and_Mapping", 26),
    ("Angle-Based_Positioning_Estimation", 27),
    ("Non-Line-of-Sight_Localization", 28),
    ("s11277-012-0934-6", 29),
    ("Joint_Spatiotemporal_Multipath_Mitigation", 30),
    ("Maximum_Likelihood_Joint_Angle_and_Delay", 31),
    ("Effect_of_Path-Overlap", 32),
    ("Single-Anchor_Ultra-Wideband", 33),
    ("Cramer-Rao_bound_for_location_systems", 34),
    ("Leveraging_RIS-Enabled", 35),
    ("Low Angle Estimation", 36),
    ("IET Radar Sonar", 37),
    ("sensors-17-02522", 38),
    ("Diffraction-Aided_Wireless_Positioning", 39),
    ("s13638-022-02167-7", 40),
    ("International Journal of Antennas and Propagation", 41),
    ("sensors-26-00597", 42),
    ("main-222.pdf", 43),
    ("Development_of_an_Open-Source_Tool", 44),
    ("main-3333.pdf", 45),
    ("Array_Orientation_Adjustments", 46),
    ("Inferring_Direction_and_Orientation", 47),
    ("Invariance Property of the Fisher Information", 48),
    ("sensors-16-02063", 55),
]

VENUE_ABBR = {
    "vallisneri2008": "Phys Rev D", "levanon2000": "IEE Proc RSN",
    "kramaric2025": "Sensors", "villaverde2019": "Complexity",
    "proszynski2015": "Journal of Geodesy", "hehn2025": "ICLR",
    "aditya2018": "Proc IEEE", "witrisal2016": "IEEE SPM",
    "wilding2018": "Sensors", "leitinger2015": "IEEE JSAC",
    "li2026": "ISAES", "xu2025": "IEEE TMC", "fang2026": "IEEE Sensors J",
    "dai2026": "Electronics", "li2014": "IEEE JSTSP", "chen2021": "IEEE TMC",
    "li2023": "IEEE TWC", "han2016": "IEEE TIT", "dogandzic2001": "IEEE TSP",
    "setlur2013": "EURASIP JASP", "gutenkunst2007": "PLoS Comput Biol",
    "oktem2011": "ICASSP", "liu2012": "EURASIP JASP",
    "denkovski2017": "Digital Signal Processing", "ospel2026": "IEEE SPL",
    "wen2021": "IEEE TWC", "guo2024": "IEEE Sensors J", "seow2008": "IEEE TMC",
    "naik2013": "Wireless Pers Commun", "wang2019": "IEEE TSP",
    "bellili2019": "IEEE TMC", "shen2008": "IEEE ICC", "ge2022": "IEEE TMC",
    "rendas1991": "IEEE TSP", "keykhosravi2023": "IEEE VTM",
    "boman2001": "Digital Signal Processing", "wang2016": "IET RSN",
    "wielandt2017": "Sensors", "duggal2025": "IEEE TWC",
    "geng2022": "EURASIP JWCN", "marengo2012": "IJAP", "zhang2026": "Sensors",
    "aksu2011": "Computer Networks", "odonoughue2023": "IEEE RADAR",
    "bishop2010": "Automatica", "li2022": "IEEE SPL", "ibrahim2024": "IEEE OJ-COMS",
    "horodynski2021": "Phys Rev Lett", "hendrickson1992": "SIAM J Comput",
    "kannan2007": "IDC", "kannan2009": "WCNC", "kannan2010": "IEEE TVT",
    "kannan2011": "Wireless Networks", "liu2016": "JPDC", "ribot2016": "Sensors",
}


def capname(w):
    """每个空格、连字符、撇号后的字母大写，其余小写。"""
    out, nxt = [], True
    for ch in w.lower():
        if ch in " -'\u2019":
            out.append(ch)
            nxt = True
        elif nxt:
            out.append(ch.upper())
            nxt = False
        else:
            out.append(ch)
    return "".join(out)


def author_surnames(a):
    """取作者姓。连续的多字母词都计入姓，碰到长度 1 的名缩写即停，
    以正确处理 Ben Amor、De Strycker 这类复合姓。"""
    parts = [p.strip() for p in re.split(r",", a) if p.strip()]
    out = []
    for p in parts:
        if p.lower().startswith("et al"):
            out.append("et al")
            continue
        ws = p.split()
        sur = []
        for w in ws:
            if len(w) == 1:
                break
            sur.append(w)
        if not sur:
            sur = [ws[0]]
        out.append(capname(" ".join(sur)))
    if len(out) > 4:
        out = out[:4]
        if out[-1] != "et al":
            out.append("et al")
    return " ".join(out)


def build_name(num, r):
    return "[%d] %s %s %s %s.pdf" % (
        num, author_surnames(r["authors"]), r.get("year") or "",
        VENUE_ABBR.get(r["key"], ""), sanitize(r["title"]))


def sanitize(s):
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    s = s.replace("^{3}", "3").replace("\\&", "&")
    s = re.sub(r"[{}\\]", "", s)
    s = re.sub(r'[<>:"/\\|?*\u2010\u2011]', " ", s)
    return re.sub(r"\s+", " ", s).strip()


def first_page(path):
    try:
        r = subprocess.run(["pdftotext", "-f", "1", "-l", "1", "-enc", "UTF-8",
                            path, "-"], capture_output=True, timeout=90)
        return r.stdout.decode("utf-8", "ignore")
    except Exception:
        return ""


def bag(text, title):
    tw = [w for w in clean(title).lower().split() if len(w) > 3]
    if not tw:
        return 0.0
    tl = text.lower()
    return sum(1 for w in tw if w in tl) / len(tw)


def main():
    refs = {r["key"]: r for r in parse_refs(REF)}
    bynum = {}
    for i, r in enumerate(parse_refs(REF), 1):
        bynum[i] = r
    files = [f for f in os.listdir(SRC) if f.lower().endswith(".pdf")]

    plan = []
    unmatched = []
    for fn in files:
        num = None
        for frag, n in SRC_KEYS:
            if fn == frag:
                num = n
                break
        if num is None:
            for frag, n in SRC_KEYS:
                if frag in fn:
                    num = n
                    break
        if num is None:
            unmatched.append(fn)
            continue
        plan.append((num, fn))

    # 重复编号检查
    from collections import Counter
    dup = [n for n, c in Counter(n for n, _ in plan).items() if c > 1]

    print("=== 归档计划（源文件 -> 编号 校验）===")
    ok_cnt = 0
    for num, fn in sorted(plan):
        r = bynum[num]
        txt = first_page(os.path.join(SRC, fn))
        sc = bag(txt, r["title"])
        flag = "OK " if sc >= 0.7 else "?? "
        if sc >= 0.7:
            ok_cnt += 1
        print("[%2d] %s %-52s bag=%.2f  <- %s" % (
            num, flag, r["title"][:52], sc, fn[:48]))

    print("\n匹配文件数 = %d，校验通过 = %d，未匹配 = %d，编号重复 = %s"
          % (len(plan), ok_cnt, len(unmatched), dup or "无"))
    if unmatched:
        print("未匹配文件：")
        for fn in unmatched:
            print("   ", fn)

    dstdir = scan_dst()
    print("\n目标目录已有编号：", sorted(dstdir.keys()))
    missing = [n for n in sorted(bynum) if n not in [x[0] for x in plan] and n not in dstdir]
    print("归档后仍缺编号：", missing)

    if not APPLY:
        print("\n[干跑] 未执行复制。确认无误后加 --apply 执行。")
        return

    print("\n=== 执行复制 ===")
    os.makedirs(DST, exist_ok=True)
    made = 0
    for num, fn in sorted(plan):
        r = bynum[num]
        name = build_name(num, r)
        src = os.path.join(SRC, fn)
        dst = os.path.join(DST, name)
        if os.path.exists(dst):
            continue
        shutil.copy2(src, dst)
        made += 1
    print("复制完成，共 %d 篇。" % made)

    print("\n=== 对齐文件名 ===")
    renamed = 0
    for num, fn in sorted(plan):
        want = build_name(num, bynum[num])
        for f in os.listdir(DST):
            if not re.match(r"\[%d\]\s" % num, f) or f == want:
                continue
            if not os.path.exists(os.path.join(DST, want)):
                os.rename(os.path.join(DST, f), os.path.join(DST, want))
                renamed += 1
                print("  改名:\n    %s\n    -> %s" % (f[:72], want[:72]))
    print("对齐完成，改名 %d 个。目标目录现有 %d 篇。"
          % (renamed, len(os.listdir(DST))))


def scan_dst():
    got = {}
    if os.path.isdir(DST):
        for fn in os.listdir(DST):
            m = re.match(r"\[(\d+)\]", fn)
            if m:
                got[int(m.group(1))] = fn
    return got


if __name__ == "__main__":
    main()
