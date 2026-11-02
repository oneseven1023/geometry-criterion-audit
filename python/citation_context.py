# -*- coding: utf-8 -*-
"""citation_context.py -- 提取正文每个引用位置的上下文与对应文献标题，
用于逐条核对引用是否与语境相符（防张冠李戴）。

用法：python citation_context.py
输出：results/citation_context.md
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_doi_audit import parse_refs, clean  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAIN = os.path.join(ROOT, "paper", "main.tex")
REF = os.path.join(ROOT, "paper", "references.tex")
OUT = os.path.join(ROOT, "results", "citation_context.md")


def strip_tex(s):
    """去掉常见 LaTeX 标记，便于阅读。"""
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    s = re.sub(r"\\(?:textbf|emph|textit|mathrm|mathbf)\{([^{}]*)\}", r"\1", s)
    s = s.replace("~", " ")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = re.sub(r"[{}]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def main():
    txt = open(MAIN, encoding="utf-8").read()
    refs = parse_refs(REF)
    bykey = {r["key"]: r for r in refs}
    order = {r["key"]: i for i, r in enumerate(refs, 1)}

    # 记录每个引用位置
    hits = []  # (pos, key, raw_cluster)
    for m in re.finditer(r"\\cite\{([^}]*)\}", txt):
        pos = m.start()
        for k in [x.strip() for x in m.group(1).split(",") if x.strip()]:
            hits.append((pos, k))

    lines = ["# 引用位置与语境对照表", ""]
    lines.append("按文献编号排序，每处引用给出上下文（前后各约 130/110 字）与文献题名，")
    lines.append("用于逐条核对引用与语境是否相符。")
    lines.append("")

    # 按 key 汇总
    from collections import defaultdict
    occ = defaultdict(list)
    for pos, k in hits:
        occ[k].append(pos)
    order_keys = sorted(occ, key=lambda k: order.get(k, 999))

    for k in order_keys:
        r = bykey.get(k, {})
        num = order.get(k, "?")
        lines.append("## [%s] %s" % (num, k))
        lines.append("题名：%s" % clean(r.get("title", "?")))
        lines.append("出处：%s, %s" % (r.get("venue", "?"), r.get("year", "?")))
        lines.append("")
        for pos in sorted(occ[k]):
            pre = strip_tex(txt[max(0, pos - 200):pos])
            post = strip_tex(txt[pos:pos + 200])
            pre = pre[-130:] if len(pre) > 130 else pre
            post = post[:110]
            lines.append("- …%s【cite】%s…" % (pre, post))
        lines.append("")

    open(OUT, "w", encoding="utf-8").write("\n".join(lines))
    print("引用位置数:", len(hits))
    print("涉及文献数:", len(occ))
    print("写出:", OUT)


if __name__ == "__main__":
    main()
