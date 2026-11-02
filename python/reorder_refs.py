# -*- coding: utf-8 -*-
"""按正文首现顺序重排 paper/references.tex（GB/T 7714 顺序编码制）。

用途：向正文插入新的 \\cite{key} 之后，把 references.tex 的 bibitem 重排为
与正文首现顺序一致，并输出旧编号到新编号的映射（用于同步重命名 PDF 目录）。

输入：paper/main.tex（正文引用顺序）、paper/references.tex（现有条目）
输出：paper/references.new.tex、控制台打印映射表与校验结果
"""
import re
import sys

MAIN = "paper/main.tex"
REFS = "paper/references.tex"
OUT = "paper/references.new.tex"

# 本次新增的 5 条 TMC 文献正文（GB/T 7714 顺序编码制，sentence case）
NEW_ENTRIES = {
    "xu2025": "XU S, WU L, LI X, et al. AOA sensor placement for anchor-assisted target "
              "localization in GNSS-denied environment: formulation, bounds and "
              "optimization[J]. IEEE Transactions on Mobile Computing, 2025, 24(10): "
              "10792-10806.",
    "chen2021": "CHEN Z, ZHU G, WANG S, et al. $M^{3}$: multipath assisted Wi-Fi "
                "localization with a single access point[J]. IEEE Transactions on Mobile "
                "Computing, 2021, 20(2): 588-602.",
    "seow2008": "SEOW C K, TAN S Y. Non-line-of-sight localization in multipath "
                "environments[J]. IEEE Transactions on Mobile Computing, 2008, 7(5): "
                "647-660.",
    "bellili2019": "BELLILI F, BEN AMOR S, AFFES S, et al. Maximum likelihood joint angle "
                   "and delay estimation from multipath and multicarrier transmissions "
                   "with application to indoor localization over IEEE 802.11ac radio[J]. "
                   "IEEE Transactions on Mobile Computing, 2019, 18(5): 1116-1132.",
    "ge2022": "GE F, SHEN Y. Single-anchor ultra-wideband localization system using "
              "wrapped PDoA[J]. IEEE Transactions on Mobile Computing, 2022, 21(12): "
              "4609-4623.",
}


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def cite_order(main_text):
    """按出现顺序返回正文引用的 key 列表（去重，保留首现次序）。"""
    order = []
    seen = set()
    for m in re.finditer(r"\\cite\{([^}]*)\}", main_text):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k not in seen:
                seen.add(k)
                order.append(k)
    return order


def parse_refs(refs_text):
    """返回 (prefix, entries, suffix)。entries 为 [(key, body)]，body 不含 \\bibitem 前缀。"""
    marker = r"\begin{thebibliography}{99}"
    i = refs_text.index(marker) + len(marker)
    prefix = refs_text[:i]
    suffix = "\\end{thebibliography}"
    body = refs_text[i:refs_text.index(suffix)]
    entries = []
    for m in re.finditer(r"\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem\{|$)", body, re.S):
        entries.append((m.group(1).strip(), m.group(2).strip()))
    return prefix, entries, suffix


def main():
    main_text = read(MAIN)
    refs_text = read(REFS)
    order = cite_order(main_text)
    prefix, entries, suffix = parse_refs(refs_text)
    old = dict(entries)
    old_order = [k for k, _ in entries]

    # 校验：正文引用集合与参考文献集合的一致性
    missing = [k for k in order if k not in old and k not in NEW_ENTRIES]
    unused = [k for k in old_order if k not in order]
    if missing:
        print("!! 正文引用但既不在表中也未新增：", missing)
    if unused:
        print("!! 表中存在但正文未引用：", unused)

    # 合并新条目
    merged = dict(old)
    added = []
    for k, v in NEW_ENTRIES.items():
        if k in merged:
            print("!! 新增 key 与现有重复，跳过：", k)
            continue
        merged[k] = v
        added.append(k)

    # 新顺序：正文首现顺序；正文未引用的旧条目追加到末尾，避免丢失
    new_order = [k for k in order if k in merged]
    tail = [k for k in old_order if k not in new_order]
    new_order += tail

    # 写文件
    parts = [prefix, "\n\n"]
    for k in new_order:
        parts.append("\\bibitem{%s} %s\n\n" % (k, merged[k]))
    parts.append(suffix + "\n")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("".join(parts))

    # 映射表
    print("=== 旧编号 -> 新编号 ===")
    new_index = {k: i + 1 for i, k in enumerate(new_order)}
    for k in old_order:
        if old.get(k) != merged.get(k):
            pass
        print("  %-16s %3d -> %3d" % (k, old_order.index(k) + 1, new_index[k]))
    print("=== 新增条目 ===")
    for k in added:
        print("  %-16s 新编号 %3d" % (k, new_index[k]))
    print("总计：旧 %d 条，新 %d 条" % (len(old_order), len(new_order)))


if __name__ == "__main__":
    main()
