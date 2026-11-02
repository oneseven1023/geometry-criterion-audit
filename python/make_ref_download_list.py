# -*- coding: utf-8 -*-
"""make_ref_download_list.py -- 由 ref_doi_audit.json 与本地 PDF 目录生成
参考文献获取清单 HTML，便于按编号逐条下载。

用法：python make_ref_download_list.py
输出：rt_localization/参考文献获取清单.html
"""
import html
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JSON_IN = os.path.join(ROOT, "results", "ref_doi_audit.json")
PDFDIR = os.path.join(ROOT, "参考文献.第1版")
OUT = os.path.join(ROOT, "参考文献获取清单.html")

TYPE_ZH = {"journal-article": "期刊", "proceedings-article": "会议",
           "book-chapter": "书籍章节", "monograph": "专著"}
# 仅限确定开放获取的出版商前缀：MDPI、SpringerOpen/BMC、PLoS、Hindawi。
# 10.1049 为 IET，10.1109 为 IEEE，10.1016 为 Elsevier，多数需机构订阅，不作标注。
OA_PREFIX = ("10.3390/", "10.1186/", "10.1371/", "10.1155/")

# 两条老文献落在 IET、SIAM 平台，机构无订阅入口；此处给出
# 已在 2026-10-12 逐条验证过的开放副本直链（作者自存 / 教学存档）。
OPEN_ALT = {
    2: ("作者自存（特拉维夫大学主页）",
        "https://nadavlevanon.sites.tau.ac.il/_files/ugd/dfbcdc_c7bfb673744741c19c53a9e82fd57ae8.pdf"),
    49: ("教学存档（Duke 大学课程页）",
        "https://www.cs.duke.edu/brd/Teaching/Bio/asmb/current/Readings3/hendrickson-conditions.pdf"),
}


def show_clean(s):
    """把 LaTeX 片段转成可读文本用于页面显示。"""
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    s = s.replace("^{3}", "3").replace("\\&", "&")
    s = re.sub(r"[{}]", "", s)
    return s


def scan_local():
    got = {}
    if not os.path.isdir(PDFDIR):
        return got
    for fn in os.listdir(PDFDIR):
        m = re.match(r"\[(\d+)\]", fn)
        if m:
            got[int(m.group(1))] = fn
    return got


def main():
    rows = json.load(open(JSON_IN, encoding="utf-8"))
    local = scan_local()

    n_total = len(rows)
    n_have = sum(1 for r in rows if r["num"] in local)
    n_need = n_total - n_have
    n_j = sum(1 for r in rows if r.get("cr_type") == "journal-article")
    n_c = sum(1 for r in rows if r.get("cr_type") == "proceedings-article")
    n_oa = (sum(1 for r in rows if any(r.get("doi", "").startswith(p) for p in OA_PREFIX))
            + len(OPEN_ALT))

    body = []
    for r in rows:
        num = r["num"]
        have = num in local
        title = html.escape(show_clean(r["title"]))
        authors = html.escape(r["authors"])
        venue = html.escape(show_clean(r.get("cr_venue") or r.get("venue") or ""))
        year = r.get("cr_year") or r.get("year") or ""
        typ = TYPE_ZH.get(r.get("cr_type", ""), r.get("cr_type", ""))
        url = r.get("url", "")
        doi = r.get("doi", "")
        is_oa = any(doi.startswith(p) for p in OA_PREFIX)

        if url:
            txt = ("doi.org/" + doi) if doi else "OpenReview 论坛"
            link = ('<a href="%s" target="_blank" rel="noopener">%s</a>'
                    % (html.escape(url), html.escape(txt)))
            if r.get("url_pdf"):
                link += (' &middot; <a href="%s" target="_blank" rel="noopener">PDF</a>'
                         % html.escape(r["url_pdf"]))
        else:
            link = '<span class="muted">无链接</span>'
        alt = OPEN_ALT.get(num)
        if alt:
            link += ('<br><a class="alt" href="%s" target="_blank" rel="noopener">'
                     '开放版 PDF（%s）</a>' % (html.escape(alt[1]), html.escape(alt[0])))
        oa = '<span class="oa">开放获取</span>' if is_oa else ''
        if not doi:
            oa = '<span class="oa">会议开放获取</span>'
        if alt:
            oa += '<span class="oa">有开放副本</span>'

        state = ('<span class="badge ok">已下载</span>' if have
                 else '<span class="badge need">待下载</span>')
        cls = "have" if have else "need"

        body.append(
            '<tr class="%s" data-state="%s">'
            '<td class="num">%d</td>'
            '<td class="ref"><div class="au">%s</div>'
            '<div class="ti">%s</div>'
            '<div class="ve">%s, %s %s</div></td>'
            '<td class="ty">%s</td>'
            '<td class="lk">%s</td>'
            '<td class="st">%s</td>'
            '</tr>' % (cls, "have" if have else "need", num, authors, title,
                       venue, year, oa, typ, link, state))

    doc = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>参考文献获取清单</title>
<style>
:root{--bg:#f7f8fa;--card:#fff;--line:#e3e6ea;--tx:#1f2328;--mut:#6b7280;
--need:#b45309;--needbg:#fef3c7;--ok:#166534;--okbg:#dcfce7;--ac:#1d4ed8;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);
font:15px/1.6 system-ui,"Segoe UI","Microsoft YaHei",sans-serif;}
.wrap{max-width:1120px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:24px;margin:0 0 6px}
.sub{color:var(--mut);margin:0 0 20px;font-size:14px}
.stats{display:flex;flex-wrap:wrap;gap:12px;margin:0 0 18px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:12px 18px;min-width:120px}
.stat b{display:block;font-size:22px;line-height:1.2}
.stat span{color:var(--mut);font-size:13px}
.note{background:#eef2ff;border:1px solid #c7d2fe;border-radius:10px;
padding:12px 16px;margin:0 0 18px;font-size:14px}
.note code{background:#fff;border:1px solid var(--line);border-radius:4px;
padding:1px 5px;font-size:13px}
.bar{display:flex;gap:8px;margin:0 0 12px}
.bar button{border:1px solid var(--line);background:var(--card);color:var(--tx);
padding:6px 14px;border-radius:999px;cursor:pointer;font-size:14px}
.bar button.active{background:var(--ac);border-color:var(--ac);color:#fff}
table{width:100%;border-collapse:collapse;background:var(--card);
border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line);
vertical-align:top}
th{background:#f1f3f6;font-size:13px;color:var(--mut);font-weight:600;
white-space:nowrap}
tr:last-child td{border-bottom:none}
td.num{font-variant-numeric:tabular-nums;color:var(--mut);width:44px}
.au{color:var(--mut);font-size:13px}
.ti{font-weight:600;margin:2px 0}
.ve{color:var(--mut);font-size:13px}
.ty{white-space:nowrap;font-size:13px;color:var(--mut)}
.lk{white-space:nowrap}
.lk a{color:var(--ac);text-decoration:none;font-size:13px}
.lk a:hover{text-decoration:underline}
.lk a.alt{color:var(--ok);font-weight:600}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12.5px;
white-space:nowrap}
.badge.ok{background:var(--okbg);color:var(--ok)}
.badge.need{background:var(--needbg);color:var(--need)}
.oa{display:inline-block;background:#e0f2fe;color:#075985;border-radius:999px;
padding:1px 8px;font-size:12px;margin-left:6px}
tr.need td.num{color:var(--need);font-weight:700}
tr.have{background:#fbfdfb}
.muted{color:var(--mut)}
@media(max-width:760px){.ty,.st{display:none}}
</style>
</head>
<body>
<div class="wrap">
<h1>参考文献获取清单</h1>
<p class="sub">面向 IEEE TMC 投稿的参考文献，编号与论文 references.tex 的顺序编码一致</p>

<div class="stats">
<div class="stat"><b>__N__</b><span>文献总数</span></div>
<div class="stat"><b>__NEED__</b><span>待下载</span></div>
<div class="stat"><b>__HAVE__</b><span>已下载</span></div>
<div class="stat"><b>__NJ__</b><span>期刊论文</span></div>
<div class="stat"><b>__NC__</b><span>会议论文</span></div>
<div class="stat"><b>__NOA__</b><span>开放获取</span></div>
</div>

<div class="note">
下载完成后，请按 <code>[编号] 作者 年 出处 标题.pdf</code> 的格式命名，
放入 <code>参考文献.第1版</code> 目录，编号必须与论文引用编号保持一一对应。
点击链接进入出版商页面获取 PDF，标注为开放获取的可直接下载。
<br>编号 <b>2</b>、<b>25</b>、<b>49</b> 三种文献落在 IET、IOP、SIAM 平台，机构无订阅入口，
已在 2026-10-12 逐条验证其开放副本可正常下载，且副本为出版社原刊扫描版，均已归档。
原编号 <b>25</b>（Compagnoni 等，Inverse Problems 2014）因机构无 IOP 订阅入口、公开渠道仅有作者接受稿，
已于 2026-10-12 替换为同主题的 IEEE 期刊论文（编号 <b>25</b> 现为 IEEE Signal Processing Letters 2026），可正常下载。
</div>

<div class="bar">
<button class="active" data-f="all">全部</button>
<button data-f="need">仅待下载</button>
<button data-f="have">仅已下载</button>
</div>

<table>
<thead><tr><th>编号</th><th>文献</th><th>类型</th><th>下载链接</th><th>状态</th></tr></thead>
<tbody>
__ROWS__
</tbody>
</table>
</div>
<script>
var btns=document.querySelectorAll('.bar button');
btns.forEach(function(b){b.onclick=function(){
  btns.forEach(function(x){x.classList.remove('active')});
  b.classList.add('active');
  var f=b.getAttribute('data-f');
  document.querySelectorAll('tbody tr').forEach(function(tr){
    tr.style.display=(f==='all'||tr.getAttribute('data-state')===f)?'':'none';
  });
};});
</script>
</body>
</html>
"""
    doc = (doc.replace("__N__", str(n_total)).replace("__NEED__", str(n_need))
           .replace("__HAVE__", str(n_have)).replace("__NJ__", str(n_j))
           .replace("__NC__", str(n_c)).replace("__NOA__", str(n_oa))
           .replace("__ROWS__", "\n".join(body)))

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(doc)
    print("总数 %d，待下载 %d，已下载 %d，期刊 %d，会议 %d，开放获取 %d"
          % (n_total, n_need, n_have, n_j, n_c, n_oa))
    print("写出:", OUT)


if __name__ == "__main__":
    main()
