# -*- coding: utf-8 -*-
"""35 Nature Communications 投稿格式合规核验（Stage 5 finalize 新增核验面）。

核验基准：Nature Communications Manuscript Checklist（官方清单，逐条落实）+
           nature.com/ncomms/submit/how-to-submit 的格式要求。

与 99c/30/34 的分工：三者核验**科学内容与交付物一致性**（数字对源、图面↔图注↔数据、
依赖时间序）；本脚本只核验**期刊投稿格式**，不重复内容核验。

硬性条款（任一 FAIL 必须修完并重跑 99c/30/34/99d——改动可能引入新的失配）：
  F1  标题 ≤ 15 词且不含标点
  F2  摘要 ≤ 150 词、无引文、含 "Here we"、结果用现在时
  F3  正文（Introduction + Results + Discussion）≤ 5000 词【作者裁决：v3.1 暂缓，见 ALLOW_OVERLONG】
  F4  Introduction < 1000 词、无子标题、不引用图表
  F5  Results 子标题 < 60 字符且无标点
  F6  Methods 子标题 < 60 字符且无标点
  F7  Discussion 无子标题
  F8  图注各 ≤ 350 词（图注终点与 30/34 同一正则：下一个图注 / '---' / 文末）
  F9  显示项 ≤ 10；参考文献 ≤ 70
  F10 参考文献编号 1..N 连续、按首现顺序（由 99c [17] 复核，本脚本只查条数）
  F11 章节顺序：Methods → Data/Code availability → References → Acknowledgements →
      Author contributions → Competing interests → Additional information → Figure legends
  F12 正文 SI 引用格式统一为 (Supplementary Fig. N) / (Supplementary Table N)
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

MS = os.path.join(C.PIPE, "06_论文稿", "manuscript_v4.md")
SI = os.path.join(C.PIPE, "06_论文稿", "Supplementary_Information_v4.md")

# 作者裁决（Stage 5）：正文超长暂缓压缩，按现稿提交；改为 WARN 而非 FAIL。
ALLOW_OVERLONG = True

fails, warns, report = [], [], []


def say(x):
    print(x, flush=True)
    report.append(str(x))


def ok(label, cond, detail="", warn_only=False):
    tag = "OK" if cond else ("WW" if warn_only else "XX")
    say(f"  [{tag}] {label}{(' — ' + detail) if detail else ''}")
    if not cond:
        (warns if warn_only else fails).append(label)


def wc(s):
    """Nature 式词数：含字母数字的空白分隔 token 数。"""
    return len([x for x in re.split(r"\s+", s.strip()) if re.search(r"[A-Za-z0-9]", x)])


t = open(MS, encoding="utf-8").read().replace("\r\n", "\n")
si = open(SI, encoding="utf-8").read().replace("\r\n", "\n")

say("# Nature Communications 投稿格式合规核验（35_nc_format_check）\n")

# ---------------------------------------------------------------- F1 标题
say("## F1 标题")
title = t.split("\n")[0].lstrip("# ").strip()
tw = wc(title)
ok(f"标题 ≤ 15 词（实际 {tw} 词）", tw <= 15, title)
ok("标题不含标点", not re.search(r"[^\w\s\-]", title))

# ---------------------------------------------------------------- F2 摘要
say("")
say("## F2 摘要")
ab = re.search(r"## Abstract\n\n(.*?)\n\n---", t, re.S).group(1).strip()
aw = wc(ab)
ok(f"摘要 ≤ 150 词（实际 {aw} 词）", aw <= 150)
ok("摘要不含参考文献引文", not re.search(r"\[\d", ab))
ok("摘要含 'Here we' 引导句", "Here we" in ab)
ok("结果现在时（inflates / exceed）",
   "inflates" in ab and re.search(r"\b(exceed|raise|raises|capture)\b", ab) is not None)

# ---------------------------------------------------------------- F3 正文词数
say("")
say("## F3 正文词数（Introduction + Results + Discussion）")


def block(pat):
    m = re.search(pat + r"\n+([\s\S]*?)\n+---\n", t, re.M)
    return m.group(1) if m else ""


intro, res, disc = block(r"^## Introduction$"), block(r"^## Results$"), block(r"^## Discussion$")
meth = block(r"^## Methods$")
main = wc(intro) + wc(res) + wc(disc)
ok(f"正文 ≤ 5000 词（Introduction {wc(intro)} + Results {wc(res)} + "
   f"Discussion {wc(disc)} = {main}）",
   main <= 5000, f"超出 {main - 5000} 词，作者裁决暂缓压缩（ALLOW_OVERLONG={ALLOW_OVERLONG}）",
   warn_only=ALLOW_OVERLONG)

# ---------------------------------------------------------------- F4 Introduction
say("")
say("## F4 Introduction")
ok(f"Introduction < 1000 词（实际 {wc(intro)}）", wc(intro) < 1000)
ok("Introduction 无子标题", not re.search(r"^### ", intro, re.M))
ok("Introduction 不引用显示项（Fig./Table）",
   not re.search(r"\bFig\.\s*\d|\bFigure\s*\d|\bTable\s*\d", intro))

# ---------------------------------------------------------------- F5/F6 子标题
say("")
say("## F5 Results 子标题（< 60 字符、无标点）")
for h in re.findall(r"^### (.+)$", res, re.M):
    ok(f"{len(h):3d} 字符 | {h}", len(h) < 60 and not re.search(r"[^\w\s\-]", h))

say("")
say("## F6 Methods 子标题（< 60 字符、无标点）")
for h in re.findall(r"^### (.+)$", meth, re.M):
    ok(f"{len(h):3d} 字符 | {h}", len(h) < 60 and not re.search(r"[^\w\s\-]", h))

say("")
say("## F7 Discussion 无子标题")
ok("Discussion 无子标题", not re.search(r"^### ", disc, re.M))

# ---------------------------------------------------------------- F8 图注词数
say("")
say("## F8 图注词数（各 ≤ 350）")
legends = t.split("## Figure legends")[1]
nf = 0
for m in re.finditer(r"\*\*Figure (\d) \|(.+?)(?=\n\*\*Figure|\n---|\Z)", legends, re.S):
    nf += 1
    ok(f"Figure {m.group(1)}: {wc('Figure ' + m.group(1) + ' |' + m.group(2))} 词",
       wc("Figure " + m.group(1) + " |" + m.group(2)) <= 350)
for m in re.finditer(r"\*\*Supplementary Figure (\d) \|(.+?)(?=\n\*\*|\n---|\Z)", si, re.S):
    ok(f"Supplementary Figure {m.group(1)}: "
       f"{wc('Supplementary Figure ' + m.group(1) + ' |' + m.group(2))} 词",
       wc("Supplementary Figure " + m.group(1) + " |" + m.group(2)) <= 350)

# ---------------------------------------------------------------- F9 显示项与文献数
say("")
say("## F9 显示项与参考文献数量")
ok(f"主图数 {nf} ≤ 10", nf <= 10)
refs = re.findall(r"^(\d+)\. ", t.split("## References")[1], re.M)
ok(f"参考文献 {len(refs)} 条 ≤ 70", len(refs) <= 70)
ok("参考文献编号连续 1..N", sorted(int(x) for x in refs) == list(range(1, len(refs) + 1)))

# ---------------------------------------------------------------- F11 章节顺序
say("")
say("## F11 章节顺序（NC：… Methods → Data/Code availability → References → End notes → Figure legends）")
ORDER = ["Abstract", "Introduction", "Results", "Discussion", "Methods", "Data availability",
         "Code availability", "References", "Acknowledgements", "Author contributions",
         "Competing interests", "Additional information", "Figure legends"]
secs = re.findall(r"^## (.+)$", t, re.M)
ok(f"实际章节序列 == NC 要求序列", secs == ORDER, f"实际 {secs}")
ok("图注位于全部 end notes 之后",
   secs.index("Figure legends") > max(secs.index(s) for s in
                                      ["References", "Acknowledgements", "Author contributions",
                                       "Competing interests", "Additional information"]
                                      if s in secs))
ok("参考文献位于致谢之前", secs.index("References") < secs.index("Acknowledgements"))

# ---------------------------------------------------------------- F12 SI 引用格式
say("")
say("## F12 正文 SI 引用格式（NC：'Figure' 缩写为 'Fig.'，'Table' 不缩写）")
bad = re.findall(r"Supplementary Figures? \d", t)
ok("无 'Supplementary Figure(s) N' 全称写法（应为 Supplementary Fig. N）",
   not bad, f"违规 {sorted(set(bad))}")
ok("无 'Supplementary Figs/Tables' 复数写法",
   not re.search(r"Supplementary (?:Figs|Tables)\.", t))

# ---------------------------------------------------------------- 结论
say("")
say("## 结论")
if fails:
    say(f"**FAIL** — {len(fails)} 项不合规：")
    for f in fails:
        say(f"- {f}")
else:
    say(f"**PASS** — 全部硬性格式条款通过（{len(warns)} 项作者裁决豁免）")
    for w in warns:
        say(f"- 豁免: {w}")

with open(os.path.join(C.OUT, "35_nc_format_check_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
say("")
say("DONE 35 NC format check")
if fails:
    raise SystemExit(1)