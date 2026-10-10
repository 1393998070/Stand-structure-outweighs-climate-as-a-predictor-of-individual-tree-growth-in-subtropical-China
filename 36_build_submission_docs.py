# -*- coding: utf-8 -*-
"""36 投稿件构建：Markdown -> Word (.docx) + PDF（正文与补充材料）。

设计要点
--------
* **不改源稿**：以 manuscript_v4.md / Supplementary_Information_v4.md 为只读输入，
  在 08_定稿/_build/ 下生成中间 Markdown（插入图件）后再转换。
* **Word**：pandoc md->docx；随后用 python-docx 落实 NC 的排版要求——
  A4、单栏、正文双倍行距、不两端对齐、页脚阿拉伯页码。
* **PDF**：pandoc md->HTML5 片段（自带 CSS），再经 xhtml2pdf(reportlab) 渲染。
  字体用 matplotlib 自带的 DejaVu（Serif/Sans/Mono）注册，覆盖正文中的
  Unicode 上标（⁻¹ ⁻³）、希腊字母（ρ χ² Δ）、数学符号（× ≈ ≥ ≤ ∝）与破折号。
* **图件**：图注在前、图件紧随其后（图注充当图下题注），宽度按 A4 版心
  自动缩放（宽 ≤ 16 cm、高 ≤ 19 cm），保持长宽比。

用法: python 36_build_submission_docs.py
产出: 08_定稿/manuscript_v4.0.{docx,pdf}
      08_定稿/Supplementary_Information_v4.0.{docx,pdf}
      08_定稿/manuscript_v4.0_中文版.{docx,pdf}          （源稿 manuscript_v4_中文版.md）
      08_定稿/Supplementary_Information_v4.0_中文版.{docx,pdf}
      08_定稿/00_构建核验报告.md
"""
import os
import re
import shutil
import subprocess
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

PIPE = C.PIPE
SRC_MS = os.path.join(PIPE, "06_论文稿", "manuscript_v4.md")
SRC_SI = os.path.join(PIPE, "06_论文稿", "Supplementary_Information_v4.md")
FIGDIR = os.path.join(PIPE, "05_图表")
OUT = os.path.join(PIPE, "08_定稿")
BUILD = os.path.join(OUT, "_build")

MS_OUT = os.path.join(OUT, "manuscript_v4.0")
SI_OUT = os.path.join(OUT, "Supplementary_Information_v4.0")

MAIN_FIGS = {1: "Fig1_overview", 2: "Fig2_patterns", 3: "Fig3_cv",
             4: "Fig4_shap", 5: "Fig5_dependence", 6: "Fig6_map"}
SI_FIGS = {1: "SF1_age_decay", 2: "SF2_density", 3: "SF3_shape_stability", 4: "SF4_deltad"}

MAX_W_CM, MAX_H_CM = 16.0, 19.0

report = []


def say(x):
    print(x, flush=True)
    report.append(str(x))


# --------------------------------------------------------------------------
# 1. 图件尺寸 -> 插入宽度
# --------------------------------------------------------------------------
try:
    from PIL import Image
except ImportError:                                  # pragma: no cover
    Image = None


def fig_width_cm(stem):
    """按 A4 版心等比缩放，返回宽度（cm）。"""
    if Image is None:
        return MAX_W_CM
    p = os.path.join(FIGDIR, stem + ".png")
    w_px, h_px = Image.open(p).size
    ar = h_px / w_px
    w = min(MAX_W_CM, MAX_H_CM / ar)
    return round(w, 2)


# --------------------------------------------------------------------------
# 2. 生成插入图件的中间 Markdown
# --------------------------------------------------------------------------
def inject_figures(md_text, mapping, label_re, rel_prefix):
    """在每条图注前插入对应图件（图注成为图下题注）。"""
    def repl(m):
        n = int(m.group(1))
        stem = mapping.get(n)
        if stem is None:
            return m.group(0)
        rel = rel_prefix + stem + ".png"
        return f"![]({rel}){{width={fig_width_cm(stem)}cm}}\n\n" + m.group(0)
    return re.sub(label_re, repl, md_text)


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


os.makedirs(BUILD, exist_ok=True)
ms_md = open(SRC_MS, encoding="utf-8").read().replace("\r\n", "\n")
si_md = open(SRC_SI, encoding="utf-8").read().replace("\r\n", "\n")

# 图件相对 08_定稿/_build/ 的路径
REL_FIG = os.path.relpath(FIGDIR, BUILD).replace("\\", "/") + "/"

ms_md_fig = inject_figures(ms_md, MAIN_FIGS, r"\*\*Figure (\d) \|", REL_FIG)
si_md_fig = inject_figures(si_md, SI_FIGS, r"\*\*Supplementary Figure (\d) \|", REL_FIG)

MS_MD = os.path.join(BUILD, "manuscript_build.md")
SI_MD = os.path.join(BUILD, "si_build.md")
write(MS_MD, ms_md_fig)
write(SI_MD, si_md_fig)

say("# 投稿件构建报告（36_build_submission_docs）\n")
say(f"- 源稿: `06_论文稿/manuscript_v4.md`、`Supplementary_Information_v4.md`（只读）")
say(f"- 输出: `08_定稿/`")
say("")
say("## 1. 图件插入")
for n, stem in MAIN_FIGS.items():
    say(f"  - Figure {n} -> {stem}.png，插入宽度 {fig_width_cm(stem)} cm")
for n, stem in SI_FIGS.items():
    say(f"  - Supplementary Figure {n} -> {stem}.png，插入宽度 {fig_width_cm(stem)} cm")


# --------------------------------------------------------------------------
# 3. Word：pandoc -> docx，再后处理排版
# --------------------------------------------------------------------------
PANDOC = shutil.which("pandoc") or r"C:\Users\AA\AppData\Local\Pandoc\pandoc.exe"
# 读取器选择：gfm 不支持 link_attributes（图片宽度属性会漏成正文文字）。
# 改用 pandoc markdown 并按需关掉三个会改动正文的扩展：
#   -smart            保持直引号，使输出与源稿逐字一致
#   -raw_tex          防止 `\|`（SI 表格里转义竖线）被当成 TeX 命令而静默丢弃
#   -tex_math_dollars 正文无数学环境，避免误判
MD_READER = "markdown+pipe_tables+link_attributes-smart-raw_tex-tex_math_dollars"


def to_docx(md_path, out_stem):
    out = out_stem + ".docx"
    # 不加 --toc：期刊稿件正文惯例不带目录，且 pandoc 的 TOC 域在 Word 中打开前为空白
    cmd = [PANDOC, md_path, "-f", MD_READER, "-o", out,
           "--resource-path", os.path.dirname(md_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"pandoc docx 失败:\n{r.stderr}")
    return out


def polish_docx(path, double_spaced=True):
    """NC 排版要求：A4、单栏、正文双倍行距、不两端对齐、页脚阿拉伯页码。"""
    import docx
    from docx.shared import Cm, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    d = docx.Document(path)

    for s in d.sections:
        s.page_width, s.page_height = Cm(21.0), Cm(29.7)      # A4
        s.left_margin = s.right_margin = Cm(2.5)
        s.top_margin = s.bottom_margin = Cm(2.5)

    skip = ("Heading", "Title", "Subtitle", "Caption", "TOC", "Image")
    for p in d.paragraphs:
        st = (p.style.name or "")
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT               # 不两端对齐
        if any(st.startswith(k) for k in skip):
            continue
        if double_spaced:
            p.paragraph_format.line_spacing = 2.0
        p.paragraph_format.space_after = Pt(0)

    # 页脚页码（PAGE 域；Word 打开时自动计算）
    footer = d.sections[0].footer
    fp = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    fp.text = ""
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run()
    b = OxmlElement("w:fldChar"); b.set(qn("w:fldCharType"), "begin")
    i = OxmlElement("w:instrText"); i.set(qn("xml:space"), "preserve"); i.text = "PAGE"
    e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), "end")
    for el in (b, i, e):
        run._r.append(el)

    d.save(path)


# --------------------------------------------------------------------------
# 4. PDF：pandoc -> HTML5 片段 + CSS -> xhtml2pdf
# --------------------------------------------------------------------------
def prepare_fonts():
    """把 DejaVu（matplotlib 自带）与 CJK 兜底字体复制到 _build/fonts/，
    供 @font-face 以纯 ASCII 相对路径引用。

    CJK 兜底不可省略：文末版本注含中文路径「04_分析结果」，DejaVu 无汉字字形，
    缺兜底会在 PDF 中渲染成空白/方框（构建日志会报 "No font ... can draw '结'"）。
    """
    import matplotlib
    dst = os.path.join(BUILD, "fonts")
    os.makedirs(dst, exist_ok=True)
    FONT_FILES = {
        "DJVSerif": (os.path.join(matplotlib.get_data_path(), "fonts", "ttf"), (
            "DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf",
            "DejaVuSerif-Italic.ttf", "DejaVuSerif-BoldItalic.ttf")),
        "DJVSans": (os.path.join(matplotlib.get_data_path(), "fonts", "ttf"), (
            "DejaVuSans.ttf", "DejaVuSans-Bold.ttf",
            "DejaVuSans-Oblique.ttf", "DejaVuSans-BoldOblique.ttf")),
        "DJVMono": (os.path.join(matplotlib.get_data_path(), "fonts", "ttf"), (
            "DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf",
            "DejaVuSansMono-Oblique.ttf", "DejaVuSansMono-BoldOblique.ttf")),
        # 等线 DengXian：单面 TTF，含完整汉字覆盖（msyh/simsun 为 .ttc，reportlab 不便直接加载）
        "DJVCJK": (r"C:\Windows\Fonts", (
            "Deng.ttf", "Dengb.ttf", "Deng.ttf", "Dengb.ttf")),
    }
    for src_dir, files in FONT_FILES.values():
        for f in files:
            shutil.copyfile(os.path.join(src_dir, f), os.path.join(dst, f))
    return {k: v[1] for k, v in FONT_FILES.items()}


FONTS = prepare_fonts()


def font_face_css():
    """@font-face：xhtml2pdf 对自定义字体只认这一条注册途径（reportlab 直接注册不被其字体表识别）。"""
    out = []
    for fam, (reg, bold, ital, bi) in FONTS.items():
        out.append(f"@font-face {{ font-family: {fam}; font-weight: normal;"
                   f" font-style: normal; src: url('fonts/{reg}'); }}")
        out.append(f"@font-face {{ font-family: {fam}; font-weight: bold;"
                   f" font-style: normal; src: url('fonts/{bold}'); }}")
        out.append(f"@font-face {{ font-family: {fam}; font-weight: normal;"
                   f" font-style: italic; src: url('fonts/{ital}'); }}")
        out.append(f"@font-face {{ font-family: {fam}; font-weight: bold;"
                   f" font-style: italic; src: url('fonts/{bi}'); }}")
    return "\n".join(out)


CSS = font_face_css() + """
@page {
  size: A4;
  margin: 2.2cm 2.2cm 2.2cm 2.2cm;
}
body { font-family: DJVSerif, DJVCJK; font-size: 10pt; line-height: 1.45;
       text-align: left; color: #000000; }
/* 中文版：<title> 若不加字体规则会用默认 Helvetica，xhtml2pdf 会对每个汉字
   报 "No font ... can draw"，故显式指定 CJK 字体族 */
html { font-family: DJVSerif, DJVCJK; }
title { font-family: DJVSans, DJVCJK; }
h1 { font-family: DJVSans, DJVCJK; font-size: 15pt; line-height: 1.3; margin: 0 0 10pt 0; }
h2 { font-family: DJVSans, DJVCJK; font-size: 12.5pt; margin: 16pt 0 6pt 0; }
h3 { font-family: DJVSans, DJVCJK; font-size: 11pt; margin: 12pt 0 4pt 0; }
p { margin: 0 0 6pt 0; }
img { max-width: 100%; }
em { font-style: italic; }
strong { font-weight: bold; }
code { font-family: DJVMono, DJVCJK; font-size: 8.5pt; }
table { border-collapse: collapse; width: 100%; font-size: 8pt; margin: 6pt 0; }
th, td { border: 0.5pt solid #888888; padding: 2pt 3pt; text-align: left;
         vertical-align: top; }
th { font-family: DJVSans, DJVCJK; background-color: #eeeeee; }
hr { border: none; border-top: 0.5pt solid #bbbbbb; margin: 10pt 0; }
"""


def to_html(md_path, html_path, title):
    r = subprocess.run([PANDOC, md_path, "-f", MD_READER, "-t", "html5",
                        "--resource-path", os.path.dirname(md_path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"pandoc html 失败:\n{r.stderr}")
    html = ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{title}</title><style>{CSS}</style></head><body>"
            f"{r.stdout}</body></html>")
    write(html_path, html)
    return html_path


def to_pdf(html_path, out_stem):
    """渲染为 PDF。注意：xhtml2pdf 新版把本地读取限制在 base_dir 内，
    故显式给出 base_dir=PIPE，使 05_图表/ 与 _build/fonts/ 都落在允许范围内。"""
    from pathlib import Path
    from xhtml2pdf import pisa
    from xhtml2pdf.config.resources import ResourceAccessPolicy, use_policy

    base = os.path.dirname(html_path)

    def link_callback(uri, rel):
        from urllib.parse import unquote
        if uri.startswith("http://") or uri.startswith("https://"):
            return uri
        p = os.path.normpath(os.path.join(base, unquote(uri)))
        return p if os.path.exists(p) else uri

    out = out_stem + ".pdf"
    policy = ResourceAccessPolicy(base_dir=Path(PIPE))
    with use_policy(policy):
        with open(out, "wb") as f:
            st = pisa.CreatePDF(open(html_path, encoding="utf-8").read(),
                                dest=f, link_callback=link_callback,
                                encoding="utf-8", path=base)
    if st.err:
        raise RuntimeError(f"xhtml2pdf 报错 {st.err} 处")
    stamp_page_numbers(out)
    return out


def stamp_page_numbers(pdf_path):
    """xhtml2pdf 不支持 @page 页边框，故渲染后逐页叠加居中页码。"""
    import io
    from reportlab.pdfgen import canvas
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(pdf_path)
    n = len(reader.pages)
    writer = PdfWriter()
    for idx, page in enumerate(reader.pages, 1):
        w = float(page.mediabox.width)
        h = float(page.mediabox.height)
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=(w, h))
        c.setFont("Helvetica", 9)
        c.drawCentredString(w / 2.0, 30, f"{idx} / {n}")
        c.save()
        buf.seek(0)
        page.merge_page(PdfReader(buf).pages[0])
        writer.add_page(page)
    with open(pdf_path, "wb") as f:
        writer.write(f)


# --------------------------------------------------------------------------
# 5. 执行
# --------------------------------------------------------------------------
say("")
say("## 2. Word 生成")
ms_docx = to_docx(MS_MD, MS_OUT); polish_docx(ms_docx, double_spaced=True)
si_docx = to_docx(SI_MD, SI_OUT); polish_docx(si_docx, double_spaced=False)
for p in (ms_docx, si_docx):
    say(f"  - {os.path.relpath(p, PIPE)}  ({os.path.getsize(p) / 1024:.0f} KB)")

say("")
say("## 3. PDF 生成")
ms_html = to_html(MS_MD, os.path.join(BUILD, "manuscript_build.html"), "manuscript_v4.0")
si_html = to_html(SI_MD, os.path.join(BUILD, "si_build.html"), "Supplementary_Information_v4.0")
ms_pdf = to_pdf(ms_html, MS_OUT)
si_pdf = to_pdf(si_html, SI_OUT)
for p in (ms_pdf, si_pdf):
    say(f"  - {os.path.relpath(p, PIPE)}  ({os.path.getsize(p) / 1024:.0f} KB)")

# --------------------------------------------------------------------------
# 6. 构建核验
# --------------------------------------------------------------------------
from pypdf import PdfReader

say("")
say("## 4. 构建核验（PDF 文本与图件抽取）")

CHECKS = {
    "正文": (ms_pdf, ["Abstract", "Introduction", "Results", "Discussion", "Methods",
                     "Data availability", "Code availability", "References",
                     "Acknowledgements", "Author contributions", "Competing interests",
                     "Additional information", "Figure legends",
                     "Stand structure outweighs climate as a predictor",
                     "honest tree-level predictive skill",
                     "10⁻³", "R²", "×", "≈", "ρ", "Δ", "—",
                     "Supplementary Table 7"], 6),
    "补充材料": (si_pdf, ["Supplementary Information", "Supplementary Methods",
                       "Supplementary Tables", "Supplementary Figure Legends",
                       "Supplementary Table 1", "Supplementary Table 7",
                       "10⁻³", "ρ", "Manuscript v4"], 4),
}
fails = []
for tag, (pdf, tokens, n_expect_img) in CHECKS.items():
    rd = PdfReader(pdf)
    txt = "\n".join((pg.extract_text() or "") for pg in rd.pages)
    say(f"  - {tag}: {len(rd.pages)} 页，抽取文本 {len(txt)} 字符")
    for tk in tokens:
        ok = tk in txt
        if not ok:
            fails.append(f"{tag} 缺文本 '{tk}'")
        say(f"    [{'OK' if ok else 'XX'}] 含 '{tk}'")
    n_img = 0
    for pg in rd.pages:
        res = pg.get("/Resources") or {}
        xo = res.get("/XObject")
        if xo is None:
            continue
        xo = xo.get_object()
        for k in xo:
            if xo[k].get_object().get("/Subtype") == "/Image":
                n_img += 1
    ok_img = n_img >= n_expect_img
    if not ok_img:
        fails.append(f"{tag} 图件数 {n_img} < 期望 {n_expect_img}")
    say(f"    [{'OK' if ok_img else 'XX'}] PDF 内嵌位图 {n_img} 个（期望 ≥ {n_expect_img}）")
    # 反向断言：pandoc 未消费的图片属性会以文字形式出现在版面上（gfm 缺 link_attributes 的典型症状）
    leak = "{width=" in txt
    if leak:
        fails.append(f"{tag} 版面上残留未解析的图片属性 '{{width='")
    say(f"    [{'OK' if not leak else 'XX'}] 无未解析的图片属性残留（'{'{'}width='）")

say("")
say("## 4b. 反向断言：Word 正文不得残留未解析属性")
for f in ("manuscript_v4.0.docx", "Supplementary_Information_v4.0.docx"):
    z = zipfile.ZipFile(os.path.join(OUT, f))
    xml = z.read("word/document.xml").decode("utf-8")
    leak = "{width=" in xml
    if leak:
        fails.append(f"{f} 正文残留 '{{width='")
    say(f"  [{'OK' if not leak else 'XX'}] {f} 无 '{{width=' 残留")

say("")
say("## 5. 构建核验（Word 版面与 PDF 字体嵌入）")

for f, need_double in (("manuscript_v4.0.docx", True),
                       ("Supplementary_Information_v4.0.docx", False)):
    z = zipfile.ZipFile(os.path.join(OUT, f))
    media = [n for n in z.namelist() if n.startswith("word/media/")]
    doc = z.read("word/document.xml").decode("utf-8")
    foot = "word/footer1.xml"
    checks = [
        ("A4 页面（w:w=11906 / w:h=16838）",
         'w:w="11906"' in doc and 'w:h="16838"' in doc),
        ('双倍行距 w:line="480"', 'w:line="480"' in doc),
        ("页脚 PAGE 域", foot in z.namelist()
         and "PAGE" in z.read(foot).decode("utf-8")),
        ("无目录域（期刊稿件不带 TOC）", "TOC" not in doc),
    ]
    say(f"  - {f}: 内嵌图件 {len(media)} 个")
    for lbl, cond in checks:
        if need_double or "双倍行距" not in lbl:
            if not cond:
                fails.append(f"{f} 缺 {lbl}")
            say(f"    [{'OK' if cond else 'XX'}] {lbl}")
        else:
            say(f"    [--] {lbl}：补充材料按单倍行距，不适用")

for f in ("manuscript_v4.0.pdf", "Supplementary_Information_v4.0.pdf"):
    p = os.path.join(OUT, f)
    rd = PdfReader(p)
    raw = open(p, "rb").read()
    embedded = raw.count(b"/FontFile2") + raw.count(b"/FontFile3") + raw.count(b"/FontFile")
    first = (rd.pages[0].extract_text() or "").strip()
    pn = re.search(r"(\d+)\s*/\s*(\d+)\s*$", first)
    ok_font = embedded > 0
    ok_pn = bool(pn)
    if not ok_font:
        fails.append(f"{f} 未嵌入字体（Unicode 字符可能丢字）")
    if not ok_pn:
        fails.append(f"{f} 首页未检出页脚页码")
    say(f"  - {f}: 嵌入字体程序 {embedded} 个；首页页脚页码 "
       f"{pn.group(0) if pn else '未检出'}")
    say(f"    [{'OK' if ok_font else 'XX'}] 字体已嵌入（保证 ⁻¹ ρ Δ 等字符不丢）")
    say(f"    [{'OK' if ok_pn else 'XX'}] 页脚页码已叠加")

say("")
say("## 6. 目视核验取样（渲染为 PNG 供逐页放大检查）")
# 交付物级核验面：版式缺陷（图片溢出、压边框、表格断行）只有看图才能发现，
# 与 Stage 4.5 清单 B 节同一理由。此处只取样若干页，输出到 08_定稿/_preview/。
PREV = os.path.join(OUT, "_preview")
try:
    import pymupdf
    os.makedirs(PREV, exist_ok=True)
    for pdf, pages, tag in (("manuscript_v4.0.pdf", [1, 21, 23], "MS"),
                            ("Supplementary_Information_v4.0.pdf", [1, 16], "SI")):
        doc = pymupdf.open(os.path.join(OUT, pdf))
        for pno in pages:
            if 1 <= pno <= doc.page_count:
                doc.load_page(pno - 1).get_pixmap(dpi=110).save(
                    os.path.join(PREV, f"{tag}_p{pno:02d}.png"))
        say(f"  - {pdf}: 取样页 {pages} / 共 {doc.page_count} 页")
        doc.close()
except ImportError:
    say("  - 跳过：未安装 pymupdf（pip install pymupdf）")

# --------------------------------------------------------------------------
# 7. 中文版构建（manuscript_v4_中文版.md / Supplementary_Information_v4_中文版.md）
# --------------------------------------------------------------------------
# 中文版的图注标签为 "**图 N |" / "**补充图 N |"，与英文版的 "**Figure N |"
# 前缀不同，故 inject_figures 必须传独立的插入正则，否则图件不会被插入。
say("")
say("## 7. 中文版构建与核验")
SRC_MS_CN = os.path.join(PIPE, "06_论文稿", "manuscript_v4_中文版.md")
SRC_SI_CN = os.path.join(PIPE, "06_论文稿", "Supplementary_Information_v4_中文版.md")

if not (os.path.exists(SRC_MS_CN) and os.path.exists(SRC_SI_CN)):
    say("  - 跳过：未找到中文版源稿（manuscript_v4_中文版.md / "
        "Supplementary_Information_v4_中文版.md）")
else:
    MS_CN_OUT = os.path.join(OUT, "manuscript_v4.0_中文版")
    SI_CN_OUT = os.path.join(OUT, "Supplementary_Information_v4.0_中文版")

    ms_cn = open(SRC_MS_CN, encoding="utf-8").read().replace("\r\n", "\n")
    si_cn = open(SRC_SI_CN, encoding="utf-8").read().replace("\r\n", "\n")
    ms_cn_fig = inject_figures(ms_cn, MAIN_FIGS, r"\*\*图 (\d) \|", REL_FIG)
    si_cn_fig = inject_figures(si_cn, SI_FIGS, r"\*\*补充图 (\d) \|", REL_FIG)

    MS_CN_MD = os.path.join(BUILD, "manuscript_cn_build.md")
    SI_CN_MD = os.path.join(BUILD, "si_cn_build.md")
    write(MS_CN_MD, ms_cn_fig)
    write(SI_CN_MD, si_cn_fig)

    n_ms, n_si = ms_cn_fig.count("![]("), si_cn_fig.count("![](")
    ok = (n_ms == len(MAIN_FIGS)) and (n_si == len(SI_FIGS))
    if not ok:
        fails.append(f"中文版图件插入数不符（正文 {n_ms} / 补充 {n_si}）")
    say(f"  - 图件插入：正文 {n_ms} 张 / 补充材料 {n_si} 张"
        f"（期望 {len(MAIN_FIGS)} / {len(SI_FIGS)}）  [{'OK' if ok else 'XX'}]")

    cn_docx_ms = to_docx(MS_CN_MD, MS_CN_OUT)
    cn_docx_si = to_docx(SI_CN_MD, SI_CN_OUT)
    polish_docx(cn_docx_ms, double_spaced=True)
    polish_docx(cn_docx_si, double_spaced=False)

    cn_html_ms = to_html(MS_CN_MD, os.path.join(BUILD, "manuscript_cn_build.html"),
                         "manuscript_v4.0_中文版")
    cn_html_si = to_html(SI_CN_MD, os.path.join(BUILD, "si_cn_build.html"),
                         "Supplementary_Information_v4.0_中文版")
    cn_pdf_ms = to_pdf(cn_html_ms, MS_CN_OUT)
    cn_pdf_si = to_pdf(cn_html_si, SI_CN_OUT)

    for p in (cn_docx_ms, cn_docx_si, cn_pdf_ms, cn_pdf_si):
        say(f"  - {os.path.relpath(p, PIPE)}  ({os.path.getsize(p) / 1024:.0f} KB)")

    CN_CHECKS = {
        "正文": (cn_pdf_ms, ["摘要", "引言", "结果", "讨论", "方法",
                            "数据可用性", "代码可用性", "References",
                            "致谢", "作者贡献", "利益冲突", "附加信息", "图注",
                            "林分结构", "R²", "×", "≈", "ρ", "Δ", "—",
                            "补充表7"], 6),
        "补充材料": (cn_pdf_si, ["补充信息", "补充方法", "补充表格", "补充图图注",
                              "补充表 1", "补充表 7", "ρ"], 4),
    }
    for tag, (pdf, tokens, n_expect_img) in CN_CHECKS.items():
        rd = PdfReader(pdf)
        txt = "\n".join((pg.extract_text() or "") for pg in rd.pages)
        say(f"  - 中文{tag}: {len(rd.pages)} 页，抽取文本 {len(txt)} 字符")
        for t in tokens:
            hit = t in txt
            if not hit:
                fails.append(f"中文{tag} PDF 缺 '{t}'")
            say(f"    [" + ("OK" if hit else "XX") + f"] 含 '{t}'")
        n_img = sum(len(pg.images) for pg in rd.pages)
        ok_img = n_img >= n_expect_img
        if not ok_img:
            fails.append(f"中文{tag} PDF 内嵌位图 {n_img} < {n_expect_img}")
        say(f"    [" + ("OK" if ok_img else "XX")
            + f"] PDF 内嵌位图 {n_img} 个（期望 ≥ {n_expect_img}）")
        leak = "{width=" in txt
        if leak:
            fails.append(f"中文{tag} 版面上残留未解析的图片属性")
        say("    [" + ("OK" if not leak else "XX")
            + "] 无未解析的图片属性残留（'{width='）")

    for f in ("manuscript_v4.0_中文版.docx", "Supplementary_Information_v4.0_中文版.docx"):
        z = zipfile.ZipFile(os.path.join(OUT, f))
        xml = z.read("word/document.xml").decode("utf-8")
        leak = "{width=" in xml
        if leak:
            fails.append(f"{f} 正文残留 '{{width='")
        say("    [" + ("OK" if not leak else "XX") + f"] {f} 无 '{{width=' 残留")

    for p, f in ((cn_pdf_ms, "manuscript_v4.0_中文版.pdf"),
                 (cn_pdf_si, "Supplementary_Information_v4.0_中文版.pdf")):
        rd = PdfReader(p)
        raw = open(p, "rb").read()
        emb = (raw.count(b"/FontFile2") + raw.count(b"/FontFile3")
               + raw.count(b"/FontFile"))
        first = (rd.pages[0].extract_text() or "").strip()
        pn = re.search(r"(\d+)\s*/\s*(\d+)\s*$", first)
        ok_font, ok_pn = emb > 0, bool(pn)
        if not ok_font:
            fails.append(f"{f} 未嵌入字体（汉字可能丢字）")
        if not ok_pn:
            fails.append(f"{f} 首页未检出页脚页码")
        say(f"  - {f}: 嵌入字体程序 {emb} 个；首页页脚页码 "
            f"{pn.group(0) if pn else '未检出'}")
        say("    [" + ("OK" if ok_font else "XX")
            + "] 字体已嵌入（保证汉字与 ⁻¹ ρ Δ 不丢）")
        say("    [" + ("OK" if ok_pn else "XX") + "] 页脚页码已叠加")

say("")
say("## 结论")
if fails:
    say(f"**FAIL** — {len(fails)} 项：")
    for f in fails:
        say(f"- {f}")
else:
    say("**PASS** — 全部文件（英文版 + 中文版，各 docx/PDF）均已生成，"
        "关键章节与 Unicode 字符完整，图件已内嵌。")

write(os.path.join(OUT, "00_构建核验报告.md"), "\n".join(report))
say("")
say("DONE 36 build submission docs")
if fails:
    raise SystemExit(1)