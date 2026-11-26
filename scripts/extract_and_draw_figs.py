# -*- coding: utf-8 -*-
"""
Cloud-side figure extractor & publication-grade diagram generator.
Runs on GitHub Actions ubuntu-latest (zero local encryption involved).

Extracts 21 genuine figures directly from:
  - animals-16-01417-v2.pdf (9 figures: fig2_1, fig2_2, fig3_1..fig3_7)
  - agriculture-16-01845.pdf (12 ultra-HD figures: fig4_1..fig4_12)
And draws 3 crisp bilingual vector-quality diagrams with matplotlib + Noto Sans CJK:
  - fig1_1.png (Overall technical roadmap)
  - fig5_1.png (5-layer integrated monitoring & warning architecture)
  - fig5_2.png (Collaborative workflow of warning & abnormality verification)
"""
import io
import os
import fitz
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

FIG_DIR = "/tmp/thesis_figs"
os.makedirs(FIG_DIR, exist_ok=True)


def find_cjk_font():
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    for f in font_manager.findSystemFonts():
        low = f.lower()
        if "notosanscjk" in low or "wqy" in low or "simhei" in low or "msyh" in low:
            return f
    return None


CJK_FONT_PATH = find_cjk_font()
if CJK_FONT_PATH:
    font_manager.fontManager.addfont(CJK_FONT_PATH)
    CJK_PROP = font_manager.FontProperties(fname=CJK_FONT_PATH)
    plt.rcParams["font.sans-serif"] = [CJK_PROP.get_name(), "DejaVu Sans"]
else:
    CJK_PROP = None
plt.rcParams["axes.unicode_minus"] = False


def load_pil_from_pdf_page_img(doc, page_1based, img_idx_0based):
    """Extract embedded image by page (1-based) in visual reading order (top-to-bottom, left-to-right)."""
    pg = doc[page_1based - 1]
    seen = set()
    large = []
    for im in pg.get_images(full=True):
        xref, w, h = im[0], im[2], im[3]
        if xref in seen or w < 120 or h < 120:
            continue
        seen.add(xref)
        rects = pg.get_image_rects(xref)
        if rects:
            r = rects[0]
            sort_key = (int(round(r.y0 / 40.0)), r.x0)
        else:
            sort_key = (9999, xref)
        large.append((sort_key, xref))
    large.sort(key=lambda item: item[0])
    assert img_idx_0based < len(large), f"Page {page_1based} only has {len(large)} large imgs, asked #{img_idx_0based}"
    xref = large[img_idx_0based][1]
    base = doc.extract_image(xref)
    pil = Image.open(io.BytesIO(base["image"]))
    if pil.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", pil.size, (255, 255, 255))
        bg.paste(pil, mask=pil.split()[-1])
        pil = bg
    elif pil.mode != "RGB":
        pil = pil.convert("RGB")
    return pil


def save_clean(pil_img, out_name, max_dim=2200, prefer_jpeg=False):
    w, h = pil_img.size
    scale = min(1.0, max_dim / float(max(w, h)))
    if scale < 1.0:
        pil_img = pil_img.resize((int(round(w * scale)), int(round(h * scale))), Image.LANCZOS)
    out_path = os.path.join(FIG_DIR, out_name)
    if prefer_jpeg or out_name.lower().endswith(".jpg"):
        pil_img.save(out_path, "JPEG", quality=92, optimize=True)
    else:
        pil_img.save(out_path, "PNG", optimize=True)
    return out_path


def stitch_grid(imgs, rows, cols, labels=None, pad=28, label_h=44):
    """Stitch subfigures into a clean publication grid with crisp (a)-(f) labels below each panel."""
    # Normalize each cell width to median/max reasonable size
    target_w = min(900, max(im.size[0] for im in imgs))
    resized = []
    for im in imgs:
        w, h = im.size
        nh = int(round(h * (target_w / float(w))))
        resized.append(im.resize((target_w, nh), Image.LANCZOS))
    cell_h = max(im.size[1] for im in resized)
    total_w = cols * target_w + (cols + 1) * pad
    total_h = rows * (cell_h + (label_h if labels else 0)) + (rows + 1) * pad
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(CJK_FONT_PATH or "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except Exception:
        font = ImageFont.load_default()

    for idx, im in enumerate(resized):
        r = idx // cols
        c = idx % cols
        x = pad + c * (target_w + pad)
        y = pad + r * (cell_h + (label_h if labels else 0) + pad)
        # Center vertically inside cell
        y_off = y + (cell_h - im.size[1]) // 2
        canvas.paste(im, (x, y_off))
        if labels and idx < len(labels):
            lbl = labels[idx]
            bbox = draw.textbbox((0, 0), lbl, font=font)
            tw = bbox[2] - bbox[0]
            draw.text((x + (target_w - tw) // 2, y + cell_h + 6), lbl, fill=(20, 20, 20), font=font)
    return canvas


def extract_all_pdf_figures():
    print("Extracting figures from animals-16-01417-v2.pdf ...")
    ani = fitz.open("animals-16-01417-v2.pdf")
    # Chapter 2: ECA & GRU schematics (Page 10: 2 images -> Fig. 4 ECA, Fig. 5 GRU)
    save_clean(load_pil_from_pdf_page_img(ani, 10, 0), "fig2_1_eca.png", max_dim=1600)
    save_clean(load_pil_from_pdf_page_img(ani, 10, 1), "fig2_2_gru.png", max_dim=1600)

    # Chapter 3: Figs 1, 2, 3, 6 (2-panel), 7 (6-panel), 8 (6-panel), 9
    save_clean(load_pil_from_pdf_page_img(ani, 4, 0), "fig3_1_layout.png", max_dim=1800)
    save_clean(load_pil_from_pdf_page_img(ani, 5, 0), "fig3_2_sys_arch.png", max_dim=2000)
    p16_imgs = [load_pil_from_pdf_page_img(ani, 16, 0), load_pil_from_pdf_page_img(ani, 16, 1)]
    save_clean(stitch_grid(p16_imgs, 1, 2, labels=["(a) TD 候选特征相关性", "(b) THI 候选特征相关性"]), "fig3_3_pearson.png", max_dim=2000)
    save_clean(load_pil_from_pdf_page_img(ani, 7, 0), "fig3_4_model_arch.png", max_dim=1800)

    p17_imgs = [load_pil_from_pdf_page_img(ani, 17, i) for i in range(6)]
    lbls6 = ["(a) LSTM", "(b) TCN", "(c) GRU", "(d) ECA-GRU", "(e) Wavelet-GRU", "(f) Wavelet-ECA-GRU"]
    save_clean(stitch_grid(p17_imgs, 2, 3, labels=lbls6), "fig3_5_thi_curves.png", max_dim=2200)

    p19_imgs = [load_pil_from_pdf_page_img(ani, 19, i) for i in range(6)]
    save_clean(stitch_grid(p19_imgs, 2, 3, labels=lbls6), "fig3_6_td_curves.png", max_dim=2200)

    save_clean(load_pil_from_pdf_page_img(ani, 24, 0), "fig3_7_warning.png", max_dim=1800)
    ani.close()

    print("Extracting figures from agriculture-16-01845.pdf ...")
    agr = fitz.open("agriculture-16-01845.pdf")
    agr_map = [
        (5, "fig4_1_field.jpg", 2000, True),
        (6, "fig4_2_dt_arch.png", 2000, False),
        (7, "fig4_3_data_flow.png", 2000, False),
        (8, "fig4_4_blender.jpg", 2000, True),
        (10, "fig4_5_param_anchors.jpg", 2000, True),
        (13, "fig4_6_semantic_flow.png", 2000, False),
        (15, "fig4_7_affine_flow.png", 2000, False),
        (20, "fig4_8_webgl_ui.jpg", 2000, True),
        (21, "fig4_9_fps.png", 1800, False),
        (23, "fig4_10_deadbird_3d.jpg", 2000, True),
        (25, "fig4_11_robot_sync.jpg", 1800, True),
        (26, "fig4_12_robot_ctrl.jpg", 1800, True),
    ]
    for pg_num, fname, mdim, is_jpg in agr_map:
        pil = load_pil_from_pdf_page_img(agr, pg_num, 0)
        save_clean(pil, fname, max_dim=mdim, prefer_jpeg=is_jpg)
    agr.close()


# =====================================================================
# Crisp Vector-Style Diagram Generators (Fig 1-1, Fig 5-1, Fig 5-2)
# =====================================================================
def draw_box(ax, x, y, w, h, text, fc="#F0F5FA", ec="#2B5B84", lw=1.4, fontsize=9.5, bold=False, text_color="#1A1A1A"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2.0, y + h / 2.0, text,
        ha="center", va="center", fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color=text_color, fontproperties=CJK_PROP, zorder=4, linespacing=1.35
    )


def draw_arrow(ax, x1, y1, x2, y2, color="#3A506B", lw=1.4, style="-|>", rad=0.0):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=13,
        color=color, linewidth=lw,
        connectionstyle=f"arc3,rad={rad}", zorder=2
    )
    ax.add_patch(arr)


def draw_fig1_1_roadmap():
    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=240)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Top problem & objective banner
    draw_box(
        ax, 0.14, 0.88, 0.72, 0.095,
        "研究目标与问题切入（第1–2章）\n规模化笼养肉鸡舍热环境风险滞后 · 异常空间定位不直观 · 巡检管理效率受限",
        fc="#1F4E79", ec="#163A5C", lw=1.6, fontsize=10.5, bold=True, text_color="white"
    )

    # Branch Containers
    b1 = FancyBboxPatch((0.03, 0.25), 0.44, 0.58, boxstyle="round,pad=0.01,rounding_size=0.02",
                        facecolor="#F4F8FC", edgecolor="#2E75B6", linewidth=1.4, linestyle="--", zorder=1)
    b2 = FancyBboxPatch((0.53, 0.25), 0.44, 0.58, boxstyle="round,pad=0.01,rounding_size=0.02",
                        facecolor="#F4FAF6", edgecolor="#2E7D5B", linewidth=1.4, linestyle="--", zorder=1)
    ax.add_patch(b1)
    ax.add_patch(b2)

    ax.text(0.25, 0.805, "分支一：环境监测与短期预测预警（第3章 · 德州5周试验）",
            ha="center", va="center", fontsize=9.8, fontweight="bold", color="#1F4E79", fontproperties=CJK_PROP)
    ax.text(0.75, 0.805, "分支二：数字孪生监控平台构建（第4章 · 高邮100天试验）",
            ha="center", va="center", fontsize=9.8, fontweight="bold", color="#1E5631", fontproperties=CJK_PROP)

    # Left branch boxes
    left_steps = [
        (0.68, "多源环境感知体系构建\n巡检机器人 + 12固定点 + 1室外点（101 m×16 m）"),
        (0.555, "数据对齐、清洗与样本构建\n3 s对齐 → 5 min聚合（8458条）→ 36步输入/6步预测"),
        (0.43, "Wavelet-ECA-GRU 双任务独立预测\nPearson特征筛选 + db4三层小波 + ECA + GRU"),
        (0.29, "双指标在线滚动预警与验证\nTD(R²=0.9238) / THI(R²=0.9287) · 30 min超前预警"),
    ]
    for y, txt in left_steps:
        draw_box(ax, 0.06, y, 0.38, 0.092, txt, fc="white", ec="#2E75B6", lw=1.2, fontsize=8.8)

    # Right branch boxes
    right_steps = [
        (0.68, "五层架构与参数化三维场景构建\nBlender建模 + 差异化轻量化(-48.34%面片) + 1764锚点"),
        (0.555, "SLAM位姿驱动的异常语义定位\n通道/相机映射 + 笼组计算(W=1.25 m) + 分层表达"),
        (0.43, "二维仿射坐标标定与虚实同步\n6点最小二乘拟合(0.0356 m) + LOOCV(<0.1 m)"),
        (0.29, "Unity WebGL 平台部署与现场实测\n死鸡定位96.0%(1954/2035) · 67.83 fps · 控制99.50%"),
    ]
    for y, txt in right_steps:
        draw_box(ax, 0.56, y, 0.38, 0.092, txt, fc="white", ec="#2E7D5B", lw=1.2, fontsize=8.8)

    # Vertical arrows inside branches
    draw_arrow(ax, 0.35, 0.88, 0.25, 0.83)
    draw_arrow(ax, 0.65, 0.88, 0.75, 0.83)
    for y_top, y_bot in [(0.68, 0.647), (0.555, 0.522), (0.43, 0.382)]:
        draw_arrow(ax, 0.25, y_top, 0.25, y_bot, color="#2E75B6")
        draw_arrow(ax, 0.75, y_top, 0.75, y_bot, color="#2E7D5B")

    # Convergence to Ch5 and Ch6
    draw_box(
        ax, 0.10, 0.13, 0.80, 0.085,
        "时间维度趋势预测 × 空间维度语义表达：五层综合监控预警应用框架（第5章 · 方案设计边界明确）\n统一数据标识 · 全局预警与点位实测分离表达 · 异常核验与巡检任务人机协同",
        fc="#FFF8E7", ec="#C58B2A", lw=1.4, fontsize=9.0, bold=False
    )
    draw_box(
        ax, 0.22, 0.02, 0.56, 0.072,
        "研究结论、创新贡献、现阶段局限与跨场联合验证展望（第6章）",
        fc="#EEF2F6", ec="#4A607A", lw=1.3, fontsize=9.3, bold=True
    )
    draw_arrow(ax, 0.25, 0.25, 0.38, 0.215, color="#3A506B")
    draw_arrow(ax, 0.75, 0.25, 0.62, 0.215, color="#3A506B")
    draw_arrow(ax, 0.50, 0.13, 0.50, 0.092, color="#3A506B")

    plt.tight_layout(pad=0.2)
    out = os.path.join(FIG_DIR, "fig1_1_roadmap.png")
    fig.savefig(out, dpi=240, facecolor="white")
    plt.close(fig)


def draw_fig5_1_integrated_arch():
    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=240)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    layers = [
        (0.82, "#1F4E79", "#EBF3FA", "业务应用层\nBusiness Application",
         "整体TD/THI预警趋势面板  |  异常事件筛选与人工核验  |  机器人巡检任务协同与授权控制（辅助决策边界）"),
        (0.64, "#2E75B6", "#F0F7FC", "数字孪生表达层\nDigital Twin View",
         "Unity WebGL 参数化三维鸡舍  |  1764 语义笼位锚点标记  |  虚拟机器人仿射同步  |  全局预警状态栏（与点位实测分离）"),
        (0.44, "#2E7D5B", "#EEF8F3", "分析服务层\nAnalysis Services",
         "【独立服务A：环境预测预警（时间维度）】                      【独立服务B：异常语义空间映射（空间维度）】\nWavelet-ECA-GRU 30 min超前整体TD/THI预测与双指标规则        SLAM位姿+通道+相机 → 死鸡笼位级 / 张嘴红外声音分层映射"),
        (0.24, "#9E6B1E", "#FDF6E9", "数据接入与管理层\nData Access & Mgmt",
         "统一鸡舍/设备/时间戳标识  |  3 s对齐与5 min窗口聚合  |  图像/红外/音频证据封装  |  数据质量标记与人工核验留痕"),
        (0.06, "#546E7A", "#F2F5F7", "现场感知层\nField Sensing Layer",
         "室内固定监测点（温/湿/风速/CO₂） +  室外气象监测点  +  SLAM移动巡检机器人（双光谱可见光/红外、全向麦克风、环境传感器）"),
    ]

    for y, hc, bc, lname, desc in layers:
        h = 0.145 if "分析服务层" in lname else 0.13
        draw_box(ax, 0.03, y, 0.20, h, lname, fc=hc, ec=hc, lw=1.2, fontsize=9.2, bold=True, text_color="white")
        draw_box(ax, 0.24, y, 0.73, h, desc, fc=bc, ec=hc, lw=1.2, fontsize=8.6)

    # Upward data flow arrows & downward authorized control arrow
    for y1, y2 in [(0.19, 0.24), (0.37, 0.44), (0.585, 0.64), (0.77, 0.82)]:
        draw_arrow(ax, 0.50, y1, 0.50, y2, color="#2B5B84", lw=1.6)
    draw_arrow(ax, 0.93, 0.82, 0.93, 0.19, color="#B23A48", lw=1.4, style="-|>")
    ax.text(0.955, 0.50, "经权限校验的\n控制指令与反馈", rotation=90, ha="center", va="center",
            fontsize=8.2, color="#B23A48", fontweight="bold", fontproperties=CJK_PROP)

    ax.text(0.50, 0.015,
            "注：本架构基于德州环境预测研究与高邮数字孪生研究提出，属于综合应用设计方案，尚未在同一鸡舍完成跨模块联合部署验证。",
            ha="center", va="center", fontsize=8.0, color="#555555", fontproperties=CJK_PROP)

    plt.tight_layout(pad=0.2)
    out = os.path.join(FIG_DIR, "fig5_1_framework.png")
    fig.savefig(out, dpi=240, facecolor="white")
    plt.close(fig)


def draw_fig5_2_collab_flow():
    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=240)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Left lane: Environmental Warning
    draw_box(ax, 0.03, 0.76, 0.28, 0.16, "1. 多源环境实时监测\n固定点 + 机器人 + 室外点\n3 s 对齐 → 5 min 窗口聚合",
             fc="#EBF3FA", ec="#2E75B6", fontsize=8.8)
    draw_box(ax, 0.03, 0.48, 0.28, 0.18, "2. 30 min 超前滚动预测\nWavelet-ECA-GRU 独立模型\nTD（±2.5 ℃）与 THI（四级）",
             fc="#EBF3FA", ec="#2E75B6", fontsize=8.8)
    draw_box(ax, 0.03, 0.18, 0.28, 0.19, "3. 整体热环境风险提示\n区分过热/过冷偏差与湿热风险\n在全局预警面板展示趋势曲线",
             fc="#FFF4E0", ec="#C58B2A", fontsize=8.8, bold=False)

    # Middle: Human-in-the-loop decision hub
    draw_box(ax, 0.37, 0.44, 0.26, 0.24, "4. 管理人员综合研判\n（人机协同安全边界）\n结合鸡群采食/饮水/呼吸状态\n决定是否发起补充巡检或\n核查通风/供暖/降温设备",
             fc="#FFF0F0", ec="#B23A48", lw=1.6, fontsize=8.8, bold=True)

    # Right lane: Digital Twin Spatial Verification & Robot Synergy
    draw_box(ax, 0.69, 0.76, 0.28, 0.16, "5. 巡检任务交互与同步\nWebGL提交经校验的任务指令\n仿射标定同步机器人位姿状态",
             fc="#EEF8F3", ec="#2E7D5B", fontsize=8.8)
    draw_box(ax, 0.69, 0.48, 0.28, 0.18, "6. 异常事件分层空间映射\n死鸡→笼位级(Cxx-Txx-Gxxx)\n张嘴/红外/声音→区域或通道级",
             fc="#EEF8F3", ec="#2E7D5B", fontsize=8.8)
    draw_box(ax, 0.69, 0.18, 0.28, 0.19, "7. 现场精准核验与闭环记录\n点击三维锚点追溯图像/环境证据\n定位字段缺失自动转人工核验",
             fc="#EEF8F3", ec="#2E7D5B", fontsize=8.8)

    # Bottom note
    ax.text(0.50, 0.06,
            "边界说明：预测预警信息用于辅助管理人员决策，不直接自动触发环控执行器或未经人工确认的自动巡检调度。",
            ha="center", va="center", fontsize=8.3, color="#444444", fontproperties=CJK_PROP)

    # Arrows
    draw_arrow(ax, 0.17, 0.76, 0.17, 0.66, color="#2E75B6")
    draw_arrow(ax, 0.17, 0.48, 0.17, 0.37, color="#2E75B6")
    draw_arrow(ax, 0.31, 0.28, 0.37, 0.52, color="#C58B2A")
    draw_arrow(ax, 0.63, 0.59, 0.69, 0.82, color="#B23A48")
    draw_arrow(ax, 0.83, 0.76, 0.83, 0.66, color="#2E7D5B")
    draw_arrow(ax, 0.83, 0.48, 0.83, 0.37, color="#2E7D5B")
    draw_arrow(ax, 0.69, 0.28, 0.63, 0.50, color="#2E7D5B")

    plt.tight_layout(pad=0.2)
    out = os.path.join(FIG_DIR, "fig5_2_collab_flow.png")
    fig.savefig(out, dpi=240, facecolor="white")
    plt.close(fig)


def main():
    extract_all_pdf_figures()
    draw_fig1_1_roadmap()
    draw_fig5_1_integrated_arch()
    draw_fig5_2_collab_flow()
    files = sorted(os.listdir(FIG_DIR))
    print(f"Prepared {len(files)} figures in {FIG_DIR}:")
    for fn in files:
        sz = os.path.getsize(os.path.join(FIG_DIR, fn))
        print(f"  {fn}: {sz // 1024} KB")
    assert len(files) == 24, f"Expected 24 figures, got {len(files)}"


if __name__ == "__main__":
    main()
