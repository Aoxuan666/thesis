# Revision Report: `英文稿子v12.docx` → `英文稿子v13.docx`

**Scope** — structural and textual comparison of two manuscripts of the same paper, performed directly on the OOXML package (`word/document.xml`, `styles.xml`), not on rendered text. Both files were opened read-only; no tracked changes (`w:ins`/`w:del`) exist in either version, so every difference below is a hard rewrite, not a revision mark.

| Item | v12 | v13 |
|---|---|---|
| File size | 81,802,909 B (78.0 MiB) | 74,922,625 B (71.5 MiB) |
| Pages (Word, A4) | 29 | 21 |
| Words (Word count, incl. tables) | 10,242 | 7,664 |
| Body paragraphs / body words | 285 / 9,729 | 167 / 6,889 |
| Tables | 27 (7 data + 19 equation + 1 layout) | 17 (7 data + 9 equation + 1 layout) |
| Figures | 9 | 9 |
| Reference entries | 29 | 29 |
| Citation range in body | [1]–[28], [31]–[35] | [1]–[29] |
| Abstract length | 185 words | 239 words |

---

## 1. Executive summary

1. **v13 is a compressive revision of v12, not a different paper.** Title, author block, data (8,092 observations; 24,276 configuration–time cases per method; 48,552 evaluated records), the 7 data tables and all 9 figures are carried over unchanged in substance.
2. **Total length dropped 25%** by Word's own count (10,242 → 7,664 words) and 8 pages (29 → 21). Almost every section lost 25–55% of its words; the reference list is the only block that is byte-for-byte the same length.
3. **Four substantive repairs were made** that are not mere shortening: a missing `3. Results` level-1 heading was added (v12 hung 3.1–3.7 under 2.5), the equation set was rebuilt around a QP/coordinate-descent formulation (19 → 9 numbered equations), the dangling citations [31]–[35] were removed, and the Discussion subsection 4.3 was rewritten from scratch.
4. **The abstract grew by 54 words** (185 → 239), which now exceeds the usual 200-word MDPI limit — the single most likely desk-rejection trigger left in v13.
5. Terminology was narrowed from "commercial broiler houses" to **"caged broiler houses"** throughout, including the title and keywords.

---

## 2. Structural and formatting repairs

| # | Change | Evidence |
|---|---|---|
| 1 | **`3. Results` heading added** | v12 has 3.1–3.7 as level-2 headings with no level-1 parent (they dangle inside 2.5); v13 inserts a level-1 `3. Results` at the same point |
| 2 | **Title restyled and rescoped** | v12 title sits in style `MDPI_1.3_authornames` and says "Commercial Broiler Houses"; v13 uses the proper `MDPI_1.2_title` style and says "Caged Broiler Houses" |
| 3 | **Author line spacing fixed** | v12: `Jingkun Sun 1,2,Jiaze Sun 1,2` (missing space); v13: `… 1,2, Jiaze Sun 1,2` |
| 4 | **All 7 table captions normalised** | v12 mixes `MDPI_4.1_table_caption` with body style `MDPI_3.1_text`; v13 puts all 7 captions in `MDPI_4.1_table_caption` |
| 5 | **Keywords brought to journal style** | v12: `Broiler houses; Environmental control; …` (title case); v13: `caged broiler houses; environmental control; control allocation; …` (lower case, MDPI convention) |
| 6 | **Abstract rewritten around a mechanism, not a topic** | v12 opens "Environmental regulation in commercial broiler houses is complicated by differences in equipment configurations…"; v13 opens with the concrete hardware asymmetry (fan counts, VSDs, inlet actuators, relay groupings) that motivates the method |
| 7 | **Figure captions made self-contained** | e.g. Fig. 8 v12 "Raspberry Pi-based edge control platform and physical equipment execution" → v13 "Raspberry Pi 5 edge control testbed: (a) 16-channel relay output module …" |

---

## 3. Section-by-section length change

Word counts are body text only (headings, equation tables and table cells excluded), computed per section boundary.

| Section | v12 | v13 | Δ | Δ% | Paragraphs |
|---|---:|---:|---:|---:|---|
| Front matter (title/abstract/keywords) | 281 | 336 | +55 | +19.6% | 10 → 10 |
| 1. Introduction | 1,112 | 838 | −274 | −24.6% | 12 → 8 |
| 2.1. Overall System Architecture | 285 | 193 | −92 | −32.3% | 5 → 3 |
| 2.2. Environmental State and Regulation Demand Modeling | 343 | 210 | −133 | −38.8% | 12 → 3 |
| 2.3. Parameterized Equipment Capability Modeling | 475 | 261 | −214 | −45.1% | 20 → 5 |
| 2.4. Hierarchical Control and Optimization Allocation | 501 | 371 | −130 | −25.9% | 18 → 7 |
| 2.5.1. Field Monitoring Data and Preprocessing | 263 | 177 | −86 | −32.7% | 6 → 2 |
| 2.5.2. Equipment Configuration**s** and **Baseline Threshold Allocator** (retitled) | 297 | 201 | −96 | −32.3% | 6 → 2 |
| 2.5.3. Evaluation Indicators | 272 | 154 | −118 | −43.4% | 9 → 2 |
| 2.6. Edge Deployment and Physical Equipment Execution | 361 | 198 | −163 | −45.2% | 8 → 3 |
| 2.7. Cloud Communication and Three-Dimensional Visualization | 254 | 131 | −123 | −48.4% | 6 → 2 |
| **3. Results** (new level-1 heading) | — | — | — | — | — |
| 3.1. Field Environmental Conditions and Regulation Demands | 449 | 309 | −140 | −31.2% | 5 → 6 |
| 3.2. Control Allocation Performance under Different Equipment Configurations | 290 | 184 | −106 | −36.6% | 7 → 5 |
| 3.3. Regulation-Demand Tracking and Equipment Operating Characteristics | 388 | 310 | −78 | −20.1% | 12 → 9 |
| 3.4. Equipment Allocation Behavior under Representative Environmental Conditions | 488 | 320 | −168 | −34.4% | 7 → 4 |
| 3.5. Constraint Satisfaction and Numerical Consistency | 193 | 92 | −101 | −52.3% | 5 → 2 |
| 3.6. Edge Control Implementation and Physical Equipment Execution | 333 | 149 | −184 | −55.3% | 8 → 4 |
| 3.7. Cloud-Based Monitoring and Three-Dimensional Visualization | 215 | 88 | −127 | −59.1% | 5 → 2 |
| 4.1. Adaptability **of the Method to Different Equipment Configurations** → **Portability Across Heterogeneous Equipment Configurations** | 398 | 215 | −183 | −46.0% | 5 → 2 |
| 4.2. Environmental Regulation … → **Multi-Channel Trade-Offs** and Coordinated Equipment Operation | 437 | 323 | −114 | −26.1% | 6 → 3 |
| 4.3. Integration of Control Allocation with Edge-Based Execution and Remote Management → **Edge-Level Real-Time Execution and Supervisory Integration** (rewritten) | 404 | 173 | −231 | −57.2% | 7 → 2 |
| 4.4. Limitations and **Further** → **Future** Work | 231 | 227 | −4 | −1.7% | 4 → 3 |
| 5. Conclusions | 448 | 425 | −23 | −5.1% | 10 → 9 |
| References | 859 | 859 | 0 | 0.0% | 30 → 30 |
| **Total body text** | **9,729** | **6,889** | **−2,840** | **−29.2%** | 249 → 155 |

Two anomalies worth noting:

- **3.1 gained a paragraph** (5 → 6) while losing 140 words: a short lead-in was split off so the numbers are not buried in one block.
- **4.4 barely changed** (−4 words): the limitations are the one place where the authors resisted compression, which is the right call for reviewers.

---

## 4. Equation set: 19 → 9 (rewritten, not deleted)

The equation count drop is **not** a loss of modelling content. v12 devotes 19 numbered display equations to definitions that v13 folds into running text, and spends the recovered space on the solver derivation that reviewers normally ask for.

| v13 eq. | Content | Where it came from in v12 |
|---|---|---|
| (1) | clipped ventilation / thermal / moisture demand laws | v12 eqs. (2)–(4), merged into one line |
| (2) | modelled contribution `y_k = B(H)u_k` | v12 eqs. (6)–(10) |
| (3) | admissible set (binary, continuous, heating/cooling interlock) | v12 eqs. (7), (11) |
| (4) | MIQP objective in one expression | v12 eqs. (12)–(17) |
| (5) | QP condensation over continuous units | **new** |
| (6) | Hessian `H_c` and gradient `f_k` for the condensation | **new** |
| (7) | closed-form cyclic-coordinate-descent update with clipping | **new** |
| (8) | binary enumeration outer loop + concatenation | **new** |
| (9) | channel-wise RMSE | v12 eq. (19) |

Consequence: v13 states why the edge controller can solve the problem deterministically without an MIQP library, and Table 6 now validates it against **L-BFGS-B**. If a reviewer asks "how is this solved on a Raspberry Pi 5", v13 answers it; v12 did not.

One definition appears to have been dropped rather than moved: v12 eq. (18) defined the relative reduction `R = (I_ref − I_pro)/I_ref × 100%` used for the "Reduction" column of Tables 2 and 5. In v13 the column survives, so confirm that `R` is still defined somewhere in 2.5.3 or in the table footnote.

---

## 5. Citation integrity (a real repair)

| | v12 | v13 |
|---|---|---|
| Reference entries | 29 | 29 (identical text, identical order) |
| Citations used in body | 1–28, **31–35** | **1–29** |
| Dangling citations | **[31–35] — 5 references that do not exist in the list** | none |
| Uncited entries | [29], [30] | none |

The v12 defects were concentrated in the old 4.3:

- `… manufacturer's user manual and RS485 communication example [34,35]` — entries 34 and 35 were never in the list.
- `… environmental sensing, information transmission, and remote management [31–33]` — entries 31–33 likewise missing.

v13 rewrites that subsection and re-cites IoT-platform work as **[24–27]**, which are entries that do exist. Every one of the 29 entries is now cited at least once, and no citation exceeds the list length.

---

## 6. Tables and figures

- **All 7 data tables are present in both versions with identical dimensions**: Table 1 (11×8), Table 2 (5×4), Table 3 (5×4), Table 4 (10×4), Table 5 (4×5), Table 6 (5×5), Table 7 (6×4). No result table was dropped.
- The 10 "missing" tables in the count are **equation containers** (1×2 layout tables), not data.
- **All 9 figures are present**, and the embedded media are identical (11 images in both packages). Every figure caption was rewritten for specificity.
- Table captions were reworded for scope, e.g. Table 3 "Comparison of control allocation performance indicators" → "Comparison of **pooled** control allocation performance indicators"; Table 4 "RMSEs of normalized regulation effects" → "**Channel-wise** RMSEs …".

---

## 7. What was actually cut

The deletions are overwhelmingly **rhetorical padding, not evidence**:

- Repeated restatements of the same limitation in 3.5–3.7 (each section previously ended with a summary sentence that repeated the numbers above it).
- Methodological narration in 2.2–2.3 that re-explained each symbol after defining it in an equation.
- The old 4.3, whose seven paragraphs each restated "modelling and execution are separated" with a different citation; the new 4.3 makes the same three points in two paragraphs (real-time constraint, supervisory integration, prior IoT platforms).
- Introduction cut from 12 paragraphs to 8 by merging the gap statement and the contribution list.

Nothing in the surviving text contradicts v12: the field site (Gaoyou, Jiangsu), the two houses, the 42-day monitoring window, the three equipment configurations with 6/6/7 decision variables and 24/12/12 admissible binary combinations, the objective weights (`W = diag(2.0, 3.0, 1.2)`, `λ_E = 0.12`, `λ_S = 0.075`) and the hardware inventory are all unchanged.

---

## 8. Open items before submission

- [ ] **Abstract is 239 words** — trim to ≤200 if the target journal enforces the MDPI limit.
- [ ] **Confirm the definition of the relative reduction `R`** (v12 eq. 18) still appears in text or as a table footnote.
- [ ] **Re-check equation numbering references** — v13 text refers to "Equation (4)", "(5)", "(7)" etc. after the renumbering; a full pass over cross-references is cheap insurance.
- [ ] **Check that "caged" is now the exclusive scope claim** — if any experiment used floor-reared birds, the narrowed title over-claims.
- [ ] **Verify §4.1's new title matches its content** ("Portability Across Heterogeneous Equipment Configurations" — the section must now argue portability, not just adaptability).
- [ ] **Figure resolution** — the file is 71.5 MB, dominated by images; MDPI requires ≥300 dpi (≥1000 dpi for line art) and will re-check TIFF/EPS conversion.

---

## Appendix — how this comparison was produced

1. Both `.docx` files were unzipped and `word/document.xml` parsed with a streaming XML reader; body children were walked in document order, so paragraph/table interleaving is preserved.
2. Paragraph text was reconstructed from `w:t`, `w:tab`, `w:br` runs; display equations were read separately from `m:t` (Office Math) nodes, because they are invisible to plain-text extraction.
3. Sections were cut at `MDPI_2.x_heading*` style boundaries, and paragraphs were aligned inside each section pair with a longest-common-subsequence diff (`difflib.SequenceMatcher`, no autojunk) to separate deletions, insertions and in-place rewrites.
4. Tables were compared by row count, column count and header row; equation tables were classified by their 1×2 geometry and excluded from the data-table inventory.
5. Word-level statistics (pages, words) were taken from Word itself via COM, not estimated.
