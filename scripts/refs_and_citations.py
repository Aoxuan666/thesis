# -*- coding: utf-8 -*-
"""
Verified Bibliography & Author-Year Citation Database strictly following
TJAU 2019 Attachment 5 (天津农学院研究生学位论文“参考文献”著录规则):
  - In-text: Author-Year format (著者-出版年制)
      - Parenthetical: (Author Year) or (Author1 and Author2 Year) or (Author1 et al. Year)
      - Multi-citation: sorted alphabetically by author, separated by '; '
      - Same author & year: 2026a, 2026b
  - Reference list:
      - NO numbering
      - English entries first (alphabetical by first author surname), then Chinese entries (Pinyin order)
      - ALL authors listed in full (NO 'et al.')
      - Foreign journal names italicized (marked with <<I>>...<</I>>)
      - Format: Authors. Year. Title. <<I>>Journal<</I>>, Vol(Issue): Pages
  - 100% bidirectional verification: every cited key is listed, every listed entry is cited!
"""
import re

CITED_KEYS = set()

# Each entry: key -> dict(lang='en'|'zh', sort_key, author_cite, year, full_entry)
REFS = {
    # ---------------- ENGLISH REFERENCES (Verified from Animals 2026 & Agriculture 2026) ----------------
    "arulmozhi2024": {
        "lang": "en",
        "sort_key": "arulmozhi 2024",
        "author_cite": "Arulmozhi et al.",
        "year": "2024",
        "full_entry": "Arulmozhi E, Deb N C, Tamrakar N, Kang D Y, Kang M Y, Kook J, Basak J K, Kim H T. 2024. From reality to virtuality: Revolutionizing livestock farming through digital twins. <<I>>Agriculture<</I>>, 14: 2231",
    },
    "astill2020": {
        "lang": "en",
        "sort_key": "astill 2020",
        "author_cite": "Astill et al.",
        "year": "2020",
        "full_entry": "Astill J, Dara R A, Fraser E D G, Roberts B, Sharif S. 2020. Smart poultry management: Smart sensors, big data, and the Internet of Things. <<I>>Computers and Electronics in Agriculture<</I>>, 170: 105291",
    },
    "balthazar2025": {
        "lang": "en",
        "sort_key": "balthazar 2025",
        "author_cite": "Balthazar et al.",
        "year": "2025",
        "full_entry": "Balthazar G D R, Silveira R M F, Soares T C, Da Silva I J O. 2025. RobôFrango: Proof of concept of a mobile biosensor robot for environmental monitoring in broiler houses. <<I>>Results in Engineering<</I>>, 28: 107796",
    },
    "berckmans2017": {
        "lang": "en",
        "sort_key": "berckmans 2017",
        "author_cite": "Berckmans",
        "year": "2017",
        "full_entry": "Berckmans D. 2017. General introduction to precision livestock farming. <<I>>Animal Frontiers<</I>>, 7: 6~11",
    },
    "brugaletta2022": {
        "lang": "en",
        "sort_key": "brugaletta 2022",
        "author_cite": "Brugaletta et al.",
        "year": "2022",
        "full_entry": "Brugaletta G, Teyssier J R, Rochell S J, Dridi S, Sirri F. 2022. A review of heat stress in chickens. Part I: Insights into physiology and gut health. <<I>>Frontiers in Physiology<</I>>, 13: 934381",
    },
    "cartoni2023": {
        "lang": "en",
        "sort_key": "cartoni mancinelli 2023",
        "author_cite": "Cartoni Mancinelli et al.",
        "year": "2023",
        "full_entry": "Cartoni Mancinelli A, Baldi G, Soglia F, Mattioli S, Sirri F, Petracci M, Castellini C, Zampiga M. 2023. Impact of chronic heat stress on behavior, oxidative status and meat quality traits of fast-growing broiler chickens. <<I>>Frontiers in Physiology<</I>>, 14: 1242094",
    },
    "chen2023": {
        "lang": "en",
        "sort_key": "chen 2023",
        "author_cite": "Chen et al.",
        "year": "2023",
        "full_entry": "Chen X, Yang L, Xue H, Li L, Yu Y. 2023. A machine learning model based on GRU and LSTM to predict the environmental parameters in a layer house, taking CO2 concentration as an example. <<I>>Sensors<</I>>, 24: 244",
    },
    "cho2014": {
        "lang": "en",
        "sort_key": "cho 2014",
        "author_cite": "Cho et al.",
        "year": "2014",
        "full_entry": "Cho K, van Merriënboer B, Gulcehre C, Bahdanau D, Bougares F, Schwenk H, Bengio Y. 2014. Learning phrase representations using RNN encoder-decoder for statistical machine translation. In: Moschitti A, Pang B, Daelemans W. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP). Doha: Association for Computational Linguistics: 1724~1734",
    },
    "chung2014": {
        "lang": "en",
        "sort_key": "chung 2014",
        "author_cite": "Chung et al.",
        "year": "2014",
        "full_entry": "Chung J, Gulcehre C, Cho K, Bengio Y. 2014. Empirical evaluation of gated recurrent neural networks on sequence modeling. <<I>>arXiv<</I>>, arXiv:1412.3555",
    },
    "dasilva2025": {
        "lang": "en",
        "sort_key": "da silva 2025",
        "author_cite": "Da Silva et al.",
        "year": "2025",
        "full_entry": "Da Silva N C, de Nääs I A, Barros J d S G, Moura D J de. 2025. Modeling broiler discomfort under commercial housing: Seasonal trends and predictive insights for precision livestock farming. <<I>>Poultry<</I>>, 4: 38",
    },
    "deng2024": {
        "lang": "en",
        "sort_key": "deng 2024",
        "author_cite": "Deng et al.",
        "year": "2024",
        "full_entry": "Deng S, Li Z, Wei Y, Wang Y, Li B, Zheng W. 2024. Assessing temperature distribution inside commercial stacked cage broiler houses in winter. <<I>>Animals<</I>>, 14: 2638",
    },
    "guo2021": {
        "lang": "en",
        "sort_key": "guo 2021",
        "author_cite": "Guo et al.",
        "year": "2021",
        "full_entry": "Guo Y, Aggrey S E, Oladeinde A, Johnson J, Zock G, Chai L. 2021. A machine vision-based method optimized for restoring broiler chicken images occluded by feeding and drinking equipment. <<I>>Animals<</I>>, 11: 123",
    },
    "hao2022": {
        "lang": "en",
        "sort_key": "hao 2022",
        "author_cite": "Hao et al.",
        "year": "2022",
        "full_entry": "Hao H, Fang P, Duan E, Yang Z, Wang L, Wang H. 2022. A dead broiler inspection system for large-scale breeding farms based on deep learning. <<I>>Agriculture<</I>>, 12: 1176",
    },
    "hinkle2003": {
        "lang": "en",
        "sort_key": "hinkle 2003",
        "author_cite": "Hinkle et al.",
        "year": "2003",
        "full_entry": "Hinkle D E, Wiersma W, Jurs S G. 2003. Applied Statistics for the Behavioral Sciences. 5th ed. Boston: Houghton Mifflin",
    },
    "ji2025": {
        "lang": "en",
        "sort_key": "ji 2025",
        "author_cite": "Ji and Teng",
        "year": "2025",
        "full_entry": "Ji H, Teng G. 2025. Multistep prediction of temperature and humidity in poultry houses based on the GFF-Transformer model. <<I>>Engineering in Agriculture<</I>>, 12: 803",
    },
    "kim2025": {
        "lang": "en",
        "sort_key": "kim 2025",
        "author_cite": "Kim et al.",
        "year": "2025",
        "full_entry": "Kim H R, Seong P, Seol K H, Park J E, Kim H, Park W, Cho J H, Lee S D. 2025. Effects of heat stress on growth performance, physiological responses, and carcass traits in broilers. <<I>>Journal of Thermal Biology<</I>>, 127: 103994",
    },
    "kio2024": {
        "lang": "en",
        "sort_key": "kio 2024",
        "author_cite": "Kio et al.",
        "year": "2024",
        "full_entry": "Kio A E, Xu J, Gautam N, Ding Y. 2024. Wavelet decomposition and neural networks: A potent combination for short term wind speed and power forecasting. <<I>>Frontiers in Energy Research<</I>>, 12: 1277464",
    },
    "kucuktopcu2024": {
        "lang": "en",
        "sort_key": "kucuktopcu 2024",
        "author_cite": "Küçüktopçu et al.",
        "year": "2024",
        "full_entry": "Küçüktopçu E, Cemek B, Simsek H. 2024. Modeling environmental conditions in poultry production: Computational fluid dynamics approach. <<I>>Animals<</I>>, 14: 501",
    },
    "lara2013": {
        "lang": "en",
        "sort_key": "lara 2013",
        "author_cite": "Lara and Rostagno",
        "year": "2013",
        "full_entry": "Lara L J, Rostagno M H. 2013. Impact of heat stress on poultry production. <<I>>Animals<</I>>, 3: 356~369",
    },
    "lee2022": {
        "lang": "en",
        "sort_key": "lee 2022",
        "author_cite": "Lee et al.",
        "year": "2022",
        "full_entry": "Lee S, Lee I, Yeo U, Kim J, Kim R. 2022. Machine learning approach to predict air temperature and relative humidity inside mechanically and naturally ventilated duck houses: Application of recurrent neural network. <<I>>Agriculture<</I>>, 12: 318",
    },
    "liang2023": {
        "lang": "en",
        "sort_key": "liang 2023",
        "author_cite": "Liang et al.",
        "year": "2023",
        "full_entry": "Liang H, Wu J, Zhang H, Yang J. 2023. Two-stage short-term power load forecasting based on RFECV feature selection algorithm and a TCN-ECA-LSTM neural network. <<I>>Energies<</I>>, 16: 1925",
    },
    "luo2026a": {
        "lang": "en",
        "sort_key": "luo 2026a",
        "author_cite": "Luo et al.",
        "year": "2026a",
        "full_entry": "Luo S, Kong X, Xie X, Zhang W, Zheng P, Zhu H, Hao D, Sun J, Chen C. 2026a. Monitoring and early warning of the cage-rearing broiler farming environment based on an inspection robot. <<I>>Animals<</I>>, 16(9): 1417",
    },
    "luo2026b": {
        "lang": "en",
        "sort_key": "luo 2026b",
        "author_cite": "Luo et al.",
        "year": "2026b",
        "full_entry": "Luo S, Zhang W, Hao D, Zhu H, Sun J, Sun J, Chen C. 2026b. Development and application of an inspection-robot-based digital twin platform for cage-reared broilers. <<I>>Agriculture<</I>>, 16(17): 1845",
    },
    "ma2024": {
        "lang": "en",
        "sort_key": "ma w 2024",
        "author_cite": "Ma et al.",
        "year": "2024",
        "full_entry": "Ma W, Wang X, Yang S X, Xue X, Li M, Wang R, Yu L, Song L, Li Q. 2024. Autonomous inspection robot for dead laying hens in caged layer house. <<I>>Computers and Electronics in Agriculture<</I>>, 227: 109595",
    },
    "ma2025": {
        "lang": "en",
        "sort_key": "ma y 2025",
        "author_cite": "Ma et al.",
        "year": "2025",
        "full_entry": "Ma Y, Guo Y, Gao B, Zheng P, Chen C. 2025. Development of a tool to detect open-mouthed respiration in caged broilers. <<I>>Animals<</I>>, 15: 2732",
    },
    "martinez2021": {
        "lang": "en",
        "sort_key": "martinez 2021",
        "author_cite": "Martinez et al.",
        "year": "2021",
        "full_entry": "Martinez A A G, Nääs I D A, De Carvalho-Curi T M R, Abe J M, Da Silva Lima N D. 2021. A heuristic and data mining model for predicting broiler house environment suitability. <<I>>Animals<</I>>, 11: 2780",
    },
    "neethirajan2021": {
        "lang": "en",
        "sort_key": "neethirajan 2021",
        "author_cite": "Neethirajan and Kemp",
        "year": "2021",
        "full_entry": "Neethirajan S, Kemp B. 2021. Digital twins in livestock farming. <<I>>Animals<</I>>, 11: 1008",
    },
    "neethirajan2026": {
        "lang": "en",
        "sort_key": "neethirajan 2026",
        "author_cite": "Neethirajan",
        "year": "2026",
        "full_entry": "Neethirajan S. 2026. Digital twins for cows and chickens: From hype cycles to hard evidence in precision livestock farming. <<I>>Agriculture<</I>>, 16: 166",
    },
    "norton2019": {
        "lang": "en",
        "sort_key": "norton 2019",
        "author_cite": "Norton et al.",
        "year": "2019",
        "full_entry": "Norton T, Chen C, Larsen M L V, Berckmans D. 2019. Review: Precision livestock farming: Building \"digital representations\" to bring the animals closer to the farmer. <<I>>Animal<</I>>, 13: 3009~3017",
    },
    "ozenturk2024": {
        "lang": "en",
        "sort_key": "ozenturk 2024",
        "author_cite": "Özentürk et al.",
        "year": "2024",
        "full_entry": "Özentürk U, Chen Z, Jamone L, Versace E. 2024. Robotics for poultry farming: Challenges and opportunities. <<I>>Computers and Electronics in Agriculture<</I>>, 226: 109411",
    },
    "park2022": {
        "lang": "en",
        "sort_key": "park 2022",
        "author_cite": "Park et al.",
        "year": "2022",
        "full_entry": "Park M, Britton D, Daley W, McMurray G, Navaei M, Samoylov A, Usher C, Xu J. 2022. Artificial intelligence, sensors, robots, and transportation systems drive an innovative future for poultry broiler and breeder management. <<I>>Animal Frontiers<</I>>, 12: 40~48",
    },
    "purcell2023": {
        "lang": "en",
        "sort_key": "purcell 2023",
        "author_cite": "Purcell and Neubauer",
        "year": "2023",
        "full_entry": "Purcell W, Neubauer T. 2023. Digital twins in agriculture: A state-of-the-art review. <<I>>Smart Agricultural Technology<</I>>, 3: 100094",
    },
    "pylianidis2021": {
        "lang": "en",
        "sort_key": "pylianidis 2021",
        "author_cite": "Pylianidis et al.",
        "year": "2021",
        "full_entry": "Pylianidis C, Osinga S, Athanasiadis I N. 2021. Introducing digital twins to agriculture. <<I>>Computers and Electronics in Agriculture<</I>>, 184: 105942",
    },
    "qi2023": {
        "lang": "en",
        "sort_key": "qi 2023",
        "author_cite": "Qi et al.",
        "year": "2023",
        "full_entry": "Qi F, Zhao X, Shi Z, Li H, Zhao W. 2023. Environmental factor detection and analysis technologies in livestock and poultry houses: A review. <<I>>Agriculture<</I>>, 13: 1489",
    },
    "tagarakis2024": {
        "lang": "en",
        "sort_key": "tagarakis 2024",
        "author_cite": "Tagarakis et al.",
        "year": "2024",
        "full_entry": "Tagarakis A C, Benos L, Kyriakarakos G, Pearson S, Sorensen C G, Bochtis D. 2024. Digital twins in agriculture and forestry: A review. <<I>>Sensors<</I>>, 24: 3117",
    },
    "vasconcelos2024": {
        "lang": "en",
        "sort_key": "vasconcelos 2024",
        "author_cite": "Vasconcelos et al.",
        "year": "2024",
        "full_entry": "Vasconcelos M D C, Sousa L S, Lopes T S B, Gonçalves L M, De Souza A B, Avelar N M, Oliveira J M F, Leme F D O P, Lara L J C, Araújo I C S. 2024. Impact of increased pre-start diet density on broiler chick behavior, corticosterone levels, and performance responses under cold stress during early life. <<I>>Journal of Thermal Biology<</I>>, 124: 103974",
    },
    "verdouw2021": {
        "lang": "en",
        "sort_key": "verdouw 2021",
        "author_cite": "Verdouw et al.",
        "year": "2021",
        "full_entry": "Verdouw C, Tekinerdogan B, Beulens A, Wolfert S. 2021. Digital twins in smart farming. <<I>>Agricultural Systems<</I>>, 189: 103046",
    },
    "vroegindeweij2018": {
        "lang": "en",
        "sort_key": "vroegindeweij 2018",
        "author_cite": "Vroegindeweij et al.",
        "year": "2018",
        "full_entry": "Vroegindeweij B A, Blaauw S K, IJsselmuiden J M M, Van Henten E J. 2018. Evaluation of the performance of PoultryBot, an autonomous mobile robotic platform for poultry houses. <<I>>Biosystems Engineering<</I>>, 174: 295~315",
    },
    "wang2020": {
        "lang": "en",
        "sort_key": "wang 2020",
        "author_cite": "Wang et al.",
        "year": "2020",
        "full_entry": "Wang Q, Wu B, Zhu P, Li P, Zuo W, Hu Q. 2020. ECA-Net: Efficient channel attention for deep convolutional neural networks. In: Proceedings of the 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Seattle: IEEE: 11531~11539",
    },
    "xu2020": {
        "lang": "en",
        "sort_key": "xu z 2020",
        "author_cite": "Xu et al.",
        "year": "2020",
        "full_entry": "Xu Z, Zou X, Yin Z, Zhang S, Song Y, Zhang J, Lu J. 2020. Prediction model of ammonia concentration in yellow-feather broilers house during winter based on EEMD-GRU. <<I>>INMATEH Agricultural Engineering<</I>>, 61: 59~70",
    },
    "xu2025": {
        "lang": "en",
        "sort_key": "xu d 2025",
        "author_cite": "Xu et al.",
        "year": "2025",
        "full_entry": "Xu D, Zeng Q, Wang W, Zhang X, Wang Y, Zang H. 2025. A deep learning runoff prediction model based on wavelet decomposition and dynamic feature fusion. <<I>>Scientific Reports<</I>>, 15: 37235",
    },
    "yang2023": {
        "lang": "en",
        "sort_key": "yang 2023",
        "author_cite": "Yang et al.",
        "year": "2023",
        "full_entry": "Yang J, Zhang T, Fang C, Zheng H. 2023. A defencing algorithm based on deep learning improves the detection accuracy of caged chickens. <<I>>Computers and Electronics in Agriculture<</I>>, 204: 107501",
    },
    "you2020": {
        "lang": "en",
        "sort_key": "you 2020",
        "author_cite": "You et al.",
        "year": "2020",
        "full_entry": "You G, Dai J, Tian Y, Wang Y, Zhu Q, Li D, Zeng Z, Zhao X. 2020. Study on the suitable environmental temperature and humidity for Hy-Line Brown layer breeders in summer. <<I>>Journal of Yunnan Agricultural University<</I>>, 35: 990~995",
    },
    "zheng2025": {
        "lang": "en",
        "sort_key": "zheng 2025",
        "author_cite": "Zheng et al.",
        "year": "2025",
        "full_entry": "Zheng P, Zhang W, Gao B, Ma Y, Chen C. 2025. Multi-step apparent temperature prediction in broiler houses using a hybrid SE-TCN-Transformer model with Kalman filtering. <<I>>Sensors<</I>>, 25: 6124",
    },
    "zhu2026": {
        "lang": "en",
        "sort_key": "zhu 2026",
        "author_cite": "Zhu et al.",
        "year": "2026",
        "full_entry": "Zhu H, Zhao G, Wang H, Luo S, Sun J, Hao D, Chen C. 2026. Design and application of a cloud platform for broiler inspection robots. <<I>>Agriculture<</I>>, 16: 1201",
    },

    # ---------------- CHINESE STANDARDS (Cited in Animals 2026 Sec. 2.4 & 2.5) ----------------
    "db1304_2021": {
        "lang": "zh",
        "sort_key": "handanshi 2021",
        "author_cite": "邯郸市市场监督管理局",
        "year": "2021",
        "full_entry": "邯郸市市场监督管理局. 2021. DB1304/T 373-2021 肉鸡立体笼养技术规程. 邯郸：邯郸市市场监督管理局",
    },
    "nyt1871_2010": {
        "lang": "zh",
        "sort_key": "zhonghuarenmingongheguonongyebu 2010",
        "author_cite": "中华人民共和国农业部",
        "year": "2010",
        "full_entry": "中华人民共和国农业部. 2010. NY/T 1871-2010 黄羽肉鸡饲养管理技术规程. 北京：中国农业出版社",
    },
    "gbt19664_2005": {
        "lang": "zh",
        "sort_key": "zhonghuarenmingongheguoguojiazhiliang 2005",
        "author_cite": "中华人民共和国国家质量监督检验检疫总局和中国国家标准化管理委员会",
        "year": "2005",
        "full_entry": "中华人民共和国国家质量监督检验检疫总局, 中国国家标准化管理委员会. 2005. GB/T 19664-2005 商品肉鸡生产技术规程. 北京：中国标准出版社",
    },
}

# Mapping from old draft numeric IDs [1]..[28] to verified keys
OLD_NUM_TO_KEY = {
    1: "lara2013",
    2: "brugaletta2022",
    3: "kim2025",
    4: "astill2020",
    5: "qi2023",
    6: "lee2022",
    7: "chen2023",
    8: "xu2020",
    9: "balthazar2025",
    10: "park2022",
    11: "vroegindeweij2018",
    12: "ozenturk2024",
    13: "berckmans2017",
    14: "norton2019",
    15: "verdouw2021",
    16: "pylianidis2021",
    17: "neethirajan2021",
    18: "purcell2023",
    19: "tagarakis2024",
    20: "arulmozhi2024",
    21: "luo2026a",
    22: "luo2026b",
    23: "xu2025",
    24: "kio2024",
    25: "liang2023",
    26: "wang2020",
    27: "cho2014",
    28: "chung2014",
}


def cite(*keys):
    """Parenthetical Author-Year citation per TJAU Attachment 5, sorted alphabetically, '; ' separated."""
    for k in keys:
        assert k in REFS, f"Unknown reference key: {k}"
        CITED_KEYS.add(k)
    # Group same author_cite (e.g., Luo et al. 2026a, 2026b)
    sorted_keys = sorted(keys, key=lambda k: (0 if REFS[k]["lang"] == "en" else 1, REFS[k]["sort_key"]))
    grouped = []
    for k in sorted_keys:
        ac = REFS[k]["author_cite"]
        yr = REFS[k]["year"]
        if grouped and grouped[-1][0] == ac:
            grouped[-1][1].append(yr)
        else:
            grouped.append((ac, [yr]))
    parts = [f"{ac} {', '.join(yrs)}" for ac, yrs in grouped]
    return "（" + "; ".join(parts) + "）"


def cite_year(key):
    """Inline year citation when author is already named in prose, e.g. Astill等（2020）."""
    assert key in REFS, f"Unknown reference key: {key}"
    CITED_KEYS.add(key)
    return f"（{REFS[key]['year']}）"


def expand_bracket_nums(spec_str):
    """Expand '1-3,15' -> [1, 2, 3, 15]."""
    nums = []
    for part in re.split(r"[,，]", spec_str):
        part = part.strip()
        m = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", part)
        if m:
            nums.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        elif part.isdigit():
            nums.append(int(part))
    return nums


def convert_bracket_citations_in_text(text, p_idx=None):
    """
    Convert all [n] / [a-b] / [a,b] citations in a paragraph to TJAU Author-Year citations,
    fixing known citation mismatches from the original draft (P24, P38, P39, P43)
    and preserving mathematical closed intervals like [0,1] / [-1,1].
    """
    # Normalize mathematical intervals first so they are never mistaken for citations
    text = text.replace("[0,1]", "[0, 1]").replace("[-1,1]", "[-1, 1]").replace("[−1,1]", "[−1, 1]")

    # Fix 4 specific known citation mismatches in original draft Chapter 1
    if p_idx == 24:
        text = text.replace("[8-13]", cite("verdouw2021", "pylianidis2021", "neethirajan2021", "purcell2023", "tagarakis2024", "arulmozhi2024", "neethirajan2026"))
    elif p_idx == 38:
        text = text.replace("[14]", cite("ji2025", "zheng2025"))
    elif p_idx == 39:
        text = text.replace("[1-3,15]", cite("lara2013", "brugaletta2022", "kim2025", "martinez2021", "dasilva2025"))
    elif p_idx == 43:
        text = text.replace("Vroegindeweij等对PoultryBot自主移动机器人平台在家禽舍中的运行性能进行了评估[11]", f"Vroegindeweij等{cite_year('vroegindeweij2018')}对PoultryBot自主移动机器人平台在家禽舍中的运行性能进行了评估")
        text = text.replace("Ma等研究了笼养蛋鸡死鸡自主巡检机器人[16]", f"Ma等{cite_year('ma2024')}研究了笼养蛋鸡死鸡自主巡检机器人，Hao等{cite_year('hao2022')}构建了规模化养殖场死鸡深度学习巡检系统，Ma等{cite_year('ma2025')}进一步研发了笼养肉鸡张嘴呼吸检测工具")
        text = text.replace("Özentürk等对家禽养殖机器人面临的技术挑战与发展机会进行了综述[12]", f"Özentürk等{cite_year('ozenturk2024')}对家禽养殖机器人面临的技术挑战与发展机会进行了综述")
    elif p_idx == 32:
        text = text.replace("Astill等对智慧家禽管理中的智能传感器、大数据和物联网技术进行了综述，指出多源数据采集和信息处理是提升家禽生产管理水平的重要基础[4]",
                            f"Astill等{cite_year('astill2020')}对智慧家禽管理中的智能传感器、大数据和物联网技术进行了综述，指出多源数据采集和信息处理是提升家禽生产管理水平的重要基础")
        text = text.replace("Qi等对畜禽舍环境因子检测与分析技术进行了总结，涵盖环境参数检测设备及相关监测方法，为畜禽舍环境监测系统设计提供了参考[5]",
                            f"Qi等{cite_year('qi2023')}对畜禽舍环境因子检测与分析技术进行了总结，涵盖环境参数检测设备及相关监测方法，为畜禽舍环境监测系统设计提供了参考")
    elif p_idx == 34:
        text = text.replace("Balthazar等提出了用于肉鸡舍环境监测的移动生物传感机器人概念验证，展示了移动机器人在鸡舍环境信息采集方面的应用潜力[9]",
                            f"Balthazar等{cite_year('balthazar2025')}提出了用于肉鸡舍环境监测的移动生物传感机器人概念验证，展示了移动机器人在鸡舍环境信息采集方面的应用潜力")
        text = text.replace("Park等也指出，人工智能、传感器、机器人和运输系统的发展将推动肉鸡及种鸡管理方式创新[10]",
                            f"Park等{cite_year('park2022')}也指出，人工智能、传感器、机器人和运输系统的发展将推动肉鸡及种鸡管理方式创新")
    elif p_idx == 37:
        text = text.replace("Lee等将循环神经网络应用于鸭舍温度和相对湿度预测[6]", f"Lee等{cite_year('lee2022')}将循环神经网络应用于鸭舍温度和相对湿度预测")
        text = text.replace("Chen等利用GRU和LSTM模型开展蛋鸡舍二氧化碳浓度预测研究[7]", f"Chen等{cite_year('chen2023')}利用GRU和LSTM模型开展蛋鸡舍二氧化碳浓度预测研究")
        text = text.replace("Xu等研究了基于EEMD-GRU的肉鸡舍氨气浓度预测方法[8]", f"Xu等{cite_year('xu2020')}研究了基于EEMD-GRU的肉鸡舍氨气浓度预测方法")
    elif p_idx == 46:
        text = text.replace("Berckmans从精准畜牧业角度阐述了利用数据和技术实现动物生产精细化管理的基本思路[13]", f"Berckmans{cite_year('berckmans2017')}从精准畜牧业角度阐述了利用数据和技术实现动物生产精细化管理的基本思路")
        text = text.replace("Norton等提出通过构建动物及养殖过程的数字化表示，使生产管理人员能够更加全面地掌握动物状态和生产信息[14]", f"Norton等{cite_year('norton2019')}提出通过构建动物及养殖过程的数字化表示，使生产管理人员能够更加全面地掌握动物状态和生产信息")
    elif p_idx == 47:
        text = text.replace("Verdouw等对智慧农业中的数字孪生进行了研究，讨论了数字孪生在农业生产过程中的应用方式[15]", f"Verdouw等{cite_year('verdouw2021')}对智慧农业中的数字孪生进行了研究，讨论了数字孪生在农业生产过程中的应用方式")
        text = text.replace("Pylianidis等分析了数字孪生技术引入农业的可能性及其关键问题[16]", f"Pylianidis等{cite_year('pylianidis2021')}分析了数字孪生技术引入农业的可能性及其关键问题")
        text = text.replace("Neethirajan和Kemp围绕畜牧业数字孪生展开讨论，指出数字孪生有望将动物、环境和生产管理信息联系起来[17]", f"Neethirajan和Kemp{cite_year('neethirajan2021')}围绕畜牧业数字孪生展开讨论，指出数字孪生有望将动物、环境和生产管理信息联系起来")
        text = text.replace("Purcell和Neubauer、Tagarakis等分别从农业数字孪生的技术发展和应用进展角度进行了综述[18-19]",
                            f"Purcell和Neubauer{cite_year('purcell2023')}、Tagarakis等{cite_year('tagarakis2024')}以及Neethirajan{cite_year('neethirajan2026')}分别从农业及畜牧业数字孪生的技术演进与实证进展角度进行了综述")
    elif p_idx in (641, 646):
        # Remove dangling trailing [21]/[22] after period in Chapter 6 conclusions
        text = re.sub(r"\s*\[(?:21|22)\]\s*$", "", text)

    # Generic fallback for all remaining [n] / [a-b] / [a,b] citations
    def _repl(m):
        inner = m.group(1)
        # Skip mathematical intervals like [0, 1] or [-1, 1]
        if inner.startswith("0,") or inner.startswith("-1") or inner.startswith("−1"):
            return m.group(0)
        nums = expand_bracket_nums(inner)
        if not nums:
            return m.group(0)
        keys = [OLD_NUM_TO_KEY[n] for n in nums if n in OLD_NUM_TO_KEY]
        if not keys:
            return m.group(0)
        return cite(*keys)

    text = re.sub(r"\[(\d+(?:[-–,]\d+)*)\]", _repl, text)
    return text


def get_sorted_bibliography():
    """Return cited references strictly ordered: English A-Z first, then Chinese Pinyin."""
    en_keys = sorted(
        [k for k in CITED_KEYS if REFS[k]["lang"] == "en"],
        key=lambda k: REFS[k]["sort_key"],
    )
    zh_keys = sorted(
        [k for k in CITED_KEYS if REFS[k]["lang"] == "zh"],
        key=lambda k: REFS[k]["sort_key"],
    )
    return [REFS[k]["full_entry"] for k in en_keys + zh_keys]
