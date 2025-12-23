"""
Unified feature definitions and extraction for HTML element analysis.
Refactored for modularity, readability, and robustness.
"""

# @generated "partially" Gemini 3: generete regex patterns for price and currency detection
# @generated "partially" Gemini 3: generated docs strings and imporve comments

# inspired by:
# https://stackoverflow.com/questions/13336576/extracting-an-information-from-web-page-by-machine-learning

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import lxml.html
import pandas as pd
import regex as re
import requests
from PIL import ImageFile

from product_scraper.utils.console import log_error, log_warning
from product_scraper.utils.utils import get_unique_xpath, normalize_tag

# --- Configuration ---
TIMEOUT_SECONDS = 3
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ProductScraper/1.0)"}

# Constants
DEFAULT_DIST = -1
UNWANTED_TAGS = {
    "script",
    "style",
    "noscript",
    "form",
    "iframe",
    "meta",
    "link",
    "head",
}
OTHER_CATEGORY = "other"

# --- Visual Heuristics Constants ---
TAG_DEFAULT_SIZES = {
    "h1": 32.0,
    "h2": 24.0,
    "h3": 20.0,
    "h4": 18.0,
    "h5": 16.0,
    "h6": 14.0,
    "p": 16.0,
    "div": 16.0,
    "span": 16.0,
    "li": 16.0,
    "a": 16.0,
    "small": 12.0,
    "strong": 16.0,
    "b": 16.0,
    "button": 14.0,
    "nav": 0.0,
    "footer": 0.0,
    "header": 0.0,
}

# Hierarchy Scoring
TAG_IMPORTANCE = {
    "h1": 10,
    "h2": 9,
    "h3": 8,
    "h4": 7,
    "h5": 6,
    "strong": 5,
    "b": 5,
    "em": 4,
    "a": 2,
    "button": 2,
    "li": 2,
    "p": 1,
    "div": 1,
    "span": 1,
    "nav": -5,
    "footer": -5,
    "header": -2,
}

# @generated "all" Gemini 3: generete regex patterns for price and currency detection

# fmt: off
# --- Currency & Keyword Configuration ---
ISO_CURRENCIES = [
    "AED", "AFN", "ALL", "AMD", "ANG", "AOA", "ARS", "AUD", "AWG", "AZN",
    "BAM", "BBD", "BDT", "BGN", "BHD", "BIF", "BMD", "BND", "BOB", "BRL",
    "BSD", "BTN", "BWP", "BYN", "BZD", "CAD", "CDF", "CHF", "CLP", "CNY",
    "COP", "CRC", "CUP", "CVE", "CZK", "DJF", "DKK", "DOP", "DZD", "EGP",
    "ERN", "ETB", "EUR", "FJD", "FKP", "GBP", "GEL", "GHS", "GIP", "GMD",
    "GNF", "GTQ", "GYD", "HKD", "HNL", "HRK", "HTG", "HUF", "IDR", "ILS",
    "INR", "IQD", "IRR", "ISK", "JMD", "JOD", "JPY", "KES", "KGS", "KHR",
    "KMF", "KPW", "KRW", "KWD", "KYD", "KZT", "LAK", "LBP", "LKR", "LRD",
    "LSL", "LYD", "MAD", "MDL", "MGA", "MKD", "MMK", "MNT", "MOP", "MRU",
    "MUR", "MVR", "MWK", "MXN", "MYR", "MZN", "NAD", "NGN", "NIO", "NOK",
    "NPR", "NZD", "OMR", "PAB", "PEN", "PGK", "PHP", "PKR", "PLN", "PYG",
    "QAR", "RON", "RSD", "RUB", "RWF", "SAR", "SBD", "SCR", "SDG", "SEK",
    "SGD", "SHP", "SLE", "SLL", "SOS", "SRD", "SSP", "STN", "SVC", "SYP",
    "SZL", "THB", "TJS", "TMT", "TND", "TOP", "TRY", "TTD", "TWD", "TZS",
    "UAH", "UGX", "USD", "UYU", "UZS", "VES", "VND", "VUV", "WST", "XAF",
    "XCD", "XOF", "XPF", "YER", "ZAR", "ZMW", "ZWL",
]

SOLD_WORD_VARIATIONS = [
    "sold", "sold out", "out of stock", "unavailable", "discontinued",
    "vendu", "épuisé", "indisponible", " rupture de stock",
    "verkauft", "ausverkauft", "nicht vorrätig", "nicht lieferbar",
    "vendido", "agotado", "no disponible", "fuera de stock",
    "venduto", "esaurito", "non disponibile",
    "vendido", "esgotado", "indisponível",
    "verkocht", "niet op voorraad", "uitverkocht",
    "såld", "slutsåld", "slut i lager", "ej i lager",
    "solgt", "udsolgt", "ikke på lager",
    "myyty", "loppu", "ei varastossa",
    "продано", "нет в наличии", "раскуплено", "закончился",
    "sprzedane", "brak w magazynie", "wyprzedane", "niedostępny",
    "prodáno", "vyprodáno", "není skladem",
    "eladva", "elfogyott", "nincs raktáron",
    "vândut", "stoc epuizat", "indisponibil",
    "prodano", "rasprodano", "nema na zalihi",
    "售出", "已售出", "缺货", "暂时缺货", "售罄",
    "売り切れ", "在庫切れ", "完売", "品切れ",
    "품절", "매진", "재고 없음", "판매 완료",
    "đã bán", "hết hàng", "bán hết",
    "ขายแล้ว", "สินค้าหมด", "หมด",
    "terjual", "habis", "stok habis", "kosong",
    "مباع", "نفذ", "نفذت الكمية", "غير متوفر",
    "נמכר", "אזל במלאי", "לא זמין",
    "satıldı", "tükendi", "stokta yok", "temin edilemiyor",
    "बिका हुआ", "स्टॉक में नहीं", "उपलब्ध नहीं",
    "ناموجود", "فروخته شد",
    "εξαντλήθηκε", "μη διαθέσιμο", "κατόπιν παραγγελίας",
    "uppselt", "ekki til",
    "išparduota", "nėra prekyboje",
    "izpārdots", "nav pieejams",
]

REVIEW_KEYWORDS = [
    "review", "reviews", "rating", "ratings", "stars", "feedback",
    "testimonial", "testimonials", "comment", "comments", "opinion",
]

CTA_KEYWORDS = [
    "add to cart", "add to bag", "buy", "buy now",
    "checkout", "purchase", "order",
]
# fmt: on


CUSTOM_SYMBOLS = [
    "Chf",
    "Kč",
    "kr",
    "zł",
    "Rs",
    "Ft",
    "lei",
    "kn",
    "din",
    "руб",
    "₹",
    r"R\$",
    "R",
    r",-",
    r"\.-",
    "€",
    "$",
    "£",
    "¥",
]


text_based_currencies = sorted(
    ISO_CURRENCIES + [s for s in CUSTOM_SYMBOLS if s.isalpha()], key=len, reverse=True
)
symbol_based_currencies = [s for s in CUSTOM_SYMBOLS if not s.isalpha()]

escaped_text = [re.escape(s) for s in text_based_currencies]
escaped_symbols = [re.escape(s) for s in symbol_based_currencies]
text_or = "|".join(escaped_text)
symbol_or = "|".join(escaped_symbols)

# Smart Pattern: Matches currency if it is a standalone word OR if it touches a digit
# Matches: "USD", "1500Kč", "100€", "USD100"
SMART_CURRENCY_PATTERN = rf"(?:\b(?:{text_or})\b|(?<=\d)(?:{text_or})|(?:{text_or})(?=\d)|(?:{symbol_or})|\p{{Sc}})"

CURRENCY_HINTS_REGEX = re.compile(SMART_CURRENCY_PATTERN, re.UNICODE | re.IGNORECASE)

# Number pattern allowing dash decimals like "120,-"
NUMBER_PATTERN = r"(?:\d{1,3}(?:[., ]\d{3})*|\d+)(?:[.,](?:\d{1,2}|-))?"

# Combined Price Regex
PRICE_REGEX = re.compile(
    rf"(?:(?:{SMART_CURRENCY_PATTERN})\s*{NUMBER_PATTERN}|{NUMBER_PATTERN}\s*(?:{SMART_CURRENCY_PATTERN}))",
    re.UNICODE | re.IGNORECASE,
)

SOLD_REGEX = re.compile(
    r"\b(?:" + "|".join(SOLD_WORD_VARIATIONS) + r")\b", re.IGNORECASE
)
REVIEW_REGEX = re.compile(r"\b(?:" + "|".join(REVIEW_KEYWORDS) + r")\b", re.IGNORECASE)
CTA_REGEX = re.compile(r"\b(?:" + "|".join(CTA_KEYWORDS) + r")\b", re.IGNORECASE)

# end of generated code

# --- Feature Definitions ---

NUMERIC_FEATURES = [
    "num_children",
    "num_siblings",
    "dom_depth",
    "sibling_tag_ratio",
    "sibling_link_ratio",
    "is_header",
    "is_block_element",
    "is_clickable",
    "is_formatting",
    "is_list_item",
    "is_bold",
    "is_italic",
    "is_hidden",
    "has_nav_ancestor",
    "has_footer_ancestor",
    "has_header_ancestor",
    "text_len",
    "text_word_count",
    "text_digit_count",
    "text_density",
    "digit_density",
    "link_density",
    "capitalization_ratio",
    "avg_word_length",
    "font_size",
    "font_weight",
    "visual_weight",
    "tag_importance",
    "visibility_score_local",
    "visibility_score_global",
    "text_len_score_local",
    "img_width",
    "img_height",
    "img_area_raw",
    "image_area",
    "sibling_image_count",
    "img_size_rank",
    "has_currency_symbol",
    "is_price_format",
    "has_sold_keyword",
    "has_review_keyword",
    "has_cta_keyword",
    "has_href",
    "is_image",
    "has_src",
    "has_alt",
    "alt_len",
    "parent_is_link",
    "is_strikethrough",
    "tag_count_global",
    "avg_distance_to_closest_categories",
    "max_sibling_density",
]

NON_TRAINING_FEATURES = ["Category", "SourceURL", "xpath"]
CATEGORICAL_FEATURES = ["tag", "parent_tag", "gparent_tag"]
TEXT_FEATURES = ["class_str", "id_str"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES
TARGET_FEATURE = "Category"

# --- Helper Functions ---


# https://stackoverflow.com/questions/15800704/get-image-size-without-loading-image-into-memory
@lru_cache(maxsize=256)
def get_remote_image_dims(url: str) -> Tuple[int, int]:
    """
    Fetch dimensions of a remote image without downloading the full content.

    Args:
        url (str): The URL of the image.

    Returns:
        Tuple[int, int]: Width and height of the image, or (0, 0) if failed.
    """
    if not url or url.startswith("data:"):
        return 0, 0
    try:
        # Strict timeout to prevent scraper hanging
        response = requests.get(url, headers=HEADERS, timeout=2.0, stream=True)
        if response.status_code != 200:
            return 0, 0

        p = ImageFile.Parser()
        count = 0
        # Only read the first few KB to get header info
        for chunk in response.iter_content(chunk_size=1024):
            if not chunk:
                break
            p.feed(chunk)
            if p.image:
                return p.image.size
            count += 1
            if count > 5:  # Limit max chunks read
                break
        return 0, 0
    except Exception:  # pylint: disable=broad-exception-caught
        return 0, 0


def get_xpath_segments(xpath: str) -> List[str]:
    """
    Split an XPath string into its segments.

    Args:
        xpath (str): The XPath string (e.g. '/html/body/div').

    Returns:
        List[str]: List of XPath segments (e.g. ['html', 'body', 'div']).
    """
    return [s for s in xpath.split("/") if s]


def extract_index(segment: str) -> int:
    """
    Extract the numeric index from an XPath segment.

    Args:
        segment (str): The XPath segment (e.g., 'div[2]').

    Returns:
        int: The extracted index, or 1 if no index is present.
    """
    match = re.search(r"\[(\d+)\]", segment)
    return int(match.group(1)) if match else 1


def calculate_proximity_score(xpath1: str, xpath2: str) -> Tuple[int, int]:
    """
    Calculates the structural proximity between two XPath strings.

    Args:
        xpath1 (str): The first XPath string (e.g., "/html/body/div[1]").
        xpath2 (str): The second XPath string (e.g., "/html/body/div[2]").

    Returns:
        Tuple[int, int]: A tuple containing:
            - tree_distance (int): Total steps to reach one element from the other via the common ancestor.
            - index_delta (int): The absolute difference in indices at the point where the paths diverge.
    """
    path1 = get_xpath_segments(xpath1)
    path2 = get_xpath_segments(xpath2)
    min_len = min(len(path1), len(path2))

    divergence_index = 0
    for i in range(min_len):
        # If segment has no index, assume [1]
        seg1 = path1[i] if "[" in path1[i] else f"{path1[i]}[1]"
        seg2 = path2[i] if "[" in path2[i] else f"{path2[i]}[1]"

        if seg1 == seg2:
            divergence_index += 1
        else:
            break

    dist_up = len(path1) - divergence_index
    dist_down = len(path2) - divergence_index
    tree_distance = dist_up + dist_down

    index_delta = 0
    # If they share the same parent but diverged at the child level
    if divergence_index < len(path1) and divergence_index < len(path2):
        # We also need to be careful extracting index from non-indexed segments
        idx1 = extract_index(path1[divergence_index])
        idx2 = extract_index(path2[divergence_index])
        index_delta = abs(idx1 - idx2)

    return (tree_distance, index_delta)


def get_avg_distance_to_closest_categories(
    element: lxml.html.HtmlElement,
    selectors: Dict[str, List[str]],
    current_category: str,
) -> float:
    """
    Calculate average distance from the element to the closest elements of other known categories.

    Args:
        element (lxml.html.HtmlElement): The target element.
        selectors (Dict[str, List[str]]): Dictionary of known selectors by category.
        current_category (str): The category of the target element (to exclude from distance calc).

    Returns:
        float: Average distance score, or DEFAULT_DIST if no other categories exist.
    """
    if not selectors:
        return DEFAULT_DIST

    elem_xpath = get_unique_xpath(element)
    min_distances = []

    for category, xpaths in selectors.items():
        if not xpaths or current_category == category:
            continue

        # Calculate distance to all items in this category
        # We only care about the tree distance [0] for the feature
        distances = [calculate_proximity_score(elem_xpath, xp)[0] for xp in xpaths]

        if distances:
            min_distances.append(min(distances))

    if not min_distances:
        return DEFAULT_DIST

    return sum(min_distances) / len(min_distances)


# --- Feature Extraction Helpers ---
def _check_ancestor_flags(
    element: lxml.html.HtmlElement, max_depth: int = 6
) -> Tuple[int, int, int]:
    """
    Traverse up the ancestors to check for nav, footer, or header presence.
    Returns (has_nav, has_footer, has_header).
    """
    has_nav, has_footer, has_header = 0, 0, 0
    cursor = element
    depth_check = 0

    while cursor is not None and depth_check < max_depth:
        ctag = normalize_tag(cursor.tag)
        c_class = str(cursor.get("class", "")).lower()

        if (
            ctag == "nav"
            or cursor.get("role") == "navigation"
            or "nav" in c_class
            or "menu" in c_class
        ):
            has_nav = 1

        if (
            ctag == "footer"
            or cursor.get("role") == "contentinfo"
            or "footer" in c_class
        ):
            has_footer = 1

        if ctag == "header" or cursor.get("role") == "banner":
            has_header = 1

        if has_nav and has_footer and has_header:
            break

        cursor = cursor.getparent()
        depth_check += 1

    return has_nav, has_footer, has_header


def _get_structure_features(
    element: lxml.html.HtmlElement, tag: str, category: str
) -> Dict[str, Any]:
    """
    Extract structural features like DOM depth, sibling counts, and ancestry.
    """

    def get_parent_tags(element):
        parent = element.getparent()
        gparent = parent.getparent() if parent is not None else None
        parent_tag = normalize_tag(parent.tag) if parent is not None else "root"
        gparent_tag = normalize_tag(gparent.tag) if gparent is not None else "root"
        return parent_tag, gparent_tag, parent

    def get_sibling_stats(parent, element, tag):
        stats = {
            "sibling_count": 0,
            "same_tag_count": 0,
            "sibling_image_count": 0,
            "sibling_link_count": 0,
        }
        if parent is not None:
            for child in parent:
                if child is element:
                    continue
                child_tag = normalize_tag(getattr(child, "tag", ""))
                if child_tag == tag:
                    stats["same_tag_count"] += 1
                if child_tag == "img":
                    stats["sibling_image_count"] += 1
                if child_tag == "a":
                    stats["sibling_link_count"] += 1
                stats["sibling_count"] += 1
        return stats

    parent_tag, gparent_tag, parent = get_parent_tags(element)
    stats = get_sibling_stats(parent, element, tag)
    has_nav, has_footer, has_header = _check_ancestor_flags(element)
    sibling_count = stats["sibling_count"]

    return {
        "Category": category,
        "tag": tag,
        "parent_tag": parent_tag,
        "gparent_tag": gparent_tag,
        "num_children": len(element),
        "num_siblings": sibling_count,
        "dom_depth": len(list(element.iterancestors())),
        "sibling_tag_ratio": (stats["same_tag_count"] / sibling_count)
        if sibling_count > 0
        else 0.0,
        "sibling_link_ratio": (stats["sibling_link_count"] / sibling_count)
        if sibling_count > 0
        else 0.0,
        "sibling_image_count": stats["sibling_image_count"],
        "has_nav_ancestor": has_nav,
        "has_footer_ancestor": has_footer,
        "has_header_ancestor": has_header,
    }


def _get_text_features(element: lxml.html.HtmlElement) -> Dict[str, Any]:
    """
    Extract text-related features like length, density, and capitalization.
    """
    raw_text = element.text_content() or ""
    text = " ".join(raw_text.split())
    text_len = len(text)
    try:
        html_size = len(lxml.html.tostring(element, encoding="unicode"))
        text_density = (text_len / html_size) if html_size > 0 else 0.0
    except Exception:  # pylint: disable=broad-exception-caught
        text_density = 0.0
    words = text.split()
    digit_count = sum(c.isdigit() for c in text)
    return {
        "text_len": text_len,
        "text_word_count": len(words),
        "text_digit_count": digit_count,
        "text_density": text_density,
        "digit_density": (digit_count / text_len) if text_len > 0 else 0.0,
        "capitalization_ratio": (sum(1 for c in text if c.isupper()) / text_len)
        if text_len > 0
        else 0.0,
        "avg_word_length": (sum(len(w) for w in words) / len(words)) if words else 0.0,
        "_text_content": text,
    }


def _get_regex_features(text: str) -> Dict[str, int]:
    """
    Check text against pre-compiled regex patterns (Currency, Price, CTA, etc.).
    """
    return {
        "has_currency_symbol": 1 if CURRENCY_HINTS_REGEX.search(text) else 0,
        "is_price_format": 1 if PRICE_REGEX.search(text) else 0,
        "has_sold_keyword": 1 if SOLD_REGEX.search(text) else 0,
        "has_review_keyword": 1 if REVIEW_REGEX.search(text) else 0,
        "has_cta_keyword": 1 if CTA_REGEX.search(text) else 0,
    }


def _parse_font_size(style: str, tag: str, parent_tag: str) -> float:
    """
    Helper to extract and normalize font size from style string or tag defaults.
    """
    font_size = 0.0
    fs_match = re.search(r"font-size\s*:\s*([\d.]+)(px|em|rem|pt|%)?", style)
    if fs_match:
        try:
            val = float(fs_match.group(1))
            unit = fs_match.group(2)
            if unit in ("em", "rem"):
                font_size = val * 16.0
            elif unit == "%":
                font_size = (val / 100.0) * 16.0
            elif unit == "pt":
                font_size = val * 1.33
            else:
                font_size = val
        except ValueError:
            pass

    if font_size == 0.0:
        if parent_tag.startswith("h"):
            font_size = TAG_DEFAULT_SIZES.get(parent_tag, 16.0)
        else:
            font_size = TAG_DEFAULT_SIZES.get(tag, 16.0)
    return font_size


def _parse_font_weight(style: str, tag: str, parent_tag: str) -> int:
    """
    Helper to extract and normalize font weight.
    """
    font_weight = 400
    fw_match = re.search(r"font-weight\s*:\s*(\w+)", style)
    if fw_match:
        w_str = fw_match.group(1)
        if w_str in {"bold", "bolder"}:
            font_weight = 700
        elif w_str == "lighter":
            font_weight = 300
        elif w_str.isdigit():
            font_weight = int(w_str)
    elif tag in {"h1", "h2", "h3", "b", "strong"} or parent_tag in {"h1", "h2", "h3"}:
        font_weight = 700
    return font_weight


def _get_visual_features(
    element: lxml.html.HtmlElement, tag: str, parent_tag: str
) -> Dict[str, Any]:
    """
    Extract visual cues based on CSS styles, tags, and hierarchy.
    """
    style = element.get("style", "").lower()
    is_header = 1 if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} else 0
    is_formatting = (
        1 if tag in {"b", "strong", "i", "em", "u", "span", "small", "mark"} else 0
    )
    is_block = (
        1
        if tag
        in {
            "div",
            "p",
            "section",
            "article",
            "main",
            "aside",
            "header",
            "footer",
            "ul",
            "ol",
            "table",
            "form",
        }
        else 0
    )
    is_list_item = 1 if tag in {"li", "dt", "dd"} else 0

    font_size = _parse_font_size(style, tag, parent_tag)
    font_weight = _parse_font_weight(style, tag, parent_tag)

    self_imp = TAG_IMPORTANCE.get(tag, 0)
    parent_imp = TAG_IMPORTANCE.get(parent_tag, 0)
    tag_importance = max(self_imp, parent_imp)

    return {
        "is_header": is_header,
        "is_formatting": is_formatting,
        "is_block_element": is_block,
        "is_list_item": is_list_item,
        "font_size": font_size,
        "font_weight": font_weight,
        "visual_weight": font_size * (font_weight / 400.0),
        "tag_importance": tag_importance,
        "is_bold": 1 if font_weight >= 600 else 0,
        "is_italic": 1 if (tag in {"i", "em"} or "font-style:italic" in style) else 0,
        "is_hidden": 1
        if ("display:none" in style or "visibility:hidden" in style)
        else 0,
        "is_strikethrough": 1
        if tag in {"s", "strike", "del"} or "line-through" in style
        else 0,
    }


def _get_image_features(element: lxml.html.HtmlElement, tag: str) -> Dict[str, Any]:
    """
    Extract image specific features like dimensions, area, and alt text.

    Args:
        element (lxml.html.HtmlElement): The target element.
        tag (str): Normalized tag name.

    Returns:
        Dict[str, Any]: Dictionary of image features.
    """
    features = {
        "is_image": 1 if tag == "img" else 0,
        "has_src": 0,
        "has_alt": 0,
        "alt_len": 0,
        "img_width": 0,
        "img_height": 0,
        "img_area_raw": 0,
        "image_area": 0,
    }
    if tag != "img":
        return features
    features["has_src"] = 1 if element.get("src") or element.get("data-src") else 0
    features["has_alt"] = 1 if element.get("alt") else 0
    features["alt_len"] = len(element.get("alt", ""))

    img_w, img_h = 0, 0
    style = element.get("style", "").lower()
    w_attr = element.get("width")
    h_attr = element.get("height")

    if w_attr and w_attr.isdigit():
        img_w = int(w_attr)
    if h_attr and h_attr.isdigit():
        img_h = int(h_attr)

    if img_w == 0:
        w_style = re.search(r"width\s*:\s*(\d+)", style)
        if w_style:
            img_w = int(w_style.group(1))
    if img_h == 0:
        h_style = re.search(r"height\s*:\s*(\d+)", style)
        if h_style:
            img_h = int(h_style.group(1))

    src = element.get("src") or element.get("data-src") or ""
    if (img_w == 0 or img_h == 0) and src:
        dim_match = re.search(r"[-_/=](\d{3,4})[xX](\d{3,4})", src)
        if dim_match:
            img_w, img_h = int(dim_match.group(1)), int(dim_match.group(2))

    if (img_w == 0 or img_h == 0) and src.startswith("http"):
        img_w, img_h = get_remote_image_dims(src)

    features.update(
        {
            "img_width": img_w,
            "img_height": img_h,
            "img_area_raw": img_w * img_h,
            "image_area": img_w * img_h,
        }
    )
    return features


def _get_interaction_features(
    element: lxml.html.HtmlElement,
    tag: str,
    text_len: int,
    parent_tag: str,
    gparent_tag: str,
) -> Dict[str, Any]:
    """
    Extract features related to user interaction (links, buttons).

    Args:
        element (lxml.html.HtmlElement): The target element.
        tag (str): Normalized tag name.
        text_len (int): Length of the text content.
        parent_tag (str): Normalized parent tag.
        gparent_tag (str): Normalized grandparent tag.

    Returns:
        Dict[str, Any]: Interaction feature dictionary.
    """
    is_clickable = 0
    role = element.get("role", "").lower()
    if tag in {"a", "button"} or role == "button":
        is_clickable = 1
    elif tag == "input" and element.get("type", "").lower() in {
        "submit",
        "button",
        "reset",
    }:
        is_clickable = 1

    link_density, parent_is_link = 0.0, 0
    if tag == "a" or parent_tag == "a" or gparent_tag == "a":
        link_density, parent_is_link = 1.0, 1
    else:
        links_text = sum(len(a.text_content() or "") for a in element.findall(".//a"))
        link_density = links_text / text_len if text_len > 0 else 0.0

    return {
        "is_clickable": is_clickable,
        "has_href": 1 if element.get("href") else 0,
        "link_density": link_density,
        "parent_is_link": parent_is_link,
    }


def _get_attribute_features(element: lxml.html.HtmlElement) -> Dict[str, str]:
    """
    Extract raw attribute strings for CSS classes and IDs.

    Args:
        element (lxml.html.HtmlElement): The target element.

    Returns:
        Dict[str, str]: Dictionary containing class and ID strings.
    """
    class_val = element.get("class")
    return {
        "class_str": " ".join(class_val.split()) if class_val else "",
        "id_str": element.get("id", ""),
    }


def extract_element_features(
    element: lxml.html.HtmlElement,
    selectors: Optional[Dict[str, List[str]]] = None,
    category: str = OTHER_CATEGORY,
) -> Dict[str, Any]:
    """
    Master function to extract all features from a single element.

    Args:
        element (lxml.html.HtmlElement): The target element.
        selectors (Optional[Dict[str, List[str]]]): Known selectors for context features.
        category (str): The label/category of this element.

    Returns:
        Dict[str, Any]: Complete dictionary of all extracted features.
    """
    if selectors is None:
        selectors = {}

    try:
        tag = normalize_tag(element.tag)

        # Extract individual feature groups
        struct_feats = _get_structure_features(element, tag, category)
        text_feats = _get_text_features(element)
        regex_feats = _get_regex_features(text_feats.pop("_text_content"))
        visual_feats = _get_visual_features(element, tag, struct_feats["parent_tag"])
        image_feats = _get_image_features(element, tag)
        interact_feats = _get_interaction_features(
            element,
            tag,
            text_feats["text_len"],
            struct_feats["parent_tag"],
            struct_feats["gparent_tag"],
        )
        attr_feats = _get_attribute_features(element)

        # Context features (calculated relative to other known categories)
        context_feats = {
            "avg_distance_to_closest_categories": get_avg_distance_to_closest_categories(
                element, selectors, category
            ),
            # Initialize page-level features to 0, processed later in process_page_features
            "img_size_rank": 0,
            "visibility_score_local": 0.0,
            "visibility_score_global": 0.0,
            "text_len_score_local": 0.0,
            "tag_count_global": 0,
        }

        metadata_feats = {"xpath": get_unique_xpath(element)}

        return {
            **struct_feats,
            **text_feats,
            **regex_feats,
            **visual_feats,
            **image_feats,
            **interact_feats,
            **attr_feats,
            **context_feats,
            **metadata_feats,
        }
    except Exception as e:  # pylint: disable=broad-exception-caught
        log_error(f"Error extracting features for element: {e}")
        fallback: Dict[str, Any] = {k: 0 for k in NUMERIC_FEATURES}
        fallback.update({k: "" for k in CATEGORICAL_FEATURES + TEXT_FEATURES})
        fallback["Category"] = category
        fallback["xpath"] = ""
        return fallback


def process_page_features(
    element_feature_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Post-process a list of element features to add page-level relative scores.

    Args:
        element_feature_list (List[Dict[str, Any]]): List of raw feature dictionaries for a page.

    Returns:
        List[Dict[str, Any]]: The processed list with relative ranking features added.
    """
    if not element_feature_list:
        return []
    df = pd.DataFrame(element_feature_list)

    # Image Size Ranking
    if "img_area_raw" in df.columns:
        mask_imgs = df["img_area_raw"] > 0
        if mask_imgs.any():
            try:
                # Rank images into deciles (1-10)
                df.loc[mask_imgs, "img_size_rank"] = (
                    pd.qcut(
                        df.loc[mask_imgs, "img_area_raw"],
                        q=10,
                        labels=False,
                        duplicates="drop",
                    ).astype(int)
                    + 1
                )
                df["img_size_rank"] = df["img_size_rank"].fillna(1)
            except ValueError:
                df.loc[mask_imgs, "img_size_rank"] = 1
        else:
            df["img_size_rank"] = 0
    else:
        df["img_size_rank"] = 0

    # Visibility Scores (Global & Local)
    if "visual_weight" in df.columns:
        avg_vis = df["visual_weight"].mean()
        df["visibility_score_global"] = (
            df["visual_weight"] / avg_vis if avg_vis > 0 else 0.0
        )

        # Local visibility (relative to neighbors)
        rolling_vis = (
            df["visual_weight"]
            .rolling(window=5, center=True, min_periods=1)
            .mean()
            .replace(0, 1)
        )
        df["visibility_score_local"] = df["visual_weight"] / rolling_vis
    else:
        df["visibility_score_global"] = 0.0
        df["visibility_score_local"] = 0.0

    # Text Length Local Score
    if "text_len" in df.columns:
        rolling_len = (
            df["text_len"]
            .rolling(window=7, center=True, min_periods=1)
            .mean()
            .replace(0, 1)
        )
        df["text_len_score_local"] = df["text_len"] / rolling_len
    else:
        df["text_len_score_local"] = 0.0

    # Tag Frequency
    if "tag" in df.columns:
        tag_counts = df["tag"].value_counts()
        df["tag_count_global"] = df["tag"].map(tag_counts).fillna(0).astype(int)
    else:
        df["tag_count_global"] = 0

    # Ensure all keys are strings for type compatibility
    return df.to_dict("records")  # pyright: ignore[reportReturnType]


def get_feature_columns() -> Dict[str, list]:
    """
    Get the configuration of feature columns grouped by type.

    Args:
        None

    Returns:
        Dict[str, list]: Dictionary mapping feature types (numeric, categorical, text) to column names.
    """
    return {
        "numeric": NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
        "text": TEXT_FEATURES,
        "all": ALL_FEATURES,
    }


def validate_features(df: Any) -> bool:
    """
    Validate that a DataFrame contains all required feature columns.

    Args:
        df (Any): The Pandas DataFrame to check.

    Returns:
        bool: True if valid, False if missing features or empty.
    """
    if df.empty:
        return False
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if "xpath" not in df.columns:
        missing.append("xpath")
    if missing:
        log_warning(f"Missing features: {missing}")
        return False
    return True
