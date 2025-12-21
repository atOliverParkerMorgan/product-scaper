"""
Unified feature definitions and extraction for HTML element analysis.
Refactored for modularity, readability, and robustness.
"""

import logging
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import lxml.html
import pandas as pd
import regex as re
import requests
from PIL import Image

from utils.utils import get_unique_xpath, normalize_tag
from utils.console import log_warning


# --- Configuration ---
TIMEOUT_SECONDS = 3
headers = {'User-Agent': 'Mozilla/5.0 (Bot)'}

# Constants
DEFAULT_DIST = 100.0
UNWANTED_TAGS = {'script', 'style', 'noscript', 'form', 'iframe', 'header', 'footer', 'nav'}
OTHER_CATEGORY = 'other'

# --- Currency & Keyword Configuration ---
ISO_CURRENCIES = [
    'AED', 'AFN', 'ALL', 'AMD', 'ANG', 'AOA', 'ARS', 'AUD', 'AWG', 'AZN',
    'BAM', 'BBD', 'BDT', 'BGN', 'BHD', 'BIF', 'BMD', 'BND', 'BOB', 'BRL',
    'BSD', 'BTN', 'BWP', 'BYN', 'BZD', 'CAD', 'CDF', 'CHF', 'CLP', 'CNY',
    'COP', 'CRC', 'CUP', 'CVE', 'CZK', 'DJF', 'DKK', 'DOP', 'DZD', 'EGP',
    'ERN', 'ETB', 'EUR', 'FJD', 'FKP', 'GBP', 'GEL', 'GHS', 'GIP', 'GMD',
    'GNF', 'GTQ', 'GYD', 'HKD', 'HNL', 'HRK', 'HTG', 'HUF', 'IDR', 'ILS',
    'INR', 'IQD', 'IRR', 'ISK', 'JMD', 'JOD', 'JPY', 'KES', 'KGS', 'KHR',
    'KMF', 'KPW', 'KRW', 'KWD', 'KYD', 'KZT', 'LAK', 'LBP', 'LKR', 'LRD',
    'LSL', 'LYD', 'MAD', 'MDL', "MGA", "MKD", "MMK", "MNT", "MOP", "MRU",
    'MUR', 'MVR', 'MWK', 'MXN', 'MYR', 'MZN', 'NAD', 'NGN', 'NIO', 'NOK',
    'NPR', 'NZD', 'OMR', 'PAB', 'PEN', 'PGK', 'PHP', 'PKR', 'PLN', 'PYG',
    'QAR', 'RON', 'RSD', 'RUB', 'RWF', 'SAR', 'SBD', 'SCR', 'SDG', 'SEK',
    'SGD', 'SHP', 'SLE', 'SLL', 'SOS', 'SRD', 'SSP', 'STN', 'SVC', 'SYP',
    'SZL', 'THB', 'TJS', 'TMT', 'TND', 'TOP', 'TRY', 'TTD', 'TWD', 'TZS',
    'UAH', 'UGX', 'USD', 'UYU', 'UZS', 'VES', 'VND', 'VUV', 'WST', 'XAF',
    'XCD', 'XOF', 'XPF', 'YER', 'ZAR', 'ZMW', 'ZWL'
]

SOLD_WORD_VARIATIONS = [
    'sold', 'sold out', 'out of stock', 'unavailable', 'discontinued',
    'vendu', 'épuisé', 'indisponible', ' rupture de stock',
    'verkauft', 'ausverkauft', 'nicht vorrätig', 'nicht lieferbar',
    'vendido', 'agotado', 'no disponible', 'fuera de stock',
    'venduto', 'esaurito', 'non disponibile',
    'vendido', 'esgotado', 'indisponível',
    'verkocht', 'niet op voorraad', 'uitverkocht',
    'såld', 'slutsåld', 'slut i lager', 'ej i lager',
    'solgt', 'udsolgt', 'ikke på lager',
    'myyty', 'loppu', 'ei varastossa',
    'продано', 'нет в наличии', 'раскуплено', 'закончился',
    'sprzedane', 'brak w magazynie', 'wyprzedane', 'niedostępny',
    'prodáno', 'vyprodáno', 'není skladem',
    'eladva', 'elfogyott', 'nincs raktáron',
    'vândut', 'stoc epuizat', 'indisponibil',
    'prodano', 'rasprodano', 'nema na zalihi',
    '售出', '已售出', '缺货', '暂时缺货', '售罄',
    '売り切れ', '在庫切れ', '完売', '品切れ',
    '품절', '매진', '재고 없음', '판매 완료',
    'đã bán', 'hết hàng', 'bán hết',
    'ขายแล้ว', 'สินค้าหมด', 'หมด',
    'terjual', 'habis', 'stok habis', 'kosong',
    'مباع', 'نفذ', 'نفذت الكمية', 'غير متوفر',
    'נמכר', 'אזל במלאי', 'לא זמין',
    'satıldı', 'tükendi', 'stokta yok', 'temin edilemiyor',
    'बिका हुआ', 'स्टॉक में नहीं', 'उपलब्ध नहीं',
    'ناموجود', 'فروخته شد',
    'εξαντλήθηκε', 'μη διαθέσιμο', 'κατόπιν παραγγελίας',
    'uppselt', 'ekki til',
    'išparduota', 'nėra prekyboje',
    'izpārdots', 'nav pieejams'
]

REVIEW_KEYWORDS = [
    'review', 'reviews', 'rating', 'ratings', 'stars', 'feedback',
    'testimonial', 'testimonials', 'comment', 'comments', 'opinion',
    'avis', 'commentaire', 'notations', 'témoignage',
    'bewertung', 'rezension', 'kundenmeinung', 'erfahrungsbericht',
    'opiniones', 'reseña', 'valoración', 'comentarios',
    'recensione', 'recensioni', 'opinioni', 'giudizi',
    'avaliação', 'opiniões', 'comentários', 'classificação',
    'beoordeling', 'recensie', 'klantbeoordeling', 'ervaringen',
    'recension', 'betyg', 'omdöme', 'kommentarer',
    'anmeldelse', 'vurdering',
    'arvostelu', 'arviot', 'kommentit',
    'обзор', 'отзыв', 'рейтинг', 'комментарии', 'оценка',
    'recenzja', 'opinia', 'ocena', 'komentarze',
    'recenze', 'hodnocení', 'názor',
    'vélemény', 'értékelés', 'hozzászólás',
    'recenzie', 'păreri', 'calificativ',
    '评价', '评论', '评分', '晒单',
    'レビュー', '口コミ', '評価', '評判', 'コメント',
    '리뷰', '평점', '후기', '댓글', '상품평',
    'đánh giá', 'nhận xét', 'bình luận',
    'รีวิว', 'ความเห็น', 'ให้คะแนน',
    'ulasan', 'penilaian', 'komentar', 'testimoni',
    'مراجعة', 'تقييم', 'آراء', 'تعليقات',
    'ביקורת', 'חוות דעת', 'דירוג',
    'inceleme', 'değerlendirme', 'yorum', 'puan',
    'समीक्षा', 'रेटिंग', 'टिप्पणी',
    'نقد', 'بررسی', 'نظر',
    'κριτική', 'αξιολόγηση', 'σχόλια'
]

CTA_KEYWORDS = [
    'add to cart', 'add to bag', 'add to basket', 'buy', 'buy now',
    'checkout', 'purchase', 'order', 'shop now', 'get it now',
    'ajouter au panier', 'acheter', 'commander', 'panier',
    'in den warenkorb', 'kaufen', 'jetzt kaufen', 'zur kasse', 'bestellen',
    'añadir al carrito', 'comprar', 'pagar', 'cesta', 'ordenar',
    'aggiungi al carrello', 'compra', 'acquista', 'ordina', 'cassa',
    'adicionar ao carrinho', 'comprar', 'finalizar compra', 'cesto',
    'in winkelwagen', 'kopen', 'bestellen', 'afrekenen',
    'lägg i varukorgen', 'köp', 'till kassan',
    'læg i kurv', 'køb', 'bestil',
    'legg i handlekurven', 'kjøp',
    'lisää ostoskoriin', 'osta', 'tilaa', 'kassalle',
    'купить', 'в корзину', 'оформить заказ', 'заказать',
    'dodaj do koszyka', 'kup', 'zamów', 'do kasy',
    'vložit do košíku', 'koupit', 'objednat', 'pokladna',
    'kosárba', 'megrendelés', 'vásárlás',
    'adaugă în coș', 'cumpără', 'comandă',
    '加入购物车', '购买', '立即购买', '结算', '下单',
    'カートに入れる', '購入', '注文する', 'レジに進む',
    '장바구니 담기', '구매', '주문하기', '결제',
    'thêm vào giỏ', 'mua ngay', 'thanh toán', 'đặt hàng',
    'หยิบใส่ตะกร้า', 'ซื้อเลย', 'ชำระเงิน', 'สั่งซื้อ',
    'tambah ke keranjang', 'beli', 'bayar', 'pesan',
    'أضف إلى السلة', 'شراء', 'إتمام الشراء', 'اطلب الآن',
    'הוסף לסל', 'קנה', 'תשלום', 'הזמן',
    'sepete ekle', 'satın al', 'sipariş ver', 'öde',
    'कार्ट में डालें', 'खरीदें', 'अभी खरीदें', 'ऑर्डर करें',
    'افزودن به سبد', 'خرید', 'سفارش',
    'προσθήκη στο καλάθι', 'αγορά', 'ταμείο', 'παραγγελία'
]

CUSTOM_SYMBOLS = [
    'Chf', 'Kč', 'kr', 'zł', 'Rs', 'Ft', 'lei', 'kn', 'din', 'руб', '₹', r'R\$', 'R',
    r',-', r'\.-'
]

# --- Compiled Regex Patterns ---
text_based_currencies = sorted(ISO_CURRENCIES + [s for s in CUSTOM_SYMBOLS if s.isalpha()])
symbol_based_currencies = [s for s in CUSTOM_SYMBOLS if not s.isalpha()]

# 1. Text codes: \b(USD|EUR|kr)\b
text_pattern = r'\b(?:' + '|'.join(re.escape(s) for s in text_based_currencies) + r')\b'
# 2. Symbols: (?:$|€|,-) -> No boundaries needed/wanted for things like ",-" or "$"
symbol_pattern = r'(?:' + '|'.join(re.escape(s) for s in symbol_based_currencies) + r')'
# 3. Unicode Symbols: \p{Sc}
unicode_pattern = r'\p{Sc}'

# Combine: Match Text OR Symbol OR Unicode
FULL_CURRENCY_PATTERN = f"{text_pattern}|{symbol_pattern}|{unicode_pattern}"
CURRENCY_HINTS_REGEX = re.compile(FULL_CURRENCY_PATTERN, re.UNICODE | re.IGNORECASE)

# Updated Number Pattern to allow dash decimals (e.g. 150,-)
NUMBER_PATTERN = r'(?:\d{1,3}(?:[., ]\d{3})+|\d+)(?:[.,](?:\d{1,2}|-))?'

PRICE_REGEX = re.compile(
    fr'(?:(?:{FULL_CURRENCY_PATTERN})\s*{NUMBER_PATTERN}|{NUMBER_PATTERN}\s*(?:{FULL_CURRENCY_PATTERN}))',
    re.UNICODE | re.IGNORECASE
)

SOLD_REGEX = re.compile(r'\b(?:' + '|'.join(SOLD_WORD_VARIATIONS) + r')\b', re.IGNORECASE)
REVIEW_REGEX = re.compile(r'\b(?:' + '|'.join(REVIEW_KEYWORDS) + r')\b', re.IGNORECASE)
CTA_REGEX = re.compile(r'\b(?:' + '|'.join(CTA_KEYWORDS) + r')\b', re.IGNORECASE)

# --- Feature Definitions ---

NUMERIC_FEATURES = [
    # Structural
    'num_children',
    'num_siblings',
    'dom_depth',
    'sibling_tag_ratio',

    # Flags
    'is_header',
    'is_block_element',
    'is_clickable',
    'is_formatting',
    'is_list_item',
    'is_bold',
    'is_italic',
    'is_hidden',

    # Text Metrics
    'text_len',
    'text_word_count',
    'text_digit_count',
    'text_density',
    'digit_density',
    'link_density',
    'capitalization_ratio',
    'avg_word_length',

    # Visual/Style
    'font_size',
    'font_weight',
    'visual_weight',

    # Visual/Style (Relative)
    'visibility_score_local',
    'visibility_score_global',

    # Image Features
    'img_width',
    'img_height',
    'img_area_raw',
    'image_area',
    'sibling_image_count',
    'img_size_rank',

    # Semantic
    'has_currency_symbol',
    'is_price_format',
    'has_sold_keyword',
    'has_review_keyword',
    'has_cta_keyword',

    # Attribute / Visual
    'has_href',
    'is_image',
    'has_src',
    'has_alt',
    'alt_len',
    'parent_is_link',

    'is_strikethrough',
    'tag_count_global',
    'avg_distance_to_closest_categories'
]

NON_TRAINING_FEATURES = [
    'Category',
    'SourceURL'
]

CATEGORICAL_FEATURES = [
    'tag',
    'parent_tag',
    'gparent_tag'
]

TEXT_FEATURES = [
    'class_str',
    'id_str'
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES
TARGET_FEATURE = 'Category'

# --- Helper Functions ---

@lru_cache(maxsize=256)
def get_remote_image_dims(url: str) -> Tuple[int, int]:
    """Fetches image dimensions from a URL without downloading the full body."""
    if not url or url.startswith('data:'):
        return 0, 0

    try:
        response = requests.get(url, headers=headers, timeout=TIMEOUT_SECONDS, stream=True)
        response.raise_for_status()
        p = Image.FileParser()
        for chunk in response.iter_content(chunk_size=1024):
            if not chunk: break
            p.feed(chunk)
            if p.image: return p.image.size
        return 0, 0
    except Exception:
        return 0, 0

def get_xpath_segments(xpath: str) -> List[str]:
    return [s for s in xpath.split('/') if s]

def extract_index(segment: str) -> int:
    match = re.search(r'\[(\d+)\]', segment)
    return int(match.group(1)) if match else 1

def calculate_proximity_score(xpath1: str, xpath2: str) -> tuple:
    """Calculate proximity (Tree Distance, Index Delta) between two XPaths."""
    path1 = get_xpath_segments(xpath1)
    path2 = get_xpath_segments(xpath2)

    min_len = min(len(path1), len(path2))
    divergence_index = 0
    for i in range(min_len):
        if path1[i] == path2[i]: divergence_index += 1
        else: break

    dist_up = len(path1) - divergence_index
    dist_down = len(path2) - divergence_index
    tree_distance = dist_up + dist_down

    index_delta = 0
    if divergence_index < len(path1) and divergence_index < len(path2):
        idx1 = extract_index(path1[divergence_index])
        idx2 = extract_index(path2[divergence_index])
        index_delta = abs(idx1 - idx2)

    return (tree_distance, index_delta)


def get_avg_distance_to_closest_categories(
    element: lxml.html.HtmlElement,
    selectors: Dict[str, List[str]]
) -> float:
    """Calculates average tree distance to specific selectors."""
    if not selectors:
        return DEFAULT_DIST

    min_distances = []
    elem_xpath = get_unique_xpath(element)

    for _, xpaths in selectors.items():
        distances = []
        for xpath in xpaths:
            dist_tuple = calculate_proximity_score(elem_xpath, xpath)
            distances.append(dist_tuple[0])
        if distances:
            min_distances.append(min(distances))

    return sum(min_distances) / len(min_distances) if min_distances else DEFAULT_DIST

# --- Feature Extraction Helpers ---

def _get_structure_features(element: lxml.html.HtmlElement, tag: str, category: str) -> Dict[str, Any]:
    parent = element.getparent()
    gparent = parent.getparent() if parent is not None else None

    parent_tag = normalize_tag(parent.tag) if parent is not None else 'root'
    gparent_tag = normalize_tag(gparent.tag) if gparent is not None else 'root'

    sibling_count = 0
    same_tag_count = 0
    sibling_image_count = 0

    if parent is not None:
        for child in parent:
            if child is element: continue
            child_tag = normalize_tag(getattr(child, 'tag', ''))
            if child_tag == tag: same_tag_count += 1
            if child_tag == 'img': sibling_image_count += 1
            sibling_count += 1

    return {
        'Category': category,
        'tag': tag,
        'parent_tag': parent_tag,
        'gparent_tag': gparent_tag,
        'num_children': len(element),
        'num_siblings': sibling_count,
        'dom_depth': len(list(element.iterancestors())),
        'sibling_tag_ratio': (same_tag_count / sibling_count) if sibling_count > 0 else 0.0,
        'sibling_image_count': sibling_image_count
    }

def _get_text_features(element: lxml.html.HtmlElement) -> Dict[str, Any]:
    raw_text = element.text_content() or ""
    text = " ".join(raw_text.split())
    text_len = len(text)

    # Calculate density relative to HTML source size
    try:
        html_size = len(lxml.html.tostring(element, encoding='unicode'))
        text_density = (text_len / html_size) if html_size > 0 else 0.0
    except Exception:
        text_density = 0.0

    words = text.split()
    digit_count = sum(c.isdigit() for c in text)

    return {
        'text_len': text_len,
        'text_word_count': len(words),
        'text_digit_count': digit_count,
        'text_density': text_density,
        'digit_density': (digit_count / text_len) if text_len > 0 else 0.0,
        'capitalization_ratio': (sum(1 for c in text if c.isupper()) / text_len) if text_len > 0 else 0.0,
        'avg_word_length': (sum(len(w) for w in words) / len(words)) if words else 0.0,
        '_text_content': text # Internal use for regex
    }

def _get_regex_features(text: str) -> Dict[str, int]:
    return {
        'has_currency_symbol': 1 if CURRENCY_HINTS_REGEX.search(text) else 0,
        'is_price_format': 1 if PRICE_REGEX.search(text) else 0,
        'has_sold_keyword': 1 if SOLD_REGEX.search(text) else 0,
        'has_review_keyword': 1 if REVIEW_REGEX.search(text) else 0,
        'has_cta_keyword': 1 if CTA_REGEX.search(text) else 0
    }

def _get_visual_features(element: lxml.html.HtmlElement, tag: str, parent_tag: str) -> Dict[str, Any]:
    style = element.get('style', '').lower()

    # Flags
    is_header = 1 if tag in {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'} else 0
    is_formatting = 1 if tag in {'b', 'strong', 'i', 'em', 'u', 'span', 'small', 'mark'} else 0
    is_block = 1 if tag in {'div', 'p', 'section', 'article', 'main', 'aside', 'header', 'footer', 'ul', 'ol', 'table', 'form'} else 0
    is_list_item = 1 if tag in {'li', 'dt', 'dd'} else 0

    # Visual Styles
    font_size = 16.0
    fs_match = re.search(r'font-size\s*:\s*([\d.]+)(px|em|rem|pt|%)?', style)
    if fs_match:
        try:
            val = float(fs_match.group(1))
            unit = fs_match.group(2)
            if unit in ('em', 'rem'): font_size = val * 16.0
            elif unit == '%': font_size = (val / 100.0) * 16.0
            elif unit == 'pt': font_size = val * 1.33
            else: font_size = val
        except ValueError: pass

    font_weight = 400
    fw_match = re.search(r'font-weight\s*:\s*(\w+)', style)
    if fw_match:
        w_str = fw_match.group(1)
        if w_str in {'bold', 'bolder'}: font_weight = 700
        elif w_str == 'lighter': font_weight = 300
        elif w_str.isdigit(): font_weight = int(w_str)
    elif tag in {'h1', 'h2', 'h3', 'b', 'strong'}:
        font_weight = 700

    # Strikethrough
    is_strikethrough = 0
    if tag in {'s', 'strike', 'del'} or 'line-through' in style or parent_tag in {'s', 'strike', 'del'}:
        is_strikethrough = 1

    return {
        'is_header': is_header,
        'is_formatting': is_formatting,
        'is_block_element': is_block,
        'is_list_item': is_list_item,
        'font_size': font_size,
        'font_weight': font_weight,
        'visual_weight': font_size * (font_weight / 400.0),
        'is_bold': 1 if (tag in {'b', 'strong', 'h1', 'h2', 'h3'} or font_weight >= 600 or 'font-weight:bold' in style) else 0,
        'is_italic': 1 if (tag in {'i', 'em'} or 'font-style:italic' in style) else 0,
        'is_hidden': 1 if ('display:none' in style or 'visibility:hidden' in style or 'opacity:0' in style.replace(' ', '')) else 0,
        'is_strikethrough': is_strikethrough
    }

def _get_image_features(element: lxml.html.HtmlElement, tag: str) -> Dict[str, Any]:
    features = {
        'is_image': 1 if tag == 'img' else 0,
        'has_src': 1 if element.get('src') or element.get('data-src') else 0,
        'has_alt': 1 if element.get('alt') else 0,
        'alt_len': len(element.get('alt', '')),
        'img_width': 0, 'img_height': 0, 'img_area_raw': 0
    }

    if tag != 'img':
        features['image_area'] = 0
        return features

    img_w, img_h = 0, 0
    style = element.get('style', '').lower()

    # 1. Attributes
    w_attr = element.get('width')
    h_attr = element.get('height')
    if w_attr and w_attr.isdigit(): img_w = int(w_attr)
    if h_attr and h_attr.isdigit(): img_h = int(h_attr)

    # 2. Inline Style
    if img_w == 0 or img_h == 0:
        w_style = re.search(r'width\s*:\s*(\d+)(?:px)', style)
        h_style = re.search(r'height\s*:\s*(\d+)(?:px)', style)
        if w_style: img_w = int(w_style.group(1))
        if h_style: img_h = int(h_style.group(1))

    # 3. SRC Regex
    src = element.get('src') or element.get('data-src') or ""
    if (img_w == 0 or img_h == 0) and src:
        dim_match = re.search(r'[-_/=](\d{3,4})[xX](\d{3,4})', src)
        if dim_match:
            try:
                img_w, img_h = int(dim_match.group(1)), int(dim_match.group(2))
            except: pass

    # 4. Network
    if (img_w == 0 or img_h == 0) and src.startswith('http'):
        img_w, img_h = get_remote_image_dims(src)

    features['img_width'] = img_w
    features['img_height'] = img_h
    features['img_area_raw'] = img_w * img_h
    features['image_area'] = features['img_area_raw']
    return features

def _get_interaction_features(element: lxml.html.HtmlElement, tag: str, text_len: int, parent_tag: str, gparent_tag: str) -> Dict[str, Any]:
    is_clickable = 0
    role = element.get('role', '').lower()

    if tag in {'a', 'button'} or role == 'button':
        is_clickable = 1
    elif tag == 'input' and element.get('type', '').lower() in {'submit', 'button', 'reset'}:
        is_clickable = 1

    # Link Density
    link_density = 0.0
    parent_is_link = 0
    if tag == 'a' or parent_tag == 'a' or gparent_tag == 'a':
        link_density = 1.0
        parent_is_link = 1
    else:
        links_text = sum(len(a.text_content() or "") for a in element.findall('.//a'))
        link_density = links_text / text_len if text_len > 0 else 0.0

    return {
        'is_clickable': is_clickable,
        'has_href': 1 if element.get('href') else 0,
        'link_density': link_density,
        'parent_is_link': parent_is_link
    }

def _get_attribute_features(element: lxml.html.HtmlElement) -> Dict[str, str]:
    class_val = element.get('class')
    return {
        'class_str': " ".join(class_val.split()) if class_val else "",
        'id_str': element.get('id', '')
    }

# --- Main Extraction Function ---

def extract_element_features(
    element: lxml.html.HtmlElement,
    selectors: Dict[str, List[str]] = {},
    category: str = OTHER_CATEGORY
) -> Dict[str, Any]:
    """
    Phase 1: Extraction. 
    Modularized extraction of absolute feature values.
    """
    try:
        tag = normalize_tag(element.tag)

        # 1. Structure
        struct_feats = _get_structure_features(element, tag, category)

        # 2. Text & Content
        text_feats = _get_text_features(element)
        regex_feats = _get_regex_features(text_feats.pop('_text_content'))

        # 3. Visual & Style
        visual_feats = _get_visual_features(element, tag, struct_feats['parent_tag'])

        # 4. Images
        image_feats = _get_image_features(element, tag)

        # 5. Interaction / Links
        interact_feats = _get_interaction_features(
            element, tag, text_feats['text_len'],
            struct_feats['parent_tag'], struct_feats['gparent_tag']
        )

        # 6. Attributes
        attr_feats = _get_attribute_features(element)

        # 7. Context / Proximity
        context_feats = {
            'avg_distance_to_closest_categories': get_avg_distance_to_closest_categories(element, selectors),
            # Init placeholders
            'img_size_rank': 0,
            'visibility_score_local': 0.0,
            'visibility_score_global': 0.0,
            'tag_count_global': 0
        }

        # Combine all features
        return {
            **struct_feats,
            **text_feats,
            **regex_feats,
            **visual_feats,
            **image_feats,
            **interact_feats,
            **attr_feats,
            **context_feats
        }

    except Exception as e:
        fallback = {k: 0 for k in NUMERIC_FEATURES}
        fallback.update({k: '' for k in CATEGORICAL_FEATURES + TEXT_FEATURES})
        fallback['Category'] = category
        return fallback

# --- Post Processing ---

def process_page_features(element_feature_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Takes the list of features for ALL elements on a page and computes relative rankings
    (Image Rank, Visibility Scores, Global Tag Counts).
    """
    if not element_feature_list:
        return []

    df = pd.DataFrame(element_feature_list)

    # Compute Image Rank 1-5
    # We only rank elements that actually have an area > 0
    if 'img_area_raw' in df.columns:
        mask_imgs = df['img_area_raw'] > 0
        if mask_imgs.any():
            try:
                # Create 5 buckets based on the distribution of images
                df.loc[mask_imgs, 'img_size_rank'] = pd.qcut(
                    df.loc[mask_imgs, 'img_area_raw'],
                    q=5, labels=[1, 2, 3, 4, 5], duplicates='drop'
                ).astype(int)
                # Fill NaN dropped duplicates or 0 area with 1
                df['img_size_rank'] = df['img_size_rank'].fillna(1)
            except ValueError:
                # Fallback if not enough unique values
                df.loc[mask_imgs, 'img_size_rank'] = 1
        else:
             df['img_size_rank'] = 0
    else:
        df['img_size_rank'] = 0

    # Compute Visibility Scores
    if 'visual_weight' in df.columns:
        # Global Visibility (vs Page Average)
        avg_visual_weight = df['visual_weight'].mean()
        if avg_visual_weight > 0:
            df['visibility_score_global'] = df['visual_weight'] / avg_visual_weight
        else:
            df['visibility_score_global'] = 0.0

        # Local Visibility (vs Neighbors)
        # Rolling average of 5 elements (2 before, self, 2 after)
        rolling_avg = df['visual_weight'].rolling(window=5, center=True, min_periods=1).mean()
        rolling_avg = rolling_avg.replace(0, 1) # Prevent divide by zero
        df['visibility_score_local'] = df['visual_weight'] / rolling_avg
    else:
        df['visibility_score_global'] = 0.0
        df['visibility_score_local'] = 0.0

    #  Compute Global Tag Counts
    if 'tag' in df.columns:
        tag_counts = df['tag'].value_counts()
        df['tag_count_global'] = df['tag'].map(tag_counts).fillna(0).astype(int)
    else:
        df['tag_count_global'] = 0

    return df.to_dict('records')


def get_feature_columns() -> Dict[str, list]:
    return {
        'numeric': NUMERIC_FEATURES,
        'categorical': CATEGORICAL_FEATURES,
        'text': TEXT_FEATURES,
        'all': ALL_FEATURES
    }

def validate_features(df: Any) -> bool:
    if df.empty: return False
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        log_warning(f"Missing features: {missing}")
        return False
    return True
