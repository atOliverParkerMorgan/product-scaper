"""
Unified feature definitions and extraction for HTML element analysis.

This module provides feature extraction, validation, and column definitions for HTML elements
used in product scraping and machine learning models.
"""

import logging
from typing import Any, Dict

import lxml.html
import regex as re

from utils.utils import normalize_tag

# Configure logging
logger = logging.getLogger(__name__)

# Constants for feature extraction
UNWANTED_TAGS = {'script', 'style', 'noscript', 'form', 'iframe', 'header', 'footer', 'nav'}
OTHER_CATEGORY = 'other'

# --- Currency Configuration (Unchanged) ---
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

# --- Keyword Lists (Unchanged) ---
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
    'Chf', 'Kč', 'kr', 'zł', 'Rs', 'Ft', 'lei', 'kn', 'din', 'руб', '₹', r'R\$', 'R'
]

# --- Compiled Regex Patterns (Unchanged) ---
CURRENCY_PATTERN_STR = r'\p{Sc}|' + '|'.join(re.escape(sym) for sym in CUSTOM_SYMBOLS + sorted(ISO_CURRENCIES))
CURRENCY_HINTS_REGEX = re.compile(fr'(?:{CURRENCY_PATTERN_STR})\b', re.UNICODE | re.IGNORECASE)
NUMBER_PATTERN = r'(?:\d{1,3}(?:[., ]\d{3})+|\d+)(?:[.,]\d{1,2})?'
PRICE_REGEX = re.compile(
    fr'(?:(?:{CURRENCY_PATTERN_STR})\s*{NUMBER_PATTERN}|{NUMBER_PATTERN}\s*(?:{CURRENCY_PATTERN_STR}))',
    re.UNICODE | re.IGNORECASE
)
SOLD_REGEX = re.compile(r'\b(?:' + '|'.join(SOLD_WORD_VARIATIONS) + r')\b', re.IGNORECASE)
REVIEW_REGEX = re.compile(r'\b(?:' + '|'.join(REVIEW_KEYWORDS) + r')\b', re.IGNORECASE)
CTA_REGEX = re.compile(r'\b(?:' + '|'.join(CTA_KEYWORDS) + r')\b', re.IGNORECASE)

TARGET_FEATURE = 'Category'

# --- Feature Definitions (UPDATED) ---
NUMERIC_FEATURES = [
    # Structural
    'num_children',
    'num_siblings',
    'dom_depth',
    'sibling_tag_ratio', # NEW: % of siblings that have the same tag as me

    # Flags (0/1)
    'is_header',
    'is_formatting',
    'is_list_item',
    'is_block_element',
    'is_clickable',      # NEW: Combined logic for a, button, input[submit]

    # Text Metrics
    'text_len',
    'text_word_count',
    'text_digit_count',
    'text_density',
    'digit_density',     # NEW: text_digit_count / text_len
    'link_density',
    'capitalization_ratio',
    'avg_word_length',

    # Visual/Style Features
    'font_size',
    'font_weight',
    'is_bold',
    'is_italic',
    'is_hidden',

    # Semantic / Regex
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
    'image_area',
    'parent_is_link',
    'sibling_image_count',

    # Class/ID heuristics
    'class_indicates_price',
    'class_indicates_title',
    'class_indicates_description',
    'class_indicates_image'
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


def extract_element_features(
    element: lxml.html.HtmlElement,
    category: str = OTHER_CATEGORY
) -> Dict[str, Any]:
    """
    Extract comprehensive features from a single HTML element.
    Includes robust surface calculation, sibling homogeneity, and digit density.
    """
    try:
        # --- 1. Text & Hierarchy Normalization ---
        raw_text = element.text_content()
        # Clean text: remove newlines/tabs, collapse spaces
        text = " ".join(raw_text.split())

        parent = element.getparent()
        gparent = parent.getparent() if parent is not None else None

        tag = normalize_tag(element.tag)
        parent_tag = normalize_tag(parent.tag) if parent is not None else 'root'
        gparent_tag = normalize_tag(gparent.tag) if gparent is not None else 'root'

        # --- 2. Sibling Analysis (Homogeneity) ---
        sibling_count = 0
        same_tag_count = 0
        sibling_image_count = 0

        if parent is not None:
            # We iterate parent's children. 'element' is one of them.
            for child in parent:
                if child is element:
                    continue

                child_tag = normalize_tag(getattr(child, 'tag', ''))

                # Check for same tag (homogeneity)
                if child_tag == tag:
                    same_tag_count += 1

                # Check for image siblings
                if child_tag == 'img':
                    sibling_image_count += 1

                sibling_count += 1

        sibling_tag_ratio = (same_tag_count / sibling_count) if sibling_count > 0 else 0.0

        features = {
            'Category': category,
            'tag': tag,
            'parent_tag': parent_tag,
            'gparent_tag': gparent_tag,
            'num_children': len(element),
            'num_siblings': sibling_count,
            'sibling_tag_ratio': sibling_tag_ratio,
            'dom_depth': len(list(element.iterancestors())),
            'sibling_image_count': sibling_image_count
        }

        # --- 3. Tag Type & Clickability ---
        features['is_header'] = 1 if tag in {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'} else 0
        features['is_formatting'] = 1 if tag in {'b', 'strong', 'i', 'em', 'u', 'span', 'small', 'mark'} else 0
        features['is_list_item'] = 1 if tag in {'li', 'dt', 'dd'} else 0
        features['is_block_element'] = 1 if tag in {'div', 'p', 'section', 'article', 'main', 'aside', 'header', 'footer', 'ul', 'ol', 'table', 'form'} else 0
        features['is_image'] = 1 if tag == 'img' else 0

        # Clickable detection: Anchor, Button, Input Submit, or Role=Button
        is_clickable = 0
        role = element.get('role', '').lower()
        if tag == 'a' or tag == 'button':
            is_clickable = 1
        elif tag == 'input' and element.get('type', '').lower() in {'submit', 'button', 'reset'}:
            is_clickable = 1
        elif role == 'button':
            is_clickable = 1
        features['is_clickable'] = is_clickable

        # --- 4. Visual/Style Features ---
        style = element.get('style', '').lower()

        # Font Size
        font_size = 16.0
        fs_match = re.search(r'font-size\s*:\s*([\d.]+)(px|em|rem|pt|%)?', style)
        if fs_match:
            try:
                val = float(fs_match.group(1))
                unit = fs_match.group(2)
                if unit == 'em' or unit == 'rem': font_size = val * 16.0
                elif unit == '%':
                    font_size = (val / 100.0) * 16.0
                elif unit == 'pt':
                    font_size = val * 1.33
                else:
                    font_size = val # Default px
            except ValueError:
                pass
        features['font_size'] = font_size

        # Font Weight
        font_weight = 400
        fw_match = re.search(r'font-weight\s*:\s*(\w+)', style)
        if fw_match:
            w_str = fw_match.group(1)
            if w_str in {'bold', 'bolder'}: font_weight = 700
            elif w_str in {'lighter'}: font_weight = 300
            elif w_str.isdigit(): font_weight = int(w_str)
        features['font_weight'] = font_weight

        # Visibility
        features['is_hidden'] = 1 if ('display:none' in style or 'visibility:hidden' in style or 'opacity:0' in style.replace(' ', '')) else 0

        # Style booleans
        features['is_bold'] = 1 if (tag in {'b', 'strong', 'h1', 'h2', 'h3'} or font_weight >= 600 or 'font-weight:bold' in style) else 0
        features['is_italic'] = 1 if (tag in {'i', 'em'} or 'font-style:italic' in style) else 0

        # --- 5. Attribute & Class Heuristics ---
        class_val = element.get('class')
        class_str = " ".join(class_val.split()) if class_val else ""
        id_str = element.get('id', '')

        features['class_str'] = class_str
        features['id_str'] = id_str

        meta_str = (class_str + " " + id_str).lower()
        features['class_indicates_price'] = 1 if any(x in meta_str for x in ['price', 'cost', 'amount', 'currency', 'money']) else 0
        features['class_indicates_title'] = 1 if any(x in meta_str for x in ['title', 'header', 'heading', 'name', 'subject', 'label']) else 0
        features['class_indicates_description'] = 1 if any(x in meta_str for x in ['desc', 'detail', 'summary', 'content', 'info']) else 0
        features['class_indicates_image'] = 1 if any(x in meta_str for x in ['image', 'img', 'photo', 'thumb', 'pic', 'media']) else 0

        # --- 6. Text Metrics & Density ---
        text_len = len(text)
        features['text_len'] = text_len

        words = text.split()
        features['text_word_count'] = len(words)

        digit_count = sum(c.isdigit() for c in text)
        features['text_digit_count'] = digit_count
        features['digit_density'] = (digit_count / text_len) if text_len > 0 else 0.0

        features['avg_word_length'] = (sum(len(w) for w in words) / len(words)) if words else 0.0
        features['capitalization_ratio'] = (sum(1 for c in text if c.isupper()) / text_len) if text_len > 0 else 0.0

        # Regex Matches
        features['has_currency_symbol'] = 1 if CURRENCY_HINTS_REGEX.search(text) else 0
        features['is_price_format'] = 1 if PRICE_REGEX.search(text) else 0
        features['has_sold_keyword'] = 1 if SOLD_REGEX.search(text) else 0
        features['has_review_keyword'] = 1 if REVIEW_REGEX.search(text) else 0
        features['has_cta_keyword'] = 1 if CTA_REGEX.search(text) else 0

        # Text Density (Content / Tree Size)
        num_descendants = sum(1 for _ in element.iterdescendants()) + 1
        features['text_density'] = text_len / num_descendants if num_descendants > 0 else 0

        # Link Density (Text in links / Total text)
        # Improvement: If I am an anchor, my link density is 1.0 (pure link)
        if tag == 'a' or parent_tag == 'a' or gparent_tag == 'a':
            features['link_density'] = 1.0
            features['parent_is_link'] = 1
        else:
            links_text = sum(len(a.text_content() or "") for a in element.findall('.//a'))
            features['link_density'] = links_text / text_len if text_len > 0 else 0.0
            features['parent_is_link'] = 0

        features['has_href'] = 1 if element.get('href') else 0

        # --- 7. Robust Image Surface Calculation ---
        features['has_src'] = 1 if element.get('src') or element.get('data-src') else 0
        features['has_alt'] = 1 if element.get('alt') else 0
        features['alt_len'] = len(element.get('alt', ''))

        img_w, img_h = 0, 0

        # 7a. Try explicit attributes (support % by ignoring it or treating as 0)
        try:
            w_attr = element.get('width')
            h_attr = element.get('height')
            if w_attr and w_attr.isdigit(): img_w = int(w_attr)
            if h_attr and h_attr.isdigit(): img_h = int(h_attr)
        except:
            pass

        # 7b. Try SRC/DATA-SRC URL Heuristics (e.g. image-800x600.jpg)
        if (img_w == 0 or img_h == 0):
            # Check both src and data-src (common in lazy loading)
            src_candidates = [element.get('src', ''), element.get('data-src', '')]
            for src in src_candidates:
                if not src:
                    continue

                # Pattern: 500x500
                dim_match = re.search(r'[-_/=](\d{2,4})[xX](\d{2,4})?', src)
                if dim_match:
                    try:
                        img_w = int(dim_match.group(1))
                        img_h = int(dim_match.group(2)) if dim_match.group(2) else img_w
                        break # Found a match
                    except:
                        pass

                # Pattern: width=500
                if img_w == 0:
                    w_param = re.search(r'[?&](?:width|w|resize)=?(\d+)', src)
                    if w_param:
                        try:
                            img_w = int(w_param.group(1))
                            break
                        except:
                            pass

        # 7c. Try Srcset (Largest available)
        if img_w == 0:
            srcset = element.get('srcset', '') or element.get('data-srcset', '')
            if srcset:
                widths = re.findall(r'(\d+)w', srcset)
                if widths:
                    try:
                        img_w = max(map(int, widths))
                    except:
                        pass

        # 7d. Try Inline Style
        if img_w == 0 or img_h == 0:
            w_style = re.search(r'width\s*:\s*(\d+)(?:px)', style) # Only px usually reliable for int extraction
            h_style = re.search(r'height\s*:\s*(\d+)(?:px)', style)
            if w_style:
                img_w = int(w_style.group(1))
            if h_style:
                img_h = int(h_style.group(1))

        # 7e. Aspect Ratio Assumption (If we only found one dim)
        if img_w > 0 and img_h == 0: img_h = int(img_w * 0.75)
        if img_h > 0 and img_w == 0: img_w = int(img_h * 1.33)

        features['image_area'] = img_w * img_h

        return features

    except Exception as e:
        logger.debug(f"Error extracting features from element: {e}")
        # Return safe fallback with all zeros
        return {k: 0 for k in NUMERIC_FEATURES}


def get_feature_columns() -> Dict[str, list]:
    """
    Get the complete list of feature columns by type.
    """
    return {
        'numeric': NUMERIC_FEATURES,
        'categorical': CATEGORICAL_FEATURES,
        'text': TEXT_FEATURES,
        'all': ALL_FEATURES
    }


def validate_features(df: Any) -> bool:
    """
    Validate that a DataFrame contains all required features.
    """
    if df.empty:
        return False
    missing_features = []
    for feature in ALL_FEATURES:
        if feature not in df.columns:
            missing_features.append(feature)
    if missing_features:
        logger.warning(f"Missing features in DataFrame: {missing_features}")
        return False
    return True
