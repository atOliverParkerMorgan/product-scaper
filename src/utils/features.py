"""Unified feature definitions and extraction for HTML element analysis."""

import regex as re
import logging
import lxml.html
import textstat
from typing import Dict, Any
from utils.utils import normalize_tag

# Configure logging
logger = logging.getLogger(__name__)

# Constants for feature extraction
UNWANTED_TAGS = {'script', 'style', 'noscript', 'form', 'iframe', 'header', 'footer', 'nav'}
OTHER_CATEGORY = 'other'

# --- Currency Configuration ---
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

# --- Keyword Lists & Patterns ---

SOLD_WORD_VARIATIONS = [
    # English
    'sold', 'sold out', 'out of stock', 'unavailable', 'discontinued',
    
    # Western European (Fr, De, Es, It, Pt, Nl)
    'vendu', 'épuisé', 'indisponible', ' rupture de stock', # French
    'verkauft', 'ausverkauft', 'nicht vorrätig', 'nicht lieferbar', # German
    'vendido', 'agotado', 'no disponible', 'fuera de stock', # Spanish
    'venduto', 'esaurito', 'non disponibile', # Italian
    'vendido', 'esgotado', 'indisponível', # Portuguese
    'verkocht', 'niet op voorraad', 'uitverkocht', # Dutch
    
    # Northern European (Sv, Da, No, Fi)
    'såld', 'slutsåld', 'slut i lager', 'ej i lager', # Swedish
    'solgt', 'udsolgt', 'ikke på lager', # Danish/Norwegian
    'myyty', 'loppu', 'ei varastossa', # Finnish
    
    # Eastern European (Ru, Pl, Cz, Hu, Ro, Uk, Hr/Sr)
    'продано', 'нет в наличии', 'раскуплено', 'закончился', # Russian/Ukranian
    'sprzedane', 'brak w magazynie', 'wyprzedane', 'niedostępny', # Polish
    'prodáno', 'vyprodáno', 'není skladem', # Czech
    'eladva', 'elfogyott', 'nincs raktáron', # Hungarian
    'vândut', 'stoc epuizat', 'indisponibil', # Romanian
    'prodano', 'rasprodano', 'nema na zalihi', # Croatian/Serbian
    
    # Asian (Zh, Ja, Ko, Vi, Th, Id)
    '售出', '已售出', '缺货', '暂时缺货', '售罄', # Chinese (Simplified/Traditional)
    '売り切れ', '在庫切れ', '完売', '品切れ', # Japanese
    '품절', '매진', '재고 없음', '판매 완료', # Korean
    'đã bán', 'hết hàng', 'bán hết', # Vietnamese
    'ขายแล้ว', 'สินค้าหมด', 'หมด', # Thai
    'terjual', 'habis', 'stok habis', 'kosong', # Indonesian/Malay
    
    # Middle Eastern / South Asian (Ar, He, Tr, Hi)
    'مباع', 'نفذ', 'نفذت الكمية', 'غير متوفر', # Arabic
    'נמכר', 'אזל במלאי', 'לא זמין', # Hebrew
    'satıldı', 'tükendi', 'stokta yok', 'temin edilemiyor', # Turkish
    'बिका हुआ', 'स्टॉक में नहीं', 'उपलब्ध नहीं', # Hindi
    'ناموجود', 'فروخته شد', # Persian
    
    # Others (Gr, Is, Et, Lt, Lv)
    'εξαντλήθηκε', 'μη διαθέσιμο', 'κατόπιν παραγγελίας', # Greek
    'uppselt', 'ekki til', # Icelandic
    'išparduota', 'nėra prekyboje', # Lithuanian
    'izpārdots', 'nav pieejams' # Latvian
]

# --- 2. REVIEW / RATING KEYWORDS ---
# Concepts: Review, Rating, Stars, Feedback, Comment, Opinion
REVIEW_KEYWORDS = [
    # English
    'review', 'reviews', 'rating', 'ratings', 'stars', 'feedback', 
    'testimonial', 'testimonials', 'comment', 'comments', 'opinion',
    
    # Western European
    'avis', 'commentaire', 'notations', 'témoignage', # French
    'bewertung', 'rezension', 'kundenmeinung', 'erfahrungsbericht', # German
    'opiniones', 'reseña', 'valoración', 'comentarios', # Spanish
    'recensione', 'recensioni', 'opinioni', 'giudizi', # Italian
    'avaliação', 'opiniões', 'comentários', 'classificação', # Portuguese
    'beoordeling', 'recensie', 'klantbeoordeling', 'ervaringen', # Dutch
    
    # Northern European
    'recension', 'betyg', 'omdöme', 'kommentarer', # Swedish
    'anmeldelse', 'vurdering', # Danish/Norwegian
    'arvostelu', 'arviot', 'kommentit', # Finnish
    
    # Eastern European
    'обзор', 'отзыв', 'рейтинг', 'комментарии', 'оценка', # Russian
    'recenzja', 'opinia', 'ocena', 'komentarze', # Polish
    'recenze', 'hodnocení', 'názor', # Czech
    'vélemény', 'értékelés', 'hozzászólás', # Hungarian
    'recenzie', 'păreri', 'calificativ', # Romanian
    
    # Asian
    '评价', '评论', '评分', '晒单', # Chinese
    'レビュー', '口コミ', '評価', '評判', 'コメント', # Japanese
    '리뷰', '평점', '후기', '댓글', '상품평', # Korean
    'đánh giá', 'nhận xét', 'bình luận', # Vietnamese
    'รีวิว', 'ความเห็น', 'ให้คะแนน', # Thai
    'ulasan', 'penilaian', 'komentar', 'testimoni', # Indonesian
    
    # Middle Eastern / South Asian
    'مراجعة', 'تقييم', 'آراء', 'تعليقات', # Arabic
    'ביקורת', 'חוות דעת', 'דירוג', # Hebrew
    'inceleme', 'değerlendirme', 'yorum', 'puan', # Turkish
    'समीक्षा', 'रेटिंग', 'टिप्पणी', # Hindi
    'نقد', 'بررسی', 'نظر' # Persian
    
    # Greek
    'κριτική', 'αξιολόγηση', 'σχόλια' # Greek
]

# --- 3. CALL TO ACTION (CTA) KEYWORDS ---
# Concepts: Buy, Add to Cart, Checkout, Purchase, Order, Basket
CTA_KEYWORDS = [
    # English
    'add to cart', 'add to bag', 'add to basket', 'buy', 'buy now', 
    'checkout', 'purchase', 'order', 'shop now', 'get it now',
    
    # Western European
    'ajouter au panier', 'acheter', 'commander', 'panier', # French
    'in den warenkorb', 'kaufen', 'jetzt kaufen', 'zur kasse', 'bestellen', # German
    'añadir al carrito', 'comprar', 'pagar', 'cesta', 'ordenar', # Spanish
    'aggiungi al carrello', 'compra', 'acquista', 'ordina', 'cassa', # Italian
    'adicionar ao carrinho', 'comprar', 'finalizar compra', 'cesto', # Portuguese
    'in winkelwagen', 'kopen', 'bestellen', 'afrekenen', # Dutch
    
    # Northern European
    'lägg i varukorgen', 'köp', 'till kassan', # Swedish
    'læg i kurv', 'køb', 'bestil', # Danish
    'legg i handlekurven', 'kjøp', # Norwegian
    'lisää ostoskoriin', 'osta', 'tilaa', 'kassalle', # Finnish
    
    # Eastern European
    'купить', 'в корзину', 'оформить заказ', 'заказать', # Russian
    'dodaj do koszyka', 'kup', 'zamów', 'do kasy', # Polish
    'vložit do košíku', 'koupit', 'objednat', 'pokladna', # Czech
    'kosárba', 'megrendelés', 'vásárlás', # Hungarian
    'adaugă în coș', 'cumpără', 'comandă', # Romanian
    
    # Asian
    '加入购物车', '购买', '立即购买', '结算', '下单', # Chinese
    'カートに入れる', '購入', '注文する', 'レジに進む', # Japanese
    '장바구니 담기', '구매', '주문하기', '결제', # Korean
    'thêm vào giỏ', 'mua ngay', 'thanh toán', 'đặt hàng', # Vietnamese
    'หยิบใส่ตะกร้า', 'ซื้อเลย', 'ชำระเงิน', 'สั่งซื้อ', # Thai
    'tambah ke keranjang', 'beli', 'bayar', 'pesan', # Indonesian
    
    # Middle Eastern / South Asian
    'أضف إلى السلة', 'شراء', 'إتمام الشراء', 'اطلب الآن', # Arabic
    'הוסף לסל', 'קנה', 'תשלום', 'הזמן', # Hebrew
    'sepete ekle', 'satın al', 'sipariş ver', 'öde', # Turkish
    'कार्ट में डालें', 'खरीदें', 'अभी खरीदें', 'ऑर्डर करें', # Hindi
    'افزودن به سبد', 'خرید', 'سفارش' # Persian
    
    # Greek
    'προσθήκη στο καλάθι', 'αγορά', 'ταμείο', 'παραγγελία' # Greek
]

# --- 4. AUTHORSHIP KEYWORDS ---
# Concepts: By, Author, Written by, Posted by
AUTHOR_KEYWORDS = [
    # English
    'by', 'author', 'written by', 'posted by', 'published by', 'creator',
    
    # Western European
    'par', 'auteur', 'écrit par', 'publié par', # French
    'von', 'autor', 'verfasst von', 'geschrieben von', 'urheber', # German
    'por', 'autor', 'escrito por', 'publicado por', # Spanish/Portuguese
    'di', 'autore', 'scritto da', 'pubblicato da', # Italian
    'door', 'auteur', 'geschreven door', 'geplaatst door', # Dutch
    
    # Northern European
    'av', 'författare', 'skriven av', # Swedish/Norwegian
    'af', 'forfatter', # Danish
    'kirjoittanut', 'tekijä', 'julkaissut', # Finnish
    
    # Eastern European
    'автор', 'написал', 'опубликовал', # Russian
    'autor', 'napisane przez', 'opublikowane przez', # Polish
    'napsal', 'autor', # Czech
    'szerző', 'írta', # Hungarian
    'de', 'scris de', # Romanian
    
    # Asian
    '作者', '编辑', '撰稿', # Chinese
    '著者', '作者', '執筆', '作成者', # Japanese
    '저자', '글쓴이', '작성자', # Korean
    'tác giả', 'viết bởi', 'đăng bởi', # Vietnamese
    'ผู้เขียน', 'โดย', # Thai
    'penulis', 'oleh', 'dibuat oleh', # Indonesian
    
    # Middle Eastern / South Asian
    'بقلم', 'الكاتب', 'المؤلف', 'نشر بواسطة', # Arabic
    'מאת', 'מחבר', 'נכתב על ידי', # Hebrew
    'yazar', 'tarafından', 'yazan', # Turkish
    'लेखक', 'द्वारा', 'रचयिता', # Hindi
    'نویسنده', 'اثر' # Persian
]

# Custom/Local symbols and variations (literal strings to match)
CUSTOM_SYMBOLS = [
    'Chf', 'Kč', 'kr', 'zł', 'Rs', 'Ft', 'lei', 'kn', 'din', 'руб', '₹', r'R\$', 'R'
]

# --- Compiled Regex Patterns ---

# 1. Currency & Price
# Combine Unicode currency symbols (\p{Sc}) with custom symbols and ISO codes
CURRENCY_PATTERN_STR = r'\p{Sc}|' + '|'.join(re.escape(sym) for sym in CUSTOM_SYMBOLS + sorted(ISO_CURRENCIES))
CURRENCY_HINTS_REGEX = re.compile(fr'(?:{CURRENCY_PATTERN_STR})\b', re.UNICODE | re.IGNORECASE)

NUMBER_PATTERN = r'(?:\d{1,3}(?:[., ]\d{3})+|\d+)(?:[.,]\d{1,2})?'

# Fixed: Use the pattern string, not the compiled regex object
PRICE_REGEX = re.compile(
    fr'(?:(?:{CURRENCY_PATTERN_STR})\s*{NUMBER_PATTERN}|{NUMBER_PATTERN}\s*(?:{CURRENCY_PATTERN_STR}))',
    re.UNICODE | re.IGNORECASE
)

# 2. Semantic Keywords
SOLD_REGEX = re.compile(r'\b(?:' + '|'.join(SOLD_WORD_VARIATIONS) + r')\b', re.IGNORECASE)
REVIEW_REGEX = re.compile(r'\b(?:' + '|'.join(REVIEW_KEYWORDS) + r')\b', re.IGNORECASE)
CTA_REGEX = re.compile(r'\b(?:' + '|'.join(CTA_KEYWORDS) + r')\b', re.IGNORECASE)
AUTHOR_REGEX = re.compile(r'\b(?:' + '|'.join(AUTHOR_KEYWORDS) + r')\s+\w+', re.IGNORECASE)

# 3. Dates (Simple heuristics for YYYY-MM-DD, DD.MM.YYYY, Month DD, YYYY)
DATE_REGEX = re.compile(
    r'(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})|'  # 12-12-2024
    r'(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})', # Dec 12, 2024
    re.IGNORECASE
)

TARGET_FEATURE = 'Category'

# Feature column definitions
NUMERIC_FEATURES = [
    # Structural
    'num_children',
    'num_siblings',
    'dom_depth',
    'is_header',         # New
    'is_formatting',     # New (bold/italic)
    
    # Text Metrics
    'text_len',
    'text_word_count',
    'text_digit_count',
    'text_density',
    'link_density',      # New
    'capitalization_ratio', # New
    'reading_ease',
    
    # Semantic / Regex
    'has_currency_symbol',
    'is_price_format',
    'has_sold_keyword',  # New
    'has_review_keyword', # New
    'has_cta_keyword',    # New
    'has_author_keyword', # New
    'has_date_pattern',   # New
    
    # Attribute / Visual
    'has_href',
    'is_image',
    'has_src',
    'has_alt',
    'alt_len',
    'image_area',        # Surface area (width * height) - better than binary flag
    'parent_is_link',
    'sibling_image_count',
    
    # Class heuristics
    'class_indicates_price', # New
    'class_indicates_title'  # New
]

NON_TRAINING_FEATURES = [
    'Category',
    'SourceURL'
]

CATEGORICAL_FEATURES = [
    'tag',
    'parent_tag',
    'gparent_tag',
    'ggparent_tag'
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
    """
    try:
        # Normalize text
        raw_text = element.text_content()
        text = raw_text.strip()
        
        # Get hierarchy
        parent = element.getparent()
        gparent = parent.getparent() if parent is not None else None
        ggparent = gparent.getparent() if gparent is not None else None
        
        # --- 1. Base Structural Features ---
        features = {
            'Category': category,
            'tag': normalize_tag(element.tag),
            'parent_tag': normalize_tag(parent.tag) if parent is not None else 'root',
            'gparent_tag': normalize_tag(gparent.tag) if gparent is not None else 'root',
            'ggparent_tag': normalize_tag(ggparent.tag) if ggparent is not None else 'root',
            'num_children': len(element),
            'num_siblings': len(parent) - 1 if parent is not None else 0,
            'dom_depth': len(list(element.iterancestors())),
        }
        
        # Specific tag checks
        tag = features['tag']
        features['is_header'] = 1 if tag in {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'} else 0
        features['is_formatting'] = 1 if tag in {'b', 'strong', 'i', 'em', 'u'} else 0

        # --- 2. Attribute Semantic Features ---
        class_str = " ".join(element.get('class', '').split()) if element.get('class') else ""
        features['class_str'] = class_str
        features['id_str'] = element.get('id', '')
        
        # Heuristics on class names (common in frameworks like Bootstrap/Tailwind)
        class_lower = class_str.lower()
        features['class_indicates_price'] = 1 if any(x in class_lower for x in ['price', 'cost', 'amount', 'currency']) else 0
        features['class_indicates_title'] = 1 if any(x in class_lower for x in ['title', 'header', 'heading', 'name']) else 0

        # --- 3. Text Content Features ---
        features['text_len'] = len(text)
        features['text_word_count'] = len(text.split())
        features['text_digit_count'] = sum(c.isdigit() for c in text)
        
        # Capitalization (Useful for Titles vs Descriptions)
        features['capitalization_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0.0

        # Regex Matches
        features['has_currency_symbol'] = 1 if CURRENCY_HINTS_REGEX.search(text) else 0
        features['is_price_format'] = 1 if PRICE_REGEX.search(text) else 0
        features['has_sold_keyword'] = 1 if SOLD_REGEX.search(text) else 0
        features['has_review_keyword'] = 1 if REVIEW_REGEX.search(text) else 0
        features['has_cta_keyword'] = 1 if CTA_REGEX.search(text) else 0
        features['has_author_keyword'] = 1 if AUTHOR_REGEX.search(text) else 0
        features['has_date_pattern'] = 1 if DATE_REGEX.search(text) else 0
        
        # --- 4. Density & Readability ---
        num_descendants = len(list(element.iterdescendants())) + 1
        features['text_density'] = len(text) / num_descendants
        
        # Link Density (Text inside <a> tags / Total Text)
        # This helps separate navigation/lists from actual article content
        links_text_len = sum(len(a.text_content() or "") for a in element.findall('.//a'))
        features['link_density'] = links_text_len / len(text) if len(text) > 0 else 0.0
        
        try:
            # Check text length to avoid overhead on tiny fragments
            if len(text) > 20:
                features['reading_ease'] = textstat.flesch_reading_ease(text)
            else:
                features['reading_ease'] = 0
        except Exception:
            features['reading_ease'] = 0

        # --- 5. Image & Hyperlink Context ---
        features['has_href'] = 1 if element.get('href') else 0
        features['is_image'] = 1 if tag == 'img' else 0
        
        # Image-specific features
        features['has_src'] = 1 if element.get('src') else 0
        features['has_alt'] = 1 if element.get('alt') else 0
        features['alt_len'] = len(element.get('alt', ''))
        
        # Image surface area (width * height) - useful for distinguishing product images from icons/thumbnails
        try:
            width = element.get('width', '')
            height = element.get('height', '')
            
            # Try to extract numeric values
            if width and height:
                # Remove 'px' suffix if present and convert to int
                width_val = int(re.search(r'\d+', str(width)).group()) if re.search(r'\d+', str(width)) else 0
                height_val = int(re.search(r'\d+', str(height)).group()) if re.search(r'\d+', str(height)) else 0
                features['image_area'] = width_val * height_val
            else:
                features['image_area'] = 0
        except Exception:
            features['image_area'] = 0
        
        # Parent context for images
        features['parent_is_link'] = 1 if (parent is not None and normalize_tag(parent.tag) == 'a') else 0
        
        # Count nearby images
        if parent is not None:
            # Using getattr to safely handle comments or processing instructions in siblings
            sibling_images = sum(1 for sibling in parent if normalize_tag(getattr(sibling, 'tag', '')) == 'img')
            features['sibling_image_count'] = sibling_images
        else:
            features['sibling_image_count'] = 0

        return features

    except Exception as e:
        logger.debug(f"Error extracting features from element: {e}")
        return {}


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


def validate_features(df) -> bool:
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