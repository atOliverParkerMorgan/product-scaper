"""Unified feature definitions and extraction for HTML element analysis."""

import regex as re
import logging
import lxml.html
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

PRICE_REGEX = re.compile(
    fr'(?:(?:{CURRENCY_PATTERN_STR})\s*{NUMBER_PATTERN}|{NUMBER_PATTERN}\s*(?:{CURRENCY_PATTERN_STR}))',
    re.UNICODE | re.IGNORECASE
)

# 2. Semantic Keywords
SOLD_REGEX = re.compile(r'\b(?:' + '|'.join(SOLD_WORD_VARIATIONS) + r')\b', re.IGNORECASE)
REVIEW_REGEX = re.compile(r'\b(?:' + '|'.join(REVIEW_KEYWORDS) + r')\b', re.IGNORECASE)
CTA_REGEX = re.compile(r'\b(?:' + '|'.join(CTA_KEYWORDS) + r')\b', re.IGNORECASE)

# 3. Product Identification Codes
# ISBN-10: 10 digits with optional hyphens (e.g., 0-306-40615-2)
# ISBN-13: 13 digits with optional hyphens (e.g., 978-0-306-40615-7)
ISBN_REGEX = re.compile(
    r'\b(?:ISBN(?:[:-]?1[03])?[:-]?)?(?=[0-9X]{10}$|(?=(?:[0-9]+[-\ ]){3})[-\ 0-9X]{13}$|97[89][0-9]{10}$|(?=(?:[0-9]+[-\ ]){4})[-\ 0-9]{17}$)(?:97[89][-\ ]?)?[0-9]{1,5}[-\ ]?[0-9]+[-\ ]?[0-9]+[-\ ]?[0-9X]\b',
    re.IGNORECASE
)

# UPC (Universal Product Code): 12 digits
# EAN (European Article Number): 13 digits
UPC_EAN_REGEX = re.compile(
    r'\b(?:UPC|EAN|GTIN)[:-]?\s*([0-9]{12,14})\b|\b([0-9]{12,14})\b(?=.*(?:barcode|product code))',
    re.IGNORECASE
)

# ASIN (Amazon Standard Identification Number): B followed by 9 alphanumeric characters
ASIN_REGEX = re.compile(
    r'\b(?:ASIN[:-]?\s*)?B[0-9A-Z]{9}\b',
    re.IGNORECASE
)

# SKU patterns: Common formats like SKU-123456, product-abc123, etc.
SKU_REGEX = re.compile(
    r'\b(?:SKU|Product[\s-]?(?:ID|Code|Number)|Item[\s-]?(?:Number|Code))[:-]?\s*([A-Z0-9-]+)\b',
    re.IGNORECASE
)

# Model Number patterns: Often alphanumeric with hyphens
MODEL_REGEX = re.compile(
    r'\b(?:Model[\s-]?(?:Number|No\.?|#)?)[:-]?\s*([A-Z0-9][A-Z0-9-]{2,20})\b',
    re.IGNORECASE
)

TARGET_FEATURE = 'Category'

# Feature column definitions
NUMERIC_FEATURES = [
    # Structural
    'num_children',
    'num_siblings',
    'dom_depth',
    'is_header',
    'is_formatting',
    'is_list_item',
    
    # Text Metrics
    'text_len',
    'text_word_count',
    'text_digit_count',
    'text_density',
    'link_density',
    'capitalization_ratio',
    'avg_word_length',
    
    # Visual/Style Features
    'font_size',
    'font_weight',
    'is_bold',
    'is_italic',
    'is_hidden',
    'is_block_element',
    
    # Semantic / Regex
    'has_currency_symbol',
    'is_price_format',
    'has_sold_keyword',
    'has_review_keyword',
    'has_cta_keyword',
    
    # Product Identification Codes
    'has_isbn',
    'has_upc_ean',
    'has_asin',
    'has_sku',
    'has_model_number',
    
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
    """
    try:
        # Normalize text
        raw_text = element.text_content()
        text = raw_text.strip()
        
        # Get hierarchy
        parent = element.getparent()
        gparent = parent.getparent() if parent is not None else None
        
        # --- 1. Base Structural Features ---
        features = {
            'Category': category,
            'tag': normalize_tag(element.tag),
            'parent_tag': normalize_tag(parent.tag) if parent is not None else 'root',
            'gparent_tag': normalize_tag(gparent.tag) if gparent is not None else 'root',
            'num_children': len(element),
            'num_siblings': len(parent) - 1 if parent is not None else 0,
            'dom_depth': len(list(element.iterancestors())),
        }
        
        # Specific tag checks
        tag = features['tag']
        features['is_header'] = 1 if tag in {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'} else 0
        features['is_formatting'] = 1 if tag in {'b', 'strong', 'i', 'em', 'u'} else 0
        features['is_list_item'] = 1 if tag in {'li', 'dt', 'dd'} else 0
        features['is_block_element'] = 1 if tag in {'div', 'p', 'section', 'article', 'main', 'aside', 'header', 'footer'} else 0

        # --- 2. Visual/Style Features ---
        style = element.get('style', '')
        
        # Extract font-size (default to 16px if not specified)
        font_size = 16.0
        font_size_match = re.search(r'font-size\s*:\s*(\d+(?:\.\d+)?)(?:px|pt|em|rem)?', style, re.IGNORECASE)
        if font_size_match:
            font_size = float(font_size_match.group(1))
            # Normalize em/rem to approximate px (assuming 1em = 16px)
            if 'em' in style.lower() or 'rem' in style.lower():
                font_size *= 16
        features['font_size'] = font_size
        
        # Extract font-weight (default to 400 = normal)
        font_weight = 400
        font_weight_match = re.search(r'font-weight\s*:\s*(\d+|bold|normal|lighter|bolder)', style, re.IGNORECASE)
        if font_weight_match:
            weight_str = font_weight_match.group(1).lower()
            if weight_str == 'bold' or weight_str == 'bolder':
                font_weight = 700
            elif weight_str == 'normal':
                font_weight = 400
            elif weight_str == 'lighter':
                font_weight = 300
            elif weight_str.isdigit():
                font_weight = int(weight_str)
        features['font_weight'] = font_weight
        
        # Check if element is bold/italic based on tag or style
        features['is_bold'] = 1 if tag in {'b', 'strong'} or font_weight >= 600 or 'font-weight:bold' in style.lower() else 0
        features['is_italic'] = 1 if tag in {'i', 'em'} or 'font-style:italic' in style.lower() else 0
        
        # Check if element is hidden
        features['is_hidden'] = 1 if (
            'display:none' in style.lower() or 
            'display: none' in style.lower() or
            'visibility:hidden' in style.lower() or
            'visibility: hidden' in style.lower()
        ) else 0
        
        # --- 3. Attribute Semantic Features ---
        class_str = " ".join(element.get('class', '').split()) if element.get('class') else ""
        features['class_str'] = class_str
        features['id_str'] = element.get('id', '')
        
        # Heuristics on class names (common in frameworks like Bootstrap/Tailwind)
        class_lower = class_str.lower()
        id_lower = features['id_str'].lower()
        class_id_combined = class_lower + ' ' + id_lower
        
        features['class_indicates_price'] = 1 if any(x in class_id_combined for x in ['price', 'cost', 'amount', 'currency']) else 0
        features['class_indicates_title'] = 1 if any(x in class_id_combined for x in ['title', 'header', 'heading', 'name', 'product-name']) else 0
        features['class_indicates_description'] = 1 if any(x in class_id_combined for x in ['description', 'desc', 'detail', 'summary']) else 0
        features['class_indicates_image'] = 1 if any(x in class_id_combined for x in ['image', 'img', 'photo', 'picture', 'thumbnail']) else 0

        # --- 4. Text Content Features ---
        features['text_len'] = len(text)
        words = text.split()
        features['text_word_count'] = len(words)
        features['text_digit_count'] = sum(c.isdigit() for c in text)
        
        # Average word length (useful for distinguishing descriptive text from codes/IDs)
        features['avg_word_length'] = sum(len(word) for word in words) / len(words) if words else 0.0
        
        # Capitalization (Useful for Titles vs Descriptions)
        features['capitalization_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0.0

        # Regex Matches
        features['has_currency_symbol'] = 1 if CURRENCY_HINTS_REGEX.search(text) else 0
        features['is_price_format'] = 1 if PRICE_REGEX.search(text) else 0
        features['has_sold_keyword'] = 1 if SOLD_REGEX.search(text) else 0
        features['has_review_keyword'] = 1 if REVIEW_REGEX.search(text) else 0
        features['has_cta_keyword'] = 1 if CTA_REGEX.search(text) else 0
        
        # Product Code Detection
        features['has_isbn'] = 1 if ISBN_REGEX.search(text) else 0
        features['has_upc_ean'] = 1 if UPC_EAN_REGEX.search(text) else 0
        features['has_asin'] = 1 if ASIN_REGEX.search(text) else 0
        features['has_sku'] = 1 if SKU_REGEX.search(text) else 0
        features['has_model_number'] = 1 if MODEL_REGEX.search(text) else 0
        
        # --- 5. Density Metrics ---
        num_descendants = len(list(element.iterdescendants())) + 1
        features['text_density'] = len(text) / num_descendants
        
        # Link Density (Text inside <a> tags / Total Text)
        # This helps separate navigation/lists from actual article content
        links_text_len = sum(len(a.text_content() or "") for a in element.findall('.//a'))
        features['link_density'] = links_text_len / len(text) if len(text) > 0 else 0.0

        # --- 6. Image & Hyperlink Context ---
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
            
            # If no width/height attributes, try to extract from style attribute
            if not width or not height:
                style = element.get('style', '')
                if style:
                    # Look for width in style (e.g., "width:45px" or "width: 45px")
                    width_match = re.search(r'width\s*:\s*(\d+)(?:px)?', style, re.IGNORECASE)
                    height_match = re.search(r'height\s*:\s*(\d+)(?:px)?', style, re.IGNORECASE)
                    if width_match and not width:
                        width = width_match.group(1)
                    if height_match and not height:
                        height = height_match.group(1)
            
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