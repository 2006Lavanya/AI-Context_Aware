"""
entity_extractor.py
---------------------
Extracts entities from user text using a combination of:
1. spaCy's built-in NER (for PERSON, ORG, GPE/LOCATION, DATE)
2. Custom regex + keyword-based rules for domain-specific entities that
   spaCy does not know about out of the box: PRODUCT, BRAND, BUDGET,
   PRICE, ORDER_ID, QUANTITY, COLOR, SIZE, CATEGORY.

This hybrid approach is what makes the extraction genuinely useful for
an e-commerce / conversational-assistant style project, since spaCy's
default model has no idea what a "laptop brand" or "budget" is.
"""

import re
import spacy

# ---------------------------------------------------------------------
# Load spaCy model. If the small English model isn't installed, fall
# back to a blank English pipeline so the app doesn't crash - the
# custom regex/keyword rules below still work in that case, only the
# generic PERSON/ORG/GPE/DATE detection from spaCy will be skipped.
# ---------------------------------------------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except OSError:
    nlp = spacy.blank("en")
    SPACY_AVAILABLE = False
    print("[entity_extractor] Warning: 'en_core_web_sm' not found. "
          "Run: python -m spacy download en_core_web_sm "
          "Falling back to custom rules only.")

# ---------------------------------------------------------------------
# Custom domain knowledge (extend freely as the catalog grows)
# ---------------------------------------------------------------------
PRODUCT_KEYWORDS = [
    "laptop", "computer", "notebook", "smartphone", "phone", "mobile",
    "camera", "headphones", "earphones", "tablet", "smartwatch", "watch",
    "printer", "shoes", "shirt", "t-shirt", "washing machine", "refrigerator",
    "fridge", "television", "tv", "monitor", "keyboard", "mouse", "speaker",
    "charger", "router", "microwave", "air conditioner", "ac",
]

BRAND_KEYWORDS = [
    "hp", "dell", "lenovo", "asus", "acer", "apple", "samsung", "sony",
    "xiaomi", "mi", "oneplus", "realme", "vivo", "oppo", "nokia", "boat",
    "jbl", "bose", "nike", "adidas", "puma", "reebok", "bosch", "lg",
    "whirlpool", "godrej", "philips", "microsoft", "google", "canon", "nikon",
]

COLOR_KEYWORDS = [
    "red", "blue", "green", "black", "white", "yellow", "orange", "purple",
    "pink", "grey", "gray", "silver", "gold", "brown", "maroon", "navy",
]

SIZE_KEYWORDS = [
    "small", "medium", "large", "extra large", "xl", "xxl", "s", "m", "l",
]

CATEGORY_KEYWORDS = {
    "laptop": "Electronics", "computer": "Electronics", "phone": "Electronics",
    "smartphone": "Electronics", "camera": "Electronics", "tablet": "Electronics",
    "headphones": "Electronics", "television": "Electronics", "tv": "Electronics",
    "shirt": "Fashion", "shoes": "Fashion", "t-shirt": "Fashion",
    "washing machine": "Home Appliances", "refrigerator": "Home Appliances",
    "fridge": "Home Appliances", "microwave": "Home Appliances",
    "air conditioner": "Home Appliances", "ac": "Home Appliances",
}

# Short unit/abbreviation tokens that spaCy's statistical NER sometimes
# mislabels as GPE/ORG/CARDINAL when they appear standalone (e.g. "GB"
# tagged as a country code, "RAM" tagged as an organization). These are
# suppressed from the generic spaCy pass since they add noise rather
# than value - the custom rules above already handle them correctly
# (e.g. "16GB" -> RAM entity).
NOISY_SPACY_TOKENS = {
    "gb", "mb", "tb", "kg", "cm", "mm", "hz", "ghz", "ml", "kmph", "kmh",
    "ram", "rom", "usb", "hd", "hdmi", "led", "lcd", "oled",
}

RAM_PATTERN = re.compile(r"\b(\d{1,3})\s?gb\s?(ram)?\b", re.IGNORECASE)
ORDER_ID_PATTERN = re.compile(r"\b(?:ORD|ORDER)[-\s]?(\d{3,10})\b", re.IGNORECASE)
QUANTITY_PATTERN = re.compile(r"\b(\d{1,3})\s?(pcs|pieces|units|qty|quantity)\b", re.IGNORECASE)

# Budget / price patterns: "under 60000", "budget is 60000", "60000 rupees",
# "around 20k", "₹15000", "50k", etc.
BUDGET_CONTEXT_PATTERN = re.compile(
    r"(?:under|below|within|budget(?:\s+is)?|around|up\s*to|less\s*than|not\s*more\s*than)\s*"
    r"[₹]?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(k)?",
    re.IGNORECASE,
)
PRICE_CONTEXT_PATTERN = re.compile(
    r"(?:price|cost|rate)[^\d]{0,15}[₹]?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(k)?",
    re.IGNORECASE,
)
GENERIC_MONEY_PATTERN = re.compile(
    r"[₹]\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(k)?|\b(\d+(?:,\d{3})*)\s*(k)?\s*(?:rupees|rs\.?)\b",
    re.IGNORECASE,
)
BARE_NUMBER_PATTERN = re.compile(r"^\s*[₹]?\s*(\d{3,7})\s*(k)?\s*(?:rupees|rs\.?)?\s*$", re.IGNORECASE)


def _normalize_amount(value_str: str, k_suffix: str) -> int:
    value = float(value_str.replace(",", ""))
    if k_suffix:
        value *= 1000
    return int(value)


def extract_custom_entities(text: str):
    """Rule-based / keyword-based extraction for domain-specific entities."""
    entities = []
    lower_text = text.lower()

    # PRODUCT
    for product in PRODUCT_KEYWORDS:
        if re.search(rf"\b{re.escape(product)}\b", lower_text):
            entities.append({"type": "PRODUCT", "value": product})
            break  # one primary product per message is usually enough

    # BRAND
    for brand in BRAND_KEYWORDS:
        if re.search(rf"\b{re.escape(brand)}\b", lower_text):
            entities.append({"type": "BRAND", "value": brand.upper() if len(brand) <= 3 else brand.capitalize()})
            break

    # CATEGORY (derived from product)
    for keyword, category in CATEGORY_KEYWORDS.items():
        if re.search(rf"\b{re.escape(keyword)}\b", lower_text):
            entities.append({"type": "CATEGORY", "value": category})
            break

    # COLOR
    for color in COLOR_KEYWORDS:
        if re.search(rf"\b{re.escape(color)}\b", lower_text):
            entities.append({"type": "COLOR", "value": color})
            break

    # SIZE
    for size in SIZE_KEYWORDS:
        if re.search(rf"\b{re.escape(size)}\b", lower_text):
            entities.append({"type": "SIZE", "value": size.upper() if len(size) <= 3 else size})
            break

    # RAM (a common "product detail" entity for laptops/phones)
    ram_match = RAM_PATTERN.search(lower_text)
    if ram_match:
        entities.append({"type": "RAM", "value": f"{ram_match.group(1)}GB"})

    # ORDER_ID
    order_match = ORDER_ID_PATTERN.search(text)
    if order_match:
        entities.append({"type": "ORDER_ID", "value": f"ORD{order_match.group(1)}"})

    # QUANTITY
    qty_match = QUANTITY_PATTERN.search(lower_text)
    if qty_match:
        entities.append({"type": "QUANTITY", "value": qty_match.group(1)})

    # BUDGET (explicit budget/under/around context)
    budget_match = BUDGET_CONTEXT_PATTERN.search(lower_text)
    if budget_match:
        amount = _normalize_amount(budget_match.group(1), budget_match.group(2))
        entities.append({"type": "BUDGET", "value": amount})
    else:
        # PRICE inquiry context
        price_match = PRICE_CONTEXT_PATTERN.search(lower_text)
        if price_match:
            amount = _normalize_amount(price_match.group(1), price_match.group(2))
            entities.append({"type": "PRICE", "value": amount})
        else:
            money_match = GENERIC_MONEY_PATTERN.search(lower_text)
            if money_match:
                value_str = money_match.group(1) or money_match.group(3)
                k_suffix = money_match.group(2) or money_match.group(4)
                if value_str:
                    amount = _normalize_amount(value_str, k_suffix)
                    entities.append({"type": "BUDGET", "value": amount})
            else:
                # A lone number message (e.g. user just replies "60000")
                # is very likely answering a budget question.
                bare_match = BARE_NUMBER_PATTERN.match(text)
                if bare_match:
                    amount = _normalize_amount(bare_match.group(1), bare_match.group(2))
                    entities.append({"type": "BUDGET", "value": amount})

    return entities


def extract_spacy_entities(text: str):
    """Use spaCy's statistical NER model for general-purpose entities."""
    entities = []
    if not SPACY_AVAILABLE:
        return entities

    doc = nlp(text)
    label_map = {
        "PERSON": "PERSON",
        "ORG": "ORGANIZATION",
        "GPE": "LOCATION",
        "LOC": "LOCATION",
        "DATE": "DATE",
        "TIME": "DATE",
        "MONEY": "PRICE",
        "CARDINAL": "QUANTITY",
        "PRODUCT": "PRODUCT",
    }
    for ent in doc.ents:
        mapped_type = label_map.get(ent.label_, None)
        if mapped_type and ent.text.strip().lower() not in NOISY_SPACY_TOKENS:
            entities.append({"type": mapped_type, "value": ent.text})
    return entities


def extract_entities(text: str):
    """
    Combines spaCy NER with custom rule-based extraction, de-duplicating
    entities of the same type so a single dominant value wins for
    domain-specific types (PRODUCT, BRAND, BUDGET, etc.), while allowing
    multiple generic entities (e.g. two PERSON mentions).
    """
    custom_entities = extract_custom_entities(text)
    spacy_entities = extract_spacy_entities(text)

    # Domain-specific types should not be duplicated by a generic spaCy hit
    custom_types = {e["type"] for e in custom_entities}
    custom_numeric_values = {
        str(e["value"]) for e in custom_entities
        if e["type"] in ("BUDGET", "PRICE", "QUANTITY", "RAM")
    }

    filtered_spacy = []
    for e in spacy_entities:
        if e["type"] in custom_types:
            continue
        # Avoid spaCy mislabeling a bare number already captured as
        # BUDGET/PRICE/QUANTITY (e.g. "60000" -> DATE) as a separate entity.
        raw_digits = re.sub(r"[^\d]", "", str(e["value"]))
        if e["type"] in ("DATE", "QUANTITY") and raw_digits and raw_digits in custom_numeric_values:
            continue
        filtered_spacy.append(e)

    combined = custom_entities + filtered_spacy

    # De-duplicate identical (type, value) pairs
    seen = set()
    unique_entities = []
    for e in combined:
        key = (e["type"], str(e["value"]).lower())
        if key not in seen:
            seen.add(key)
            unique_entities.append(e)

    return unique_entities
