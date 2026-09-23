"""
class_mapping.py
==================
The ArASL (grayscale, 32 classes) <-> AASL (color, 31 classes) class-space
mapping, factored out as a single shared source of truth so that both the
offline evaluation script (evaluation/evaluate_real_fusion.py) and the live
inference path (app/practice_session.py's FusionPredictor) remap the two
branches' outputs identically.

AASL has 31 classes; ArASL has 32. The one ArASL class with no AASL
counterpart is "yaa" (ى, alef maksura) -- AASL does not distinguish it from
plain "ya" (ي). The fused/unified class space used everywhere in this
project is AASL's own 31-class space; GrayscaleCNN's "yaa" predictions are
simply outside that space and are remapped to zero probability rather than
guessed into some other class.

This mapping is built here explicitly from each folder name's real Arabic
letter (via ARASL_TO_ARABIC / AASL_NAME_TO_ARASL_KEY below), not by
position, so it is correct regardless of either dataset's on-disk sort
order -- see docs/scientific_paper.docx Section 4.2/4.4 for the
case-sensitivity defect this specifically guards against.
"""

import re

ARASL_TO_ARABIC = {
    'ain': 'ع', 'al': 'ال', 'aleff': 'أ', 'bb': 'ب', 'dal': 'د', 'dha': 'ظ',
    'dhad': 'ض', 'fa': 'ف', 'gaaf': 'ق', 'ghain': 'غ', 'ha': 'ه', 'haa': 'ح',
    'jeem': 'ج', 'kaaf': 'ك', 'khaa': 'خ', 'la': 'لا', 'laam': 'ل', 'meem': 'م',
    'nun': 'ن', 'ra': 'ر', 'saad': 'ص', 'seen': 'س', 'sheen': 'ش', 'ta': 'ت',
    'taa': 'ط', 'thaa': 'ث', 'thal': 'ذ', 'toot': 'ة', 'waw': 'و', 'ya': 'ي',
    'yaa': 'ى', 'zay': 'ز',
}

# AASL folder names carry a numeric suffix (e.g. "Ain_11") that is not part
# of the letter identity. Stripping it and lower-casing gives a stable key
# to map onto the same Arabic letters ArASL uses -- this also makes the
# match immune to the case-sensitivity defect: AASL's own "thal_13" folder
# is lowercase while the other 30 are capitalized, which would otherwise
# corrupt a positional mapping.
AASL_NAME_TO_ARASL_KEY = {
    'ain': 'ain', 'al': 'al', 'alef': 'aleff', 'beh': 'bb', 'dad': 'dhad',
    'dal': 'dal', 'feh': 'fa', 'ghain': 'ghain', 'hah': 'haa', 'heh': 'ha',
    'jeem': 'jeem', 'kaf': 'kaaf', 'khah': 'khaa', 'laa': 'la', 'lam': 'laam',
    'meem': 'meem', 'noon': 'nun', 'qaf': 'gaaf', 'reh': 'ra', 'sad': 'saad',
    'seen': 'seen', 'sheen': 'sheen', 'tah': 'taa', 'teh': 'ta',
    'tehmarbuta': 'toot', 'theh': 'thaa', 'waw': 'waw', 'yeh': 'ya',
    'zah': 'dha', 'zain': 'zay', 'thal': 'thal',
}


def aasl_folder_to_arasl_key(folder_name: str) -> str:
    stripped = re.sub(r'_\d+$', '', folder_name).replace('_', '').lower()
    key = AASL_NAME_TO_ARASL_KEY.get(stripped)
    if key is None:
        raise KeyError(f"Unrecognized AASL folder name '{folder_name}' (stripped: '{stripped}') "
                        f"-- update AASL_NAME_TO_ARASL_KEY if the real folder names differ from "
                        f"the ones recorded in data/aasl_upload_progress.json.")
    return key


def build_class_mappings(aasl_classes, grayscale_classes, verbose: bool = True):
    """Returns (grayscale_class_to_fused_idx, color_class_to_fused_idx), both
    keyed by LOCAL index within each network's own checkpoint, mapping into
    the unified fused space = AASL's own class order (0..30). Identical
    logic to evaluation/evaluate_real_fusion.py's build_class_mappings, kept
    here as the single shared implementation."""
    color_class_to_fused_idx = {i: i for i in range(len(aasl_classes))}

    arasl_key_to_local_idx = {name: i for i, name in enumerate(grayscale_classes)}
    grayscale_class_to_fused_idx = {}
    unmatched = []
    for fused_idx, aasl_name in enumerate(aasl_classes):
        arasl_key = aasl_folder_to_arasl_key(aasl_name)
        local_idx = arasl_key_to_local_idx.get(arasl_key)
        if local_idx is None:
            unmatched.append((aasl_name, arasl_key))
            continue
        grayscale_class_to_fused_idx[local_idx] = fused_idx

    if verbose and unmatched:
        print(f"[info] {len(unmatched)} AASL class(es) had no ArASL counterpart "
              f"(expected exactly one: 'yaa' has no AASL equivalent): {unmatched}")

    return grayscale_class_to_fused_idx, color_class_to_fused_idx
