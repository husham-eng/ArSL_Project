"""
arasl_letter_map.py
Maps ArASL folder/class names (as produced by ImageFolder, e.g. 'ain', 'bb',
'thal') to the actual Arabic letter/ligature each one represents. Used
wherever a model's predicted class index needs to become real, spoken/typed
Arabic text (the practice app, spelling correction input, TTS output).

This is genuine Arabic-language data (the actual letters), not documentation,
and is intentionally kept in Arabic.
"""

ARASL_TO_ARABIC = {
    'ain': 'ع', 'al': 'ال', 'aleff': 'أ', 'bb': 'ب', 'dal': 'د', 'dha': 'ظ',
    'dhad': 'ض', 'fa': 'ف', 'gaaf': 'ق', 'ghain': 'غ', 'ha': 'ه', 'haa': 'ح',
    'jeem': 'ج', 'kaaf': 'ك', 'khaa': 'خ', 'la': 'لا', 'laam': 'ل', 'meem': 'م',
    'nun': 'ن', 'ra': 'ر', 'saad': 'ص', 'seen': 'س', 'sheen': 'ش', 'ta': 'ت',
    'taa': 'ط', 'thaa': 'ث', 'thal': 'ذ', 'toot': 'ة', 'waw': 'و', 'ya': 'ي',
    'yaa': 'ى', 'zay': 'ز',
}
