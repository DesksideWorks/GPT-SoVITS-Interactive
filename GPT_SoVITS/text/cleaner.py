from text import cleaned_text_to_sequence
import os

from text import symbols as symbols_v1
from text import symbols2 as symbols_v2


def clean_text(text, language="ja", version=None):
    if version is None:
        version = os.environ.get("version", "v2")
    if version == "v1":
        symbols = symbols_v1.symbols
        language_module_map = {"zh": "chinese2", "ja": "japanese", "en": "english"}
    else:
        symbols = symbols_v2.symbols
        language_module_map = {"zh": "chinese2", "ja": "japanese", "en": "english"}

    if language not in language_module_map:
        raise ValueError(f"Unsupported TTS language: {language}")
    language_module = __import__("text." + language_module_map[language], fromlist=[language_module_map[language]])
    if hasattr(language_module, "text_normalize"):
        norm_text = language_module.text_normalize(text)
    else:
        norm_text = text
    if language == "zh":
        phones, word2ph = language_module.g2p(norm_text)
        assert len(phones) == sum(word2ph)
        assert len(norm_text) == len(word2ph)
    elif language == "en":
        phones = language_module.g2p(norm_text)
        if len(phones) < 4:
            phones = [","] + phones
        word2ph = None
    else:
        phones = language_module.g2p(norm_text)
        word2ph = None
    phones = ["UNK" if ph not in symbols else ph for ph in phones]
    return phones, word2ph, norm_text

def text_to_sequence(text, language, version=None):
    version = os.environ.get("version", version)
    if version is None:
        version = "v2"
    phones = clean_text(text)
    return cleaned_text_to_sequence(phones, version)


if __name__ == "__main__":
    print(clean_text("こんにちは。", "ja"))
