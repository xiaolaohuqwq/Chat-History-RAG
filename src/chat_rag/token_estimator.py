from __future__ import annotations

import re

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def estimate_tokens(text: str) -> int:
    """Conservatively estimate tokens without loading a model tokenizer."""
    cjk_count = len(_CJK.findall(text))
    non_cjk = _CJK.sub("", text)
    # Latin text commonly averages near four characters per token. Three keeps headroom.
    other_count = (len(non_cjk.encode("utf-8")) + 2) // 3
    return max(1, cjk_count + other_count)
