import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "qq2jsonl.sh"


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq is not installed")
def test_conversion_filters_placeholders_and_emits_expected_schema(tmp_path: Path) -> None:
    payload = {
        "messages": [
            {
                "timestamp": "2026-01-01",
                "sender": {"uin": "1", "name": "甲"},
                "content": {"text": "有效消息"},
            },
            {
                "timestamp": "2026-01-02",
                "sender": {"uin": "2", "name": "乙"},
                "content": {"text": "[图片]"},
            },
        ]
    }
    (tmp_path / "sample.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    result = subprocess.run(
        ["bash", str(SCRIPT), "sample"], cwd=tmp_path, text=True, capture_output=True
    )

    assert result.returncode == 0
    rows = [
        json.loads(line)
        for line in (tmp_path / "sample.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [{"time": "2026-01-01", "uid": "1", "name": "甲", "text": "有效消息"}]


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq is not installed")
def test_failed_conversion_does_not_destroy_existing_jsonl(tmp_path: Path) -> None:
    (tmp_path / "sample.json").write_text("not json", encoding="utf-8")
    output = tmp_path / "sample.jsonl"
    output.write_text("existing private index input\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(SCRIPT), "sample"], cwd=tmp_path, text=True, capture_output=True
    )

    assert result.returncode != 0
    assert output.read_text(encoding="utf-8") == "existing private index input\n"
