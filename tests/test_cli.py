import csv
from pathlib import Path

from mafnd.cli import main


def test_batch_cli_writes_predictions():
    test_output = Path("outputs/test_cli")
    test_output.mkdir(parents=True, exist_ok=True)
    source = test_output / "input.csv"
    output = test_output / "output.csv"
    source.write_text(
        "id,title,content,label\n"
        '1,Report,"Official audited report was published",real\n'
        '2,SHOCKING,"Secret miracle exposed! Everyone is brainwashed",fake\n',
        encoding="utf-8",
    )
    assert main(["--input", str(source), "--output", str(output)]) == 0
    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["prediction"] for row in rows] == ["real", "fake"]
