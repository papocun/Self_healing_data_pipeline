"""Unit tests for multi-format DataReader and resumable BatchLoader."""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.ingestion.batch_loader import BatchLoader
from src.ingestion.data_reader import (
    DataReader,
    DataReaderError,
    EmptyFileError,
    FileSizeLimitExceededError,
    UnsupportedFileFormatError,
)


@pytest.fixture
def reader() -> DataReader:
    return DataReader()


def test_read_csv_with_bom_and_whitespace_headers(tmp_path: Path, reader: DataReader):
    csv_file = tmp_path / "reviews_bom.csv"
    # Prepend UTF-8 BOM (\ufeff) and padded column headers
    csv_file.write_bytes(
        "\ufeff review_id , text , rating \nrev_1,Great product,5\nrev_2,Terrible,1\n".encode("utf-8")
    )

    df = reader.read_file(csv_file, add_source_column=True)
    assert len(df) == 2
    assert list(df.columns) == ["review_id", "text", "rating", "_source_file"]
    assert df.iloc[0]["review_id"] == "rev_1"


def test_read_csv_latin1_fallback(tmp_path: Path, reader: DataReader):
    csv_file = tmp_path / "latin1_reviews.csv"
    csv_file.write_bytes("review_id,text\nr1,Café résumé naïve\n".encode("latin1"))

    df = reader.read_file(csv_file)
    assert len(df) == 1
    assert "Café" in df.iloc[0]["text"]


def test_read_json_variants(tmp_path: Path, reader: DataReader):
    # 1. List of records
    f_list = tmp_path / "list.json"
    f_list.write_text(json.dumps([{"id": "1", "val": "a"}, {"id": "2", "val": "b"}]), encoding="utf-8")
    assert len(reader.read_file(f_list)) == 2

    # 2. Envelope dict
    f_env = tmp_path / "envelope.json"
    f_env.write_text(json.dumps({"reviews": [{"id": "10"}, {"id": "11"}]}), encoding="utf-8")
    assert len(reader.read_file(f_env)) == 2

    # 3. Columnar dict
    f_col = tmp_path / "columnar.json"
    f_col.write_text(json.dumps({"id": ["100", "200"], "score": [4, 5]}), encoding="utf-8")
    df_col = reader.read_file(f_col)
    assert len(df_col) == 2
    assert list(df_col.columns) == ["id", "score"]


def test_read_jsonl(tmp_path: Path, reader: DataReader):
    jsonl_file = tmp_path / "reviews.jsonl"
    lines = [
        json.dumps({"review_id": "jl_1", "text": "Good"}),
        json.dumps({"review_id": "jl_2", "text": "Mediocre"}),
    ]
    jsonl_file.write_text("\n".join(lines), encoding="utf-8")

    df = reader.read_file(jsonl_file)
    assert len(df) == 2
    assert df.iloc[1]["review_id"] == "jl_2"


@pytest.mark.parametrize(
    "filename,content",
    [
        ("zero_bytes.csv", ""),
        ("whitespace_only.csv", "   \n\t  \n"),
        ("headers_only.csv", "review_id,text,rating\n"),
        ("blank_rows.csv", "review_id,text,rating\n,,\n,,\n"),
        ("empty_list.json", "[]"),
        ("empty_obj.json", "{}"),
    ],
)
def test_empty_or_header_only_files_raise_empty_file_error(
    tmp_path: Path, reader: DataReader, filename: str, content: str
):
    target = tmp_path / filename
    target.write_text(content, encoding="utf-8")

    with pytest.raises(EmptyFileError):
        reader.read_file(target)


def test_file_size_limit_exceeded(tmp_path: Path):
    strict_reader = DataReader(max_file_size_mb=1)
    strict_reader.max_file_size_bytes = 64  # Set 64-byte cap for test

    large_file = tmp_path / "oversized.csv"
    large_file.write_text("id,text\n" + ("1,some_long_review_payload\n" * 10), encoding="utf-8")

    with pytest.raises(FileSizeLimitExceededError):
        strict_reader.read_file(large_file)


def test_read_missing_and_unsupported_files(tmp_path: Path, reader: DataReader):
    with pytest.raises(FileNotFoundError):
        reader.read_file(tmp_path / "nonexistent.csv")

    bad_ext = tmp_path / "notes.txt"
    bad_ext.write_text("hello", encoding="utf-8")
    with pytest.raises(UnsupportedFileFormatError):
        reader.read_file(bad_ext)


def test_read_directory_with_quarantine(tmp_path: Path, reader: DataReader):
    raw_dir = tmp_path / "raw"
    failed_dir = tmp_path / "failed"
    raw_dir.mkdir()

    valid_csv = raw_dir / "valid.csv"
    valid_csv.write_text("id,val\n1,ok\n", encoding="utf-8")

    corrupt_json = raw_dir / "corrupt.json"
    corrupt_json.write_text("{malformed_json: [", encoding="utf-8")

    results = reader.read_directory(
        directory=raw_dir,
        quarantine_invalid=True,
        quarantine_dir=failed_dir,
    )

    assert len(results) == 1
    assert results[0][0].name == "valid.csv"
    assert not corrupt_json.exists()
    quarantined_files = list(failed_dir.glob("corrupt_*.json"))
    assert len(quarantined_files) == 1


def test_batch_loader_validation_and_resume():
    with pytest.raises(ValueError):
        BatchLoader(batch_size=0)

    with pytest.raises(ValueError):
        BatchLoader(batch_size=-5)

    df = pd.DataFrame({"id": list(range(10))})
    loader = BatchLoader(batch_size=4)

    all_batches = list(loader.get_batches_from_dataframe(df))
    assert len(all_batches) == 3
    assert (all_batches[0].start_row, all_batches[0].end_row) == (0, 4)
    assert (all_batches[2].start_row, all_batches[2].end_row) == (8, 10)

    # Resume from batch 2 (simulating recovery after batch 1 succeeded)
    resumed = list(loader.get_batches_from_dataframe(df, start_batch=2))
    assert len(resumed) == 2
    assert resumed[0].batch_id == 2
    assert resumed[0].start_row == 4


def test_batch_loader_from_file(tmp_path: Path):
    test_file = tmp_path / "data.csv"
    test_file.write_text("col1\n1\n2\n3\n4\n5\n", encoding="utf-8")

    loader = BatchLoader(batch_size=2)
    batches = list(loader.get_batches_from_file(test_file))
    assert len(batches) == 3
    assert batches[0].source_file == "data.csv"



def test_detect_format(tmp_path: Path, reader: DataReader):
    # 1. JSONL by extension
    f1 = tmp_path / "test.jsonl"
    f1.write_text('{"a": 1}\n', encoding="utf-8")
    assert reader.detect_format(f1) == "jsonl"

    f2 = tmp_path / "test.ndjson"
    f2.write_text('{"a": 1}\n', encoding="utf-8")
    assert reader.detect_format(f2) == "jsonl"

    # 2. Standard JSON array
    f3 = tmp_path / "array.json"
    f3.write_text('[{"a": 1}, {"a": 2}]', encoding="utf-8")
    assert reader.detect_format(f3) == "json"

    # 3. Yelp-style JSONL with .json extension
    f4 = tmp_path / "yelp_style.json"
    f4.write_text('{"business_id": "b1", "name": "Store 1"}\n{"business_id": "b2", "name": "Store 2"}\n', encoding="utf-8")
    assert reader.detect_format(f4) == "jsonl"


def test_read_yelp_business_jsonl_multiline(tmp_path: Path, reader: DataReader):
    # Simulates yelp_academic_dataset_business.json (newline-delimited JSON with .json extension)
    business_file = tmp_path / "yelp_academic_dataset_business.json"
    lines = [
        json.dumps({"business_id": "b1", "name": "Bakery", "stars": 5.0}),
        json.dumps({"business_id": "b2", "name": "Bistro", "stars": 3.5}),
        json.dumps({"business_id": "b3", "name": "Cafe", "stars": 4.0}),
    ]
    business_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    df = reader.read_file(business_file)
    assert len(df) == 3
    assert list(df["business_id"]) == ["b1", "b2", "b3"]
    assert df.iloc[0]["name"] == "Bakery"


def test_streaming_jsonl_batches(tmp_path: Path):
    jsonl_file = tmp_path / "reviews.json"
    lines = [json.dumps({"id": i, "text": f"review {i}"}) for i in range(10)]
    jsonl_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    loader = BatchLoader(batch_size=3)
    batches = list(loader.get_batches_from_file(jsonl_file))

    assert len(batches) == 4
    assert batches[0].record_count == 3
    assert (batches[0].start_row, batches[0].end_row) == (0, 3)
    assert batches[1].record_count == 3
    assert (batches[1].start_row, batches[1].end_row) == (3, 6)
    assert batches[2].record_count == 3
    assert (batches[2].start_row, batches[2].end_row) == (6, 9)
    assert batches[3].record_count == 1
    assert (batches[3].start_row, batches[3].end_row) == (9, 10)


def test_streaming_jsonl_resume_from_checkpoint(tmp_path: Path):
    jsonl_file = tmp_path / "stream_resume.jsonl"
    lines = [json.dumps({"id": i}) for i in range(10)]
    jsonl_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    loader = BatchLoader(batch_size=3)
    # Resume from batch 3 (batches 1 & 2 skipped)
    batches = list(loader.get_batches_from_file(jsonl_file, start_batch=3))

    assert len(batches) == 2
    assert batches[0].batch_id == 3
    assert (batches[0].start_row, batches[0].end_row) == (6, 9)
    assert batches[0].data.iloc[0]["id"] == 6

    assert batches[1].batch_id == 4
    assert (batches[1].start_row, batches[1].end_row) == (9, 10)
    assert batches[1].data.iloc[0]["id"] == 9


def test_streaming_jsonl_with_malformed_lines(tmp_path: Path):
    jsonl_file = tmp_path / "with_bad_lines.json"
    content = (
        '{"id": 1, "text": "Valid 1"}\n'
        'CORRUPT_NOT_JSON\n'
        '{"id": 2, "text": "Valid 2"}\n'
        '{truncated_json\n'
        '{"id": 3, "text": "Valid 3"}\n'
    )
    jsonl_file.write_text(content, encoding="utf-8")

    loader = BatchLoader(batch_size=2)
    batches = list(loader.get_batches_from_file(jsonl_file))

    # 3 valid records partitioned into batch of 2 and batch of 1
    assert len(batches) == 2
    assert batches[0].record_count == 2
    assert batches[1].record_count == 1
    all_ids = list(batches[0].data["id"]) + list(batches[1].data["id"])
    assert all_ids == [1, 2, 3]


def test_list_files_comma_separated_patterns(tmp_path: Path, reader: DataReader):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    (raw_dir / "yelp_academic_dataset_business.json").write_text('{"b": 1}\n', encoding="utf-8")
    (raw_dir / "yelp_academic_dataset_review.json").write_text('{"r": 1}\n', encoding="utf-8")
    (raw_dir / "yelp_sample.json").write_text('[{"s": 1}]', encoding="utf-8")
    (raw_dir / "yelp_academic_dataset_user.json").write_text('{"u": 1}\n', encoding="utf-8")

    # Match reviews and sample, exclude business and user
    matched = reader.list_files(directory=raw_dir, pattern="*review*.*, *sample*.*")
    matched_names = [f.name for f in matched]

    assert "yelp_academic_dataset_review.json" in matched_names
    assert "yelp_sample.json" in matched_names
    assert "yelp_academic_dataset_business.json" not in matched_names
    assert "yelp_academic_dataset_user.json" not in matched_names
