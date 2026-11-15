"""콘솔에서 패턴을 비교하는 Mini NPU 시뮬레이터."""

import json
import math
import re
from collections import Counter
from pathlib import Path
from time import perf_counter_ns


DATA_PATH = Path(__file__).resolve().with_name("data.json")
EPSILON = 1e-9
REPETITIONS = 10
JSON_SIZES = (5, 13, 25)
STANDARD_LABELS = ("Cross", "X")
FILTER_LABELS = {"cross": "Cross", "x": "X"}
EXPECTED_LABELS = {"+": "Cross", "x": "X", "Cross": "Cross", "X": "X"}


def normalize_label(value, aliases, field):
    """원본 라벨을 검증하고 입력 종류에 맞는 표준 라벨을 반환한다."""
    if not isinstance(value, str) or value not in aliases:
        allowed = ", ".join(repr(alias) for alias in aliases)
        raise ValueError(f"{field}에는 {allowed} 중 하나가 필요합니다.")
    return aliases[value]


def mac(pattern, kernel):
    """크기를 검증한 두 행렬의 같은 위치를 곱하고 순서대로 누적한다."""
    score = 0.0
    for pattern_row, kernel_row in zip(pattern, kernel):
        for value, weight in zip(pattern_row, kernel_row):
            score += value * weight
    return score


def flatten_matrix(matrix):
    """행 순서대로 이어 붙여 길이 N²인 리스트를 만든다."""
    return [value for row in matrix for value in row]


def mac_flat(pattern, kernel):
    """평탄화된 두 행렬을 같은 순서로 한 번씩 순회한다."""
    score = 0.0
    for value, weight in zip(pattern, kernel):
        score += value * weight
    return score


def classify(score_a, score_b, label_a, label_b):
    if not (math.isfinite(score_a) and math.isfinite(score_b)):
        raise ValueError("MAC 점수가 유한한 실수 범위를 벗어났습니다.")
    if abs(score_a - score_b) < EPSILON:
        return "UNDECIDED"
    return label_a if score_a > score_b else label_b


def read_matrix(name):
    print(f"\n{name} (3줄, 공백 구분)")
    matrix = []
    while len(matrix) < 3:
        entries = input(f"  {len(matrix) + 1}행 입력: ").split()
        try:
            row = [float(entry) for entry in entries]
            if len(row) != 3 or not all(math.isfinite(value) for value in row):
                raise ValueError
        except (ValueError, OverflowError):
            print("입력 형식 오류: 이 행의 유한한 숫자 3개를 공백으로 구분해 다시 입력하세요.")
            continue
        matrix.append(row)
    return matrix


def validate_matrix(matrix, size, name):
    if not isinstance(matrix, list) or len(matrix) != size:
        raise ValueError(f"{name}은(는) {size}개 행이 필요합니다.")

    validated = []
    for index, row in enumerate(matrix, start=1):
        if not isinstance(row, list) or len(row) != size:
            raise ValueError(f"{name}의 {index}행에 숫자 {size}개가 필요합니다.")

        numeric_row = []
        for column, value in enumerate(row, start=1):
            try:
                if type(value) not in (int, float):
                    raise ValueError
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError
            except (ValueError, OverflowError):
                raise ValueError(f"{name}의 {index}행 {column}열에 유한한 숫자가 필요합니다.") from None
            numeric_row.append(number)
        validated.append(numeric_row)
    return validated


def load_filter_groups(raw_filters):
    groups = {}
    errors = {}
    if not isinstance(raw_filters, dict):
        print("필터 스키마 오류: filters는 JSON 객체여야 합니다.")
        return groups, errors

    print("\n[필터 로드]")
    for key, raw_group in raw_filters.items():
        try:
            match = re.fullmatch(r"size_([1-9][0-9]*)", key)
            if match is None:
                raise ValueError("필터 키는 size_{N} 형식이어야 합니다.")
            size = int(match[1])
            if not isinstance(raw_group, dict):
                raise ValueError("필터 그룹은 JSON 객체여야 합니다.")
            group = {}
            for field, values in raw_group.items():
                if field not in FILTER_LABELS:
                    continue
                label = normalize_label(field, FILTER_LABELS, "필터 키")
                if label in group:
                    raise ValueError(f"{label} 필터가 중복되었습니다.")
                matrix = validate_matrix(values, size, f"{label} 필터")
                group[label] = flatten_matrix(matrix)
            for label in STANDARD_LABELS:
                if label not in group:
                    raise ValueError(f"{label} 필터가 없습니다.")
        except ValueError as error:
            errors[key] = str(error)
            print(f"{key}: 필터 오류 ({error})")
        else:
            groups[key] = group
            print(f"{key}: 필터 로드 완료 (Cross, X)")
    return groups, errors


def average_mac_ms(operation, pattern, kernel):
    operation(pattern, kernel)  # 워밍업 1회는 측정에서 제외한다.
    elapsed_ns = 0
    for _ in range(REPETITIONS):
        start = perf_counter_ns()
        operation(pattern, kernel)
        elapsed_ns += perf_counter_ns() - start
    return elapsed_ns / REPETITIONS / 1_000_000


def print_benchmark(samples):
    print(f"\n[성능 분석, 방식별 MAC 1회 평균, 각 {REPETITIONS}회 측정]")
    print("크기\t2차원 평균(ms)\t1차원 평균(ms)\t연산 횟수(N²)")
    for size, pattern, kernel in samples:
        flat_pattern = flatten_matrix(pattern)
        flat_kernel = flatten_matrix(kernel)
        matrix_ms = average_mac_ms(mac, pattern, kernel)
        flat_ms = average_mac_ms(mac_flat, flat_pattern, flat_kernel)
        print(f"{size}×{size}\t{matrix_ms:.6f}\t{flat_ms:.6f}\t{size * size}")


def benchmark_sample(size):
    center = size // 2
    cross = [
        [float(row == center or col == center) for col in range(size)]
        for row in range(size)
    ]
    return size, cross, cross


def run_manual():
    filter_a = read_matrix("필터 A")
    filter_b = read_matrix("필터 B")
    print("\n필터 A, B 저장 완료")
    pattern = read_matrix("패턴")

    flat_a = flatten_matrix(filter_a)
    flat_b = flatten_matrix(filter_b)
    flat_pattern = flatten_matrix(pattern)
    start = perf_counter_ns()
    score_a = mac_flat(flat_pattern, flat_a)
    score_b = mac_flat(flat_pattern, flat_b)
    elapsed_ms = (perf_counter_ns() - start) / 1_000_000

    print("\n[MAC 결과]")
    print(f"A 점수: {score_a}")
    print(f"B 점수: {score_b}")
    print(f"연산 시간(A·B 합계): {elapsed_ms:.6f} ms")
    try:
        result = classify(score_a, score_b, "A", "B")
    except ValueError as error:
        print(f"판정: 판정 불가 ({error})")
    else:
        if result == "UNDECIDED":
            print(f"판정: 판정 불가 (UNDECIDED, |A-B| < {EPSILON:g})")
        else:
            print(f"판정: {result}")

    print_benchmark([(3, pattern, filter_a)])


def analyze_case(case_id, case, filter_groups, filter_errors):
    """케이스를 검증하고 분석한다. 실패하면 원인 유형과 상세 사유를 반환한다."""
    print(f"\n[{case_id}]")
    try:
        match = re.fullmatch(r"(size_([1-9][0-9]*))_[0-9]+", case_id)
        if match is None:
            raise ValueError("패턴 키는 size_{N}_{idx} 형식이어야 합니다.")
        filter_key, size_text = match.groups()
        if filter_key in filter_errors:
            raise ValueError(f"{filter_key} 필터 검증 실패 ({filter_errors[filter_key]})")
        if filter_key not in filter_groups:
            raise ValueError(f"{filter_key} 필터 그룹이 없습니다.")
        if not isinstance(case, dict):
            raise ValueError("패턴 항목은 JSON 객체여야 합니다.")
        pattern = validate_matrix(case.get("input"), int(size_text), "입력 패턴")
        expected = normalize_label(case.get("expected"), EXPECTED_LABELS, "expected")
    except ValueError as error:
        print(f"FAIL ({error})")
        return "스키마/데이터", str(error)

    group = filter_groups[filter_key]
    flat_pattern = flatten_matrix(pattern)
    score_cross = mac_flat(flat_pattern, group["Cross"])
    score_x = mac_flat(flat_pattern, group["X"])
    print(f"Cross 점수: {score_cross}")
    print(f"X 점수: {score_x}")
    try:
        result = classify(score_cross, score_x, "Cross", "X")
    except ValueError as error:
        print(f"FAIL ({error})")
        return "수치 범위", str(error)

    if result == expected:
        print(f"판정: {result} | expected: {expected} | PASS")
        return None

    if result == "UNDECIDED":
        category = "동점"
        gap = abs(score_cross - score_x)
        reason = f"동점 처리 (점수 차이 {gap:.3g} < {EPSILON:g})"
    else:
        category = "기대 라벨 불일치"
        reason = f"판정 {result}과 기대 라벨 {expected} 불일치"
    print(f"판정: {result} | expected: {expected} | FAIL ({reason})")
    return category, reason


def run_json(path):
    try:
        with path.open(encoding="utf-8") as source:
            data = json.load(source)
    except (OSError, ValueError, RecursionError) as error:
        print(f"JSON 로드 오류 ({path}): {error}")
        return 1

    if not isinstance(data, dict) or not isinstance(data.get("patterns"), dict):
        print("JSON 스키마 오류: 최상위 객체에 patterns 객체가 필요합니다.")
        return 1

    groups, filter_errors = load_filter_groups(data.get("filters"))
    patterns = data["patterns"]
    failures = []
    print("\n[패턴 분석]")
    for case_id, case in patterns.items():
        failure = analyze_case(case_id, case, groups, filter_errors)
        if failure is not None:
            category, reason = failure
            failures.append((case_id, category, reason))

    print_benchmark(benchmark_sample(size) for size in (3,) + JSON_SIZES)
    total = len(patterns)
    passed = total - len(failures)
    pass_rate = f"{passed / total:.1%}" if total else "해당 없음"
    print("\n[결과 요약]")
    print(f"총 테스트: {total}개")
    print(f"통과: {passed}개")
    print(f"실패: {len(failures)}개")
    print(f"통과율: {pass_rate}")
    if failures:
        counts = Counter(category for _, category, _ in failures)
        breakdown = ", ".join(f"{category} {count}개" for category, count in counts.items())
        print(f"실패 원인별 집계: {breakdown}")
        print("\n실패 케이스")
        for case_id, _, reason in failures:
            print(f"- {case_id}: {reason}")
    return 0


def main():
    print("=== Mini NPU Simulator ===")
    print("1. 사용자 입력 (3×3)")
    print("2. data.json 분석")
    try:
        while True:
            choice = input("선택: ").strip()
            if choice == "1":
                run_manual()
                return 0
            if choice == "2":
                return run_json(DATA_PATH)
            print("1 또는 2를 입력하세요.")
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 종료되어 실행을 중단합니다.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
