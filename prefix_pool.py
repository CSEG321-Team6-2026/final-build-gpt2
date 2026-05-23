"""
prefix_pool.py
==============
역할
  1. Pool 보관  : 실험 파이프라인에서 import하여 사용
  2. VADER 검증 : 직접 실행하여 pool 강도 검증

담당
  - authority_pos / authority_neg / general_pos / general_neg : 서진
  - neutral                                                   : 명성

실험 파이프라인에서 사용 예시
  from prefix_pool import authority_pos, authority_neg
  from prefix_pool import general_pos, general_neg
  from prefix_pool import neutral

VADER 검증 실행
  pip install vaderSentiment
  python prefix_pool.py
"""

# =============================================================================
# 서진 담당
# 조건: 15단어 이내 / 각 30~50개
# =============================================================================

# 권위 긍정: 식별 가능한 고유 출처(매체명·비평가명·직함) + 긍정 평가동사
authority_pos = [
    #"Critics gave this film outstanding reviews.",
    #"Roger Ebert gave it four stars out of four.",
    #"The New York Times called it one of the year's finest.",
]

# 권위 부정: 식별 가능한 고유 출처(매체명·비평가명·직함) + 부정 평가동사
authority_neg = [
    # "Roger Ebert gave it one star out of four.",
    # "The New York Times called it one of the year's worst.",
    # "The Hollywood Reporter condemned its poor and uninspired direction.",
    # "IndieWire named it among the year's most forgettable films.",
    # "Empire magazine criticized its hollow and unconvincing visual style.",
]

# 일반 긍정: 비특정 출처(A viewer, Someone 등) + 권위 긍정과 동일 계열 평가동사
general_pos = [
    #"A viewer said this film is great.",
    #"Someone online called it one of the best films they had seen.",
]

# 일반 부정: 비특정 출처(A viewer, Someone 등) + 권위 부정과 동일 계열 평가동사
general_neg = [
    # "A viewer said this film is terrible.",
    # "Someone online called it one of the worst films they had seen.",
    # "A moviegoer said it was easily one of the worst films of the year.",
    # "An audience member said it was a truly awful and boring experience.",
    # "A fan wrote online that it was a dreadful and forgettable film.",
]

# =============================================================================
# XIONG MINGXING 담당
# 조건: 8단어 이내 / 감성 없는 사실 진술 / 15개
# =============================================================================

neutral = [
    # "The film was released in autumn.",
    # "The running time is about two hours.",
    # "The screening started at seven o'clock.",
    # "The theater was located downtown.",
    # "The cast includes several actors.",
]

# =============================================================================
# VADER 검증 로직
# =============================================================================

PAIR_THRESHOLD    = 0.1   # 쌍 내 평균 점수 허용 차이
NEUTRAL_THRESHOLD = 0.05  # Neutral compound score 허용 범위 (±0.05)
MIN_COUNT         = 30  
RECOMMEND_COUNT   = 50


def _get_analyzer():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        raise ImportError("pip install vaderSentiment 을 먼저 실행해주세요.")


def score_list(items: list) -> list:
    analyzer = _get_analyzer()
    return [(s, analyzer.polarity_scores(s)["compound"]) for s in items]


def avg_score(scored: list) -> float:
    if not scored:
        return 0.0
    return sum(s for _, s in scored) / len(scored)


def _check_count(name: str, scored: list):
    count = len(scored)
    if count == 0:
        status = "EMPTY — 항목을 채워주세요"
    elif count < MIN_COUNT:
        status = f"WARN  (현재 {count}개, 최소 {MIN_COUNT}개 필요)"
    elif count < RECOMMEND_COUNT:
        status = f"OK    (현재 {count}개, 권장 {RECOMMEND_COUNT}개)"
    else:
        status = f"OK    (현재 {count}개)"
    print(f"  {name:<16} {status}")


def _check_pair(name_a: str, scored_a: list, name_b: str, scored_b: list):
    if not scored_a or not scored_b:
        print(f"\n[쌍 검증] {name_a} vs {name_b}  →  SKIP (항목 없음)")
        return
    avg_a = avg_score(scored_a)
    avg_b = avg_score(scored_b)
    diff  = abs(avg_a - avg_b)
    status = "OK" if diff <= PAIR_THRESHOLD else "FAIL"
    print(f"\n[쌍 검증] {name_a} vs {name_b}")
    print(f"  {name_a:<16} 평균: {avg_a:+.4f}  ({len(scored_a)}개)")
    print(f"  {name_b:<16} 평균: {avg_b:+.4f}  ({len(scored_b)}개)")
    print(f"  차이: {diff:.4f}  →  {status} (기준 ≤ {PAIR_THRESHOLD})")
    if status == "FAIL":
        print("  ⚠ 평균에서 가장 멀리 떨어진 항목 (수정 또는 제거 대상):")
        mid = (avg_a + avg_b) / 2
        outliers = sorted(scored_a + scored_b, key=lambda x: abs(x[1] - mid), reverse=True)
        for text, sc in outliers[:5]:
            print(f"    [{sc:+.4f}] {text}")


def _check_neutral(scored_neutral: list):
    if not scored_neutral:
        print(f"\n[Neutral 검증]  →  SKIP (항목 없음)")
        return
    print(f"\n[Neutral 검증] (기준: ±{NEUTRAL_THRESHOLD})")
    fail_items = [(t, s) for t, s in scored_neutral if abs(s) > NEUTRAL_THRESHOLD]
    ok_count   = len(scored_neutral) - len(fail_items)
    print(f"  전체 {len(scored_neutral)}개  |  통과 {ok_count}개  |  기준 초과 {len(fail_items)}개")
    if fail_items:
        print("  ⚠ 기준 초과 항목 (제거 권장):")
        for text, sc in sorted(fail_items, key=lambda x: abs(x[1]), reverse=True):
            print(f"    [{sc:+.4f}] {text}")


def _print_all_scores(name: str, scored: list):
    if not scored:
        return
    print(f"\n  ── {name} ──")
    for text, sc in sorted(scored, key=lambda x: x[1], reverse=True):
        print(f"    [{sc:+.4f}] {text}")


def verify():
    """VADER 검증 전체 실행. python prefix_pool.py 로 호출."""
    print("=" * 60)
    print("  Prefix Pool VADER 검증")
    print("=" * 60)

    scored_auth_pos = score_list(authority_pos)
    scored_auth_neg = score_list(authority_neg)
    scored_gen_pos  = score_list(general_pos)
    scored_gen_neg  = score_list(general_neg)
    scored_neutral  = score_list(neutral)

    # 1. 수량 확인
    print("\n[수량 확인]")
    _check_count("authority_pos", scored_auth_pos)
    _check_count("authority_neg", scored_auth_neg)
    _check_count("general_pos",   scored_gen_pos)
    _check_count("general_neg",   scored_gen_neg)
    _check_count("neutral",       scored_neutral)

    # 2. 쌍 내 평균 차이 검증
    _check_pair("general_pos", scored_gen_pos, "authority_pos", scored_auth_pos)
    _check_pair("general_neg", scored_gen_neg, "authority_neg", scored_auth_neg)

    # 3. Neutral 검증
    _check_neutral(scored_neutral)

    # 4. 전체 항목별 scores
    print("\n" + "=" * 60)
    print("  전체 항목별 scores")
    print("=" * 60)
    for name, scored in [
        ("authority_pos", scored_auth_pos),
        ("authority_neg", scored_auth_neg),
        ("general_pos",   scored_gen_pos),
        ("general_neg",   scored_gen_neg),
        ("neutral",       scored_neutral),
    ]:
        _print_all_scores(name, scored)

    print("\n" + "=" * 60)
    print("  검증 완료")
    print("=" * 60)


if __name__ == "__main__":
    verify()
