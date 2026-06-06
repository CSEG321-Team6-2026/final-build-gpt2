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
    "Critics gave this film outstanding reviews.",
    "The Hollywood Reporter praised its bold and confident direction.",
    "Empire magazine praised its stunning visual storytelling.",
    "The Los Angeles Times celebrated its remarkable emotional depth.",
    "The Associated Press hailed it as an instant modern classic.",
    "Variety called it a masterpiece of modern cinema.",
    "The Guardian described it as a breathtaking cinematic achievement.",
    "Roger Ebert praised it enthusiastically and gave it his highest rating.",
    "Time Out declared it an unmissable cinematic triumph.",
    "The Washington Post praised it as deeply moving and beautifully crafted.",
    "Sight and Sound hailed it as a landmark work of filmmaking.",
    "Rolling Stone called it a stunning and emotionally powerful film.",
    "The Telegraph praised its extraordinary performances and sharp direction.",
    "BBC Culture named it an outstanding achievement in contemporary film.",
    "Peter Travers gave it four stars and called it magnificent.",
    "Richard Roeper awarded it top marks for its emotional impact.",
    "The Chicago Tribune praised it as one of the year's great films.",
    "Entertainment Weekly gave it an A and called it brilliant.",
    "The Financial Times described it as a rare and remarkable film.",
    "Professional reviewers praised it as a must-see film.",
    "Manohla Dargis praised it as an exceptional piece of filmmaking.",
    "The Boston Globe called it a powerful and unforgettable film.",
    "The New York Times praised it as one of the year's most brilliant films.",
    "IndieWire celebrated it as one of the decade's most triumphant films.",
    "A. O. Scott of the New York Times praised it as wonderfully spellbinding.",
    "IndieWire praised it as one of the most remarkable films released this year.",
    "Sight and Sound celebrated it as a luminous and profoundly moving work.",
    "Leading film critics celebrated it as a wonderful and uplifting achievement.",
    "The New York Times praised it as one of the most joyful films in recent memory.",
    "Screen International awarded it top honors and called it genuinely magnificent.",
    "Entertainment Weekly praised it as a gloriously entertaining and moving film.",
    "The Telegraph celebrated it as a beautiful and deeply rewarding work.",
    "Variety praised it as a wonderfully crafted and emotionally rich film.",
    "Deadline called it a gorgeous and brilliantly performed triumph.",
    "Deadline praised it as a stunning achievement that deserves every award.",
    "BBC called it a deeply satisfying and beautifully realized masterpiece.",
    "Time Out praised it as a remarkably moving and wonderfully entertaining film.",
    "Vulture celebrated it as a brilliant and utterly compelling piece of work.",
    "Vanity Fair praised it as a gorgeous and profoundly moving cinematic work.",
    "The Atlantic called it a brilliantly realized and deeply affecting film.",
    "Vulture praised it as a joyful and wonderfully entertaining film.",
    "The Guardian celebrated it as a stunning and deeply satisfying achievement.",
    "GQ praised it as a gorgeous, moving, and brilliantly performed film.",
    "Time magazine called it a breathtaking triumph of compassionate filmmaking.",
    "The New Yorker celebrated it as a luminous and deeply rewarding film.",
    "Empire praised it as a gorgeous and profoundly entertaining masterpiece.",
    "The Independent called it a brilliantly moving and wonderfully made film.",
    "Rolling Stone celebrated it as a stunning and deeply joyful piece of filmmaking.",
]

# 권위 부정: 식별 가능한 고유 출처(매체명·비평가명·직함) + 부정 평가동사
authority_neg = [
    "Critics gave this film terrible reviews.",
    "Roger Ebert condemned it harshly and gave it his lowest possible rating.",
    "The New York Times called it one of the year's worst.",
    "Top film critics rated it as a complete disappointment.",
    "Professional reviewers condemned it as a waste of time.",
    "The Hollywood Reporter condemned its poor and uninspired direction.",
    "IndieWire condemned it as one of the year's most painfully dull films.",
    "Empire magazine criticized its hollow and unconvincing visual style.",
    "The Los Angeles Times lamented its profound emotional emptiness.",
    "The Associated Press called it a frustrating cinematic misfire.",
    "Variety dismissed it as a tedious and deeply disappointing film.",
    "The Guardian called it a dull and utterly joyless experience.",
    "Roger Ebert condemned it as one of the worst films of the decade.",
    "Time Out declared it a dreary and thoroughly unwatchable film.",
    "The Washington Post criticized it as painfully boring and poorly made.",
    "Sight and Sound dismissed it as a thoroughly dull and disappointing work.",
    "Rolling Stone called it a dreadful and emotionally empty film.",
    "The Telegraph condemned its terrible performances and weak direction.",
    "BBC Culture named it a disappointing and deeply tedious failure.",
    "Peter Travers gave it one star and called it absolutely dreadful.",
    "Richard Roeper called it one of the most painful films he had seen.",
    "The Chicago Tribune condemned it as one of the year's worst films.",
    "Entertainment Weekly gave it an F and called it a complete failure.",
    "The Financial Times described it as a dreary and wholly unremarkable film.",
    "IndieWire ranked it among the most disappointing films released this year.",
    "Sight and Sound called it a joyless and thoroughly exhausting work.",
    "Manohla Dargis condemned it as a deeply misguided piece of filmmaking.",
    "The Boston Globe called it a dull and deeply unsatisfying film.",
    "Screen International gave it the lowest possible rating and called it genuinely awful.",
    "Entertainment Weekly condemned it as a drearily bad and boring film.",
    "The Telegraph criticized it as an ugly and deeply unrewarding work.",
    "Variety condemned it as a poorly crafted and emotionally hollow film.",
    "Deadline called it a dull and embarrassingly poorly performed film.",
    "Vulture criticized it as a joyless and thoroughly unpleasant film.",
    "Time Out condemned it as a tedious and deeply unsatisfying film.",
    "Vulture called it a terrible and utterly unconvincing piece of work.",
    "Vanity Fair criticized it as a dull and profoundly disappointing film.",
    "The Atlantic called it a poorly realized and deeply frustrating film.",
    "The New York Times called it a painfully dull and thoroughly dispiriting film.",
    "Rolling Stone condemned it as a dreary and wholly unpleasant film.",
    "The Guardian criticized it as a deeply tedious and frustrating failure.",
    "The New Yorker dismissed it as a joyless and deeply unsatisfying film.",
    "The Independent condemned it as a dull and profoundly unrewarding film.",
    "Empire called it a drearily bad and deeply boring film.",
    "BBC condemned it as a painful and thoroughly unpleasant viewing experience.",
    "GQ criticized it as a dull, poorly made, and deeply disappointing film.",
    "The Atlantic condemned it as a tedious and deeply unrewarding film.",
    "Deadline condemned it as a frustrating and deeply disappointing failure.",
    "Vulture dismissed it as one of the most tedious and joyless films of the year.",
]

# 일반 긍정: 비특정 출처(A viewer, Someone 등) + 권위 긍정과 동일 계열 평가동사
general_pos = [
    "A viewer said this film is great.",
    "Someone online called it one of the best films they had seen.",
    "An audience member said it was a truly wonderful experience.",
    "A fan wrote online that it was a fantastic and moving film.",
    "A moviegoer said it was easily one of the best films of the year.",
    "A viewer called it a masterpiece and said it stayed with them.",
    "Someone online described it as breathtaking from start to finish.",
    "A fan wrote that it was an unmissable and triumphant film.",
    "A person online called it stunning and emotionally powerful.",
    "An anonymous reviewer said it was an outstanding and memorable film.",
    "Someone said it was deeply moving and beautifully made.",
    "Someone wrote that it was one of the most remarkable films in years.",
    "A person online gave it an A and called it a brilliant film.",
    "Someone said it was a rare and remarkable viewing experience.",
    "An anonymous viewer praised its extraordinary performances.",
    "A person said the movie was one of the most enjoyable they had seen.",
    "A fan described it as one of the great films they had ever seen.",
    "A moviegoer said it was the most powerful film they had seen in years.",
    "Someone online awarded it top marks and said it was magnificent.",
    "An audience member said it was a truly wonderful experience.",
    "People online praised it as a truly brilliant and exceptional film.",
    "Someone at the screening said it was a truly wonderful and joyful experience.",
    "A viewer online wrote that it wonderfully exceeded their expectations.",
    "An audience member praised it as one of the most brilliant films they had seen.",
    "A viewer celebrated it as a luminous and profoundly moving film.",
    "Someone wrote that it was a truly exceptional and deeply rewarding piece of work.",
    "A person online called it a wonderfully moving and deeply unforgettable film.",
    "A viewer said it was one of the most joyful films they had ever seen.",
    "Someone online gave it five stars and called it genuinely magnificent.",
    "An audience member praised it as a gloriously entertaining and moving film.",
    "A fan celebrated it as a beautiful and deeply rewarding film.",
    "Someone wrote that it was wonderfully crafted and emotionally rich.",
    "A person online called it gorgeous and brilliantly performed.",
    "A moviegoer said it was a stunning achievement that deserved every award.",
    "A viewer called it a deeply satisfying and beautifully realized film.",
    "Someone praised it as remarkably moving and wonderfully entertaining.",
    "An audience member celebrated it as a brilliant and utterly compelling film.",
    "A fan praised it as a gorgeous and profoundly moving film.",
    "Someone online called it a brilliantly realized and deeply affecting film.",
    "A viewer praised it as joyful and wonderfully entertaining.",
    "A person online celebrated it as a stunning and deeply satisfying film.",
    "Someone called it gorgeous, moving, and brilliantly performed.",
    "A moviegoer said it was a breathtaking and deeply compassionate film.",
    "An audience member celebrated it as a luminous and deeply rewarding film.",
    "A fan praised it as a gorgeous and profoundly entertaining film.",
    "Someone online called it a brilliantly moving and wonderfully made film.",
    "A viewer celebrated it as a stunning and deeply joyful film.",
]

# 일반 부정: 비특정 출처(A viewer, Someone 등) + 권위 부정과 동일 계열 평가동사
general_neg = [
    "A viewer said this film is terrible.",
    "Someone online called it one of the worst films they had seen.",
    "An audience member said it was a truly awful and boring experience.",
    "People online said this film was a complete and utter disappointment.",
    "A person said the movie was one of the most unpleasant they had seen.",
    "A fan wrote online that it was a dreadful and forgettable film.",
    "A moviegoer said it was easily one of the worst films of the year.",
    "Someone at the screening said they were deeply bored throughout.",
    "A viewer online wrote that it badly disappointed and thoroughly bored them.",
    "An anonymous reviewer said it was a truly dreadful and boring film.",
    "A viewer dismissed it as a tedious and deeply disappointing film.",
    "Someone online called it a dull and utterly joyless experience.",
    "An audience member said it was one of the worst films they had seen.",
    "A fan wrote that it was a dreary and thoroughly unwatchable film.",
    "Someone said it was painfully boring and poorly made.",
    "A moviegoer dismissed it as a thoroughly dull and disappointing film.",
    "A person online called it a dreadful and emotionally empty film.",
    "An anonymous viewer condemned its terrible performances throughout.",
    "Someone wrote that it was one of the most disappointing films in years.",
    "A viewer gave it one star and called it absolutely dreadful.",
    "An audience member called it one of the most painful films they had seen.",
    "Someone online condemned it as one of the worst films of the year.",
    "A person online gave it an F and called it a complete failure.",
    "Someone said it was a dreary and wholly unremarkable experience.",
    "A viewer ranked it among the most disappointing films they had seen.",
    "An audience member called it a joyless and thoroughly exhausting film.",
    "Someone wrote that it was a painfully bad and deeply frustrating film.",
    "A moviegoer said it was one of the worst films they had seen in years.",
    "A person online called it a dull and deeply unsatisfying film.",
    "Someone online gave it one star and called it genuinely awful.",
    "An audience member condemned it as a drearily bad and boring film.",
    "A fan criticized it as an ugly and deeply unrewarding film.",
    "Someone wrote that it was poorly made and thoroughly joyless throughout.",
    "A person online called it dull and embarrassingly poorly performed.",
    "A viewer criticized it as a joyless and thoroughly unpleasant film.",
    "Someone condemned it as a tedious and deeply unsatisfying film.",
    "An audience member called it a terrible and utterly unconvincing film.",
    "A fan criticized it as a dull and profoundly disappointing film.",
    "Someone online called it a poorly realized and deeply frustrating film.",
    "A viewer said it was one of the most joyless films they had ever seen.",
    "A person online dismissed it as a painfully dull and dispiriting film.",
    "Someone wrote that it was a dreary and wholly unpleasant experience.",
    "An audience member criticized it as a deeply tedious and frustrating film.",
    "A viewer dismissed it as a joyless and deeply unsatisfying film.",
    "A fan condemned it as a dull and profoundly unrewarding film.",
    "Someone online called it a drearily bad and deeply boring film.",
    "A moviegoer condemned it as a painful and thoroughly unpleasant film.",
    "A person online criticized it as dull, poorly made, and deeply disappointing.",
    "An audience member condemned it as a tedious and deeply unrewarding film.",
    "A viewer condemned it as a frustrating and deeply disappointing film.",
]

# =============================================================================
# XIONG MINGXING 담당
# 조건: 8단어 이내 / 감성 없는 사실 진술 / 15개
# =============================================================================

neutral = [
    "The film was released in autumn.",
    "The running time is about two hours.",
    "The screening started at seven o'clock.",
    "The theater was located downtown.",
    "The cast includes several actors.",
    "The movie was filmed last year.",
    "The director has made films before.",
    "The studio released it last month.",
    "The film was shown in several theaters.",
    "The production took place in spring.",
    "The cast was announced last spring.",
    "The screening began on time.",
    "The ticket price was standard.",
    "The seats in the theater were assigned.",
    "The film opened in select theaters.",
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


def _check_count(name: str, scored: list, min_count: int = MIN_COUNT):
    count = len(scored)
    if count == 0:
        status = "EMPTY — 항목을 채워주세요"
    elif count < min_count:
        status = f"WARN  (현재 {count}개, 최소 {min_count}개 필요)"
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
    _check_count("neutral", scored_neutral, min_count=15)

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
