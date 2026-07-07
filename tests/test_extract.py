from council.extract import extract_final_answer, extract_number


def test_extracts_final_answer_line():
    text = "Reasoning blah blah.\nFINAL ANSWER: 42 apples"
    assert extract_final_answer(text) == "42 apples"


def test_uses_last_final_answer_when_repeated():
    text = "FINAL ANSWER: 1\nmore thinking\nFINAL ANSWER: 2"
    assert extract_final_answer(text) == "2"


def test_case_insensitive_and_markdown_tolerant():
    assert extract_final_answer("**Final Answer:** *7*") == "7"


def test_none_when_absent():
    assert extract_final_answer("no marker here") is None


def test_number_from_final_answer_line():
    assert extract_number("The sum is 10.\nFINAL ANSWER: The total is 1,234 dollars") == 1234.0


def test_number_fallback_to_last_number_in_text():
    assert extract_number("First 3 then 7 so the result is 21.") == 21.0


def test_number_negative_and_decimal():
    assert extract_number("FINAL ANSWER: -3.5") == -3.5


def test_number_none_when_no_digits():
    assert extract_number("FINAL ANSWER: unknowable") is None
