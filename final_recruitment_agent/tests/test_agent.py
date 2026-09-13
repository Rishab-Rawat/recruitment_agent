from matching_agent import extract_requirements, build_graph, compare_candidates, generate_interview_questions

def test_extract_requirements():
    r = extract_requirements("Must have Python and React. 3+ years experience. Docker is a plus.")
    assert r["must_have"]["min_years"] == 3
    assert "Python" in r["must_have"]["skills"] or "React" in r["must_have"]["skills"]

def test_graph_compiles():
    assert build_graph() is not None

def test_empty_tools_are_safe():
    s = {"candidate_shortlist": []}
    assert "error" in compare_candidates(["A", "B"], s)
    assert "error" in generate_interview_questions("A", s)
