from app.parsers.markdown import find_fences


def test_two_separate_fences_not_merged():
    text = "```a```\n\nbetween\n\n```b```"
    fences = find_fences(text)
    assert fences == ["a", "b"]

def test_fence_with_newlines_inside():
    text = "```py\nx = 1\n```"
    fences = find_fences(text)
    assert len(fences) == 1
    assert "x = 1" in fences[0]
