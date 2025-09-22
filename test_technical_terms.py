"""
Test cases for technical terms spell check improvements.
"""
from technical_terms import is_technical_term, TECHNICAL_TERMS_WHITELIST

def test_technical_terms_whitelist():
    """Test that common technical terms are recognized."""
    technical_terms = ['API', 'SDK', 'OAuth', 'JSON', 'XML', 'SQL', 'REST']
    for term in technical_terms:
        assert is_technical_term(term), f"{term} should be recognized as technical term"
        assert is_technical_term(term.lower()), f"{term.lower()} should be recognized as technical term"
    print("✓ Technical terms whitelist test passed")

def test_microsoft_products():
    """Test that Microsoft products are recognized."""
    ms_products = ['Office', 'Excel', 'SharePoint', 'Teams', 'Azure', 'Dynamics']
    for product in ms_products:
        assert is_technical_term(product), f"{product} should be recognized as Microsoft product"
    print("✓ Microsoft products test passed")

def test_non_technical_terms():
    """Test that regular words are not in whitelist."""
    regular_words = ['hello', 'world', 'computer', 'software', 'program']
    for word in regular_words:
        assert not is_technical_term(word), f"{word} should not be in technical terms whitelist"
    print("✓ Non-technical terms test passed")

def test_case_insensitive():
    """Test case insensitive matching."""
    assert is_technical_term('api')
    assert is_technical_term('API')
    assert is_technical_term('Api')
    assert is_technical_term('office')
    assert is_technical_term('OFFICE')
    assert is_technical_term('Office')
    print("✓ Case insensitive test passed")

def test_edge_cases():
    """Test edge cases like empty strings and whitespace."""
    assert not is_technical_term('')
    assert not is_technical_term('   ')
    assert not is_technical_term(None)
    assert is_technical_term('  api  ')  # Should handle whitespace
    print("✓ Edge cases test passed")

if __name__ == "__main__":
    test_technical_terms_whitelist()
    test_microsoft_products()
    test_non_technical_terms()
    test_case_insensitive()
    test_edge_cases()
    print(f"\nAll tests passed! Whitelist contains {len(TECHNICAL_TERMS_WHITELIST)} technical terms.")
