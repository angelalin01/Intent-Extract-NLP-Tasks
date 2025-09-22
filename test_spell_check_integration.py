"""
Integration test for spell check improvements with technical terms.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from technical_terms import is_technical_term
from distanceedit import DistanceEdit

def test_technical_terms_not_flagged():
    """Test that technical terms are properly recognized and not processed for spell checking."""
    technical_terms = [
        'API', 'SDK', 'OAuth', 'JSON', 'XML', 'SQL', 'REST', 'HTTP', 'HTTPS',
        'Office', 'Excel', 'SharePoint', 'Teams', 'Azure', 'PowerShell'
    ]
    
    print("Testing technical terms recognition:")
    for term in technical_terms:
        result = is_technical_term(term)
        print(f"  {term}: {'✓ Recognized' if result else '✗ Not recognized'}")
        assert result, f"{term} should be recognized as a technical term"
    
    print("✓ All technical terms properly recognized")

def test_distance_calculation():
    """Test the distance calculation functionality."""
    distance_calc = DistanceEdit()
    
    dist1 = distance_calc.iterative_levenshtein("test", "best")
    assert dist1 == 1, f"Expected distance 1, got {dist1}"
    
    dist2 = distance_calc.technical_term_distance("API", "API")
    assert dist2 == 0, f"Expected distance 0 for identical strings, got {dist2}"
    
    print("✓ Distance calculation tests passed")

def test_spell_check_logic():
    """Test the core spell check logic with technical terms."""
    sample_entities = ['API', 'OAuth', 'misspeled', 'configure', 'SharePoint']
    
    technical_count = 0
    non_technical_count = 0
    
    for entity in sample_entities:
        if is_technical_term(entity):
            technical_count += 1
            print(f"  {entity}: Skipped (technical term)")
        else:
            non_technical_count += 1
            print(f"  {entity}: Processed for spell checking")
    
    assert technical_count == 3, f"Expected 3 technical terms, found {technical_count}"
    assert non_technical_count == 2, f"Expected 2 non-technical terms, found {non_technical_count}"
    
    print("✓ Spell check logic test passed")

def test_case_variations():
    """Test that technical terms work with different case variations."""
    test_cases = [
        ('api', True), ('API', True), ('Api', True),
        ('oauth', True), ('OAuth', True), ('OAUTH', True),
        ('sharepoint', True), ('SharePoint', True), ('SHAREPOINT', True),
        ('notechnical', False), ('NOTECHNICAL', False)
    ]
    
    print("Testing case variations:")
    for term, expected in test_cases:
        result = is_technical_term(term)
        status = "✓" if result == expected else "✗"
        print(f"  {term}: {status} {'Recognized' if result else 'Not recognized'}")
        assert result == expected, f"{term} recognition failed"
    
    print("✓ Case variations test passed")

if __name__ == "__main__":
    print("Running spell check integration tests...\n")
    
    test_technical_terms_not_flagged()
    print()
    
    test_distance_calculation()
    print()
    
    test_spell_check_logic()
    print()
    
    test_case_variations()
    print()
    
    print("🎉 All integration tests passed!")
    print("\nSummary:")
    print("- Technical terms (API, OAuth, SDK, etc.) are properly recognized")
    print("- Microsoft products (Office, SharePoint, Teams, etc.) are whitelisted")
    print("- Case-insensitive matching works correctly")
    print("- Non-technical terms are still processed for spell checking")
    print("- Distance calculations work as expected")
