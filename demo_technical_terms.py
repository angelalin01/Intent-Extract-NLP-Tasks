"""
Demonstration script showing technical terms spell check improvement.
"""

from technical_terms import is_technical_term

def demo_spell_check_improvement():
    """Demonstrate how technical terms are now properly handled"""
    
    test_terms = [
        'OAuth', 'API', 'SDK', 'Microsoft', 'Azure', 'Office365',
        'SharePoint', 'PowerShell', 'JavaScript', 'TypeScript',
        'oauth2', 'api_key', 'rest_api', 'azure_storage',
        'v1.0', '2.1', 'https://example.com'
    ]
    
    print("Technical Terms Spell Check Demo")
    print("=" * 40)
    print("The following terms are now whitelisted and won't be flagged as misspellings:")
    print()
    
    for term in test_terms:
        is_tech = is_technical_term(term)
        status = "✓ WHITELISTED" if is_tech else "✗ NOT WHITELISTED"
        print(f"{term:20} -> {status}")
    
    print()
    print("Non-technical terms (should still be spell-checked):")
    non_tech_terms = ['hello', 'wrold', 'speling', 'eror']
    
    for term in non_tech_terms:
        is_tech = is_technical_term(term)
        status = "✓ WHITELISTED" if is_tech else "✗ WILL BE SPELL-CHECKED"
        print(f"{term:20} -> {status}")

if __name__ == '__main__':
    demo_spell_check_improvement()
