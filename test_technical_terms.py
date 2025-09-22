#!/usr/bin/env python3
"""
Simple test to verify technical terms whitelist functionality
"""

TECHNICAL_TERMS_WHITELIST = {
    'oauth', 'api', 'sdk', 'rest', 'json', 'xml', 'http', 'https', 'ssl', 'tls',
    'microsoft', 'azure', 'office', 'outlook', 'sharepoint', 'onedrive', 'teams',
    'dynamics', 'powerbi', 'powerapps', 'flow', 'intune', 'activedirectory',
    'actionpack', 'partnercenter', 'msdn', 'technet', 'visualstudio', 'vscode',
    'dotnet', 'csharp', 'javascript', 'typescript', 'nodejs', 'npm', 'nuget',
    'github', 'gitlab', 'devops', 'cicd', 'docker', 'kubernetes', 'aws', 'gcp'
}

def is_technical_term(term):
    """Check if a term is in the technical terms whitelist (case-insensitive)"""
    return term.lower().strip() in TECHNICAL_TERMS_WHITELIST

def test_technical_terms_whitelist():
    """Test that common technical terms are properly whitelisted"""
    test_terms = ['OAuth', 'API', 'SDK', 'Microsoft', 'Azure', 'REST', 'JSON', 'PowerBI', 'ActionPack']
    
    print("Testing technical terms whitelist...")
    for term in test_terms:
        result = is_technical_term(term)
        print(f"{term}: {result}")
        assert result == True, f"{term} should be whitelisted"
    
    print("\nTesting case insensitivity...")
    case_test_terms = ['oauth', 'OAUTH', 'Api', 'SDK', 'microsoft', 'MICROSOFT']
    for term in case_test_terms:
        result = is_technical_term(term)
        print(f"{term}: {result}")
        assert result == True, f"{term} should be whitelisted (case insensitive)"
    
    print("\nTesting non-technical terms...")
    non_technical_terms = ['hello', 'world', 'random', 'word']
    for term in non_technical_terms:
        result = is_technical_term(term)
        print(f"{term}: {result}")
        assert result == False, f"{term} should NOT be whitelisted"
    
    print("\nAll technical terms properly whitelisted!")
    print("Whitelist functionality working correctly!")

if __name__ == "__main__":
    test_technical_terms_whitelist()
