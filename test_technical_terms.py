"""
Unit tests for technical terms whitelist functionality.
"""

import unittest
from technical_terms import is_technical_term, get_technical_terms_list

class TestTechnicalTerms(unittest.TestCase):
    
    def test_exact_matches(self):
        """Test exact technical term matches"""
        self.assertTrue(is_technical_term('OAuth'))
        self.assertTrue(is_technical_term('oauth'))
        self.assertTrue(is_technical_term('API'))
        self.assertTrue(is_technical_term('api'))
        self.assertTrue(is_technical_term('SDK'))
        self.assertTrue(is_technical_term('Microsoft'))
        self.assertTrue(is_technical_term('microsoft'))
        
    def test_case_insensitive(self):
        """Test case insensitive matching"""
        self.assertTrue(is_technical_term('OAUTH'))
        self.assertTrue(is_technical_term('Api'))
        self.assertTrue(is_technical_term('MicroSoft'))
        
    def test_pattern_matches(self):
        """Test pattern-based matching"""
        self.assertTrue(is_technical_term('oauth2'))
        self.assertTrue(is_technical_term('api_key'))
        self.assertTrue(is_technical_term('rest_api'))
        self.assertTrue(is_technical_term('azure_storage'))
        self.assertTrue(is_technical_term('office365_admin'))
        
    def test_version_numbers(self):
        """Test version number patterns"""
        self.assertTrue(is_technical_term('v1.0'))
        self.assertTrue(is_technical_term('2.1'))
        self.assertTrue(is_technical_term('v3.14'))
        
    def test_non_technical_terms(self):
        """Test that non-technical terms are not matched"""
        self.assertFalse(is_technical_term('hello'))
        self.assertFalse(is_technical_term('world'))
        self.assertFalse(is_technical_term('random'))
        self.assertFalse(is_technical_term(''))
        self.assertFalse(is_technical_term(None))
        
    def test_get_technical_terms_list(self):
        """Test getting the complete list"""
        terms = get_technical_terms_list()
        self.assertIsInstance(terms, set)
        self.assertIn('oauth', terms)
        self.assertIn('api', terms)
        self.assertIn('microsoft', terms)

if __name__ == '__main__':
    unittest.main()
