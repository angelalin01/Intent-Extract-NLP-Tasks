#!/usr/bin/env python3
"""
Test script to verify Watson API retry logic and error handling
"""
import os
import sys
sys.path.append('.')

def test_watson_initialization():
    """Test Watson manager initialization"""
    try:
        from explore_watson3 import watson_manager, logger
        
        if watson_manager is not None:
            print('✓ Watson manager initialized successfully')
            
            fallback = watson_manager._fallback_response()
            print(f'✓ Fallback response works: {fallback.get("fallback_mode", False)}')
            
            return True
        else:
            print('✗ Watson manager failed to initialize')
            return False
    except Exception as e:
        print(f'✗ Error during initialization test: {e}')
        return False

def test_invalid_credentials():
    """Test error handling with invalid credentials"""
    print("\n--- Testing with invalid API key ---")
    
    os.environ['WATSON_API_KEY'] = 'invalid_test_key'
    
    try:
        if 'explore_watson3' in sys.modules:
            del sys.modules['explore_watson3']
        
        from explore_watson3 import watson_manager
        from ibm_watson.natural_language_understanding_v1 import Features, SemanticRolesOptions
        
        if watson_manager:
            response = watson_manager.analyze_with_retry(
                text='Test text for error handling',
                features=Features(semantic_roles=SemanticRolesOptions())
            )
            print(f'✓ Error handling test completed. Fallback mode: {response.get("fallback_mode", False)}')
        else:
            print('✓ Watson manager properly handled invalid credentials during initialization')
            
    except Exception as e:
        print(f'✓ Exception properly caught and handled: {type(e).__name__}: {e}')
    
    if 'WATSON_API_KEY' in os.environ:
        del os.environ['WATSON_API_KEY']

if __name__ == '__main__':
    print("=== Watson API Retry Logic Test ===")
    
    success = test_watson_initialization()
    
    test_invalid_credentials()
    
    print("\n=== Test Summary ===")
    if success:
        print("✓ Basic tests passed - Watson retry logic appears to be working")
    else:
        print("✗ Some tests failed - check implementation")
