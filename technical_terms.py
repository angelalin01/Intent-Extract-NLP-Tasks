"""
Technical terms whitelist for spell check accuracy improvement.
Contains common technical terms that should not be flagged as misspellings.
"""

import re

TECHNICAL_TERMS = {
    'oauth', 'oauth2', 'api', 'sdk', 'jwt', 'ssl', 'tls', 'https', 'http',
    'saml', 'ldap', 'sso', 'mfa', '2fa', 'auth', 'token', 'bearer',
    
    'microsoft', 'azure', 'office365', 'sharepoint', 'onedrive', 'outlook',
    'teams', 'skype', 'xbox', 'windows', 'powershell', 'activedirectory',
    'dynamics', 'sql', 'sqlserver', 'iis', 'dotnet', 'aspnet', 'csharp',
    'visualstudio', 'vscode', 'github', 'nuget', 'xamarin', 'blazor',
    
    'aws', 'gcp', 'kubernetes', 'docker', 'vm', 'vms', 'cdn', 'dns',
    'vpc', 'vpn', 'firewall', 'loadbalancer', 'autoscaling', 'cicd',
    
    'json', 'xml', 'yaml', 'html', 'css', 'javascript', 'typescript',
    'python', 'java', 'nodejs', 'npm', 'pip', 'git', 'github', 'gitlab',
    'bitbucket', 'svn', 'ci', 'cd', 'devops', 'agile', 'scrum',
    
    'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'nosql',
    'blob', 'queue', 'cache', 'backup', 'snapshot', 'replication',
    
    'tcp', 'udp', 'ip', 'ipv4', 'ipv6', 'smtp', 'pop3', 'imap',
    'ftp', 'sftp', 'ssh', 'telnet', 'snmp', 'dhcp', 'nat',
    
    'pdf', 'docx', 'xlsx', 'pptx', 'csv', 'tsv', 'zip', 'tar',
    'gz', 'rar', 'iso', 'exe', 'msi', 'dmg', 'deb', 'rpm',
    
    'url', 'uri', 'uuid', 'guid', 'id', 'ui', 'ux', 'gui', 'cli',
    'rest', 'soap', 'rpc', 'grpc', 'graphql', 'crud', 'orm',
    'mvc', 'mvp', 'mvvm', 'spa', 'pwa', 'seo', 'cms', 'crm',
    'erp', 'bi', 'etl', 'iot', 'ai', 'ml', 'nlp', 'ocr'
}

def is_technical_term(word):
    """
    Check if a word is a technical term that should be whitelisted.
    
    Args:
        word (str): The word to check
        
    Returns:
        bool: True if the word is a technical term, False otherwise
    """
    if not word:
        return False
    
    word_lower = word.lower().strip()
    
    if word_lower in TECHNICAL_TERMS:
        return True
    
    if any(pattern in word_lower for pattern in ['.com', '.net', '.org', '.gov', '.edu', 'http://', 'https://']):
        return True
    
    if re.match(r'^v?\d+\.\d+', word_lower):
        return True
    
    technical_patterns = [
        r'^api[_-]',  # api_key, api-endpoint
        r'[_-]api$',  # rest_api, web-api
        r'^oauth',    # oauth2, oauthtoken
        r'microsoft', # any word containing microsoft
        r'^azure',    # azure services
        r'office365', # office365 variants
    ]
    
    for pattern in technical_patterns:
        if re.search(pattern, word_lower):
            return True
    
    return False

def get_technical_terms_list():
    """
    Get the complete list of technical terms.
    
    Returns:
        set: Set of technical terms
    """
    return TECHNICAL_TERMS.copy()
