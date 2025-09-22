"""
Technical terms whitelist for spell check accuracy improvement.
Contains common technical abbreviations and Microsoft product names.
"""

TECHNICAL_ABBREVIATIONS = {
    'api', 'sdk', 'oauth', 'jwt', 'json', 'xml', 'html', 'css', 'sql', 'nosql',
    'rest', 'soap', 'http', 'https', 'ftp', 'ssh', 'ssl', 'tls', 'tcp', 'udp',
    'dns', 'dhcp', 'vpn', 'lan', 'wan', 'wifi', 'bluetooth', 'usb', 'hdmi',
    'cpu', 'gpu', 'ram', 'ssd', 'hdd', 'bios', 'uefi', 'os', 'ui', 'ux',
    'ide', 'cli', 'gui', 'crm', 'erp', 'saas', 'paas', 'iaas', 'devops',
    'cicd', 'git', 'svn', 'npm', 'pip', 'maven', 'gradle', 'docker', 'kubernetes',
    'aws', 'gcp', 'azure', 'cdn', 'cms', 'seo', 'smtp', 'imap', 'pop3',
    'ldap', 'ad', 'sso', 'mfa', '2fa', 'rbac', 'acl', 'cors', 'csrf',
    'xss', 'ddos', 'dos', 'mitm', 'aes', 'rsa', 'sha', 'md5', 'base64',
    'utf8', 'ascii', 'unicode', 'regex', 'xpath', 'css3', 'html5', 'es6'
}

MICROSOFT_PRODUCTS = {
    'office', 'excel', 'word', 'powerpoint', 'outlook', 'onenote', 'access',
    'sharepoint', 'teams', 'skype', 'onedrive', 'azure', 'dynamics', 'exchange',
    'activedirectory', 'sqlserver', 'visualstudio', 'vscode', 'dotnet', 'csharp',
    'powershell', 'windows', 'xbox', 'surface', 'hololens', 'intune', 'sccm',
    'office365', 'microsoft365', 'azuread', 'powerbi', 'flow', 'powerapps',
    'sharepoint', 'yammer', 'planner', 'project', 'visio', 'publisher',
    'actionpack', 'biztalk', 'systemcenter', 'hyper-v', 'iis', 'asp.net'
}

TECHNICAL_TERMS_WHITELIST = TECHNICAL_ABBREVIATIONS | MICROSOFT_PRODUCTS

def is_technical_term(term):
    """
    Check if a term is in the technical terms whitelist.
    Performs case-insensitive matching.
    """
    if not term:
        return False
    return term.lower().strip() in TECHNICAL_TERMS_WHITELIST
