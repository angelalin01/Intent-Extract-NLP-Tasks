# Environment Setup Instructions

## Watson API Key Configuration

This project requires an IBM Watson Natural Language Understanding API key to function properly.

### Setting up the Environment Variable

1. **Get your Watson API Key:**
   - Sign up for IBM Cloud at https://cloud.ibm.com/
   - Create a Natural Language Understanding service instance
   - Copy your API key from the service credentials

2. **Set the environment variable:**

   **On Linux/macOS:**
   ```bash
   export WATSON_API_KEY=your_actual_api_key_here
   ```

   **On Windows:**
   ```cmd
   set WATSON_API_KEY=your_actual_api_key_here
   ```

   **Using .env file (recommended for development):**
   ```bash
   cp .env.example .env
   # Edit .env file and replace 'your_watson_api_key_here' with your actual API key
   ```

3. **Verify the setup:**
   ```bash
   python -c "import os; print('API key set:', bool(os.getenv('WATSON_API_KEY')))"
   ```

### Security Notes

- Never commit your actual API key to version control
- The `.env` file is ignored by git to prevent accidental commits
- Use environment variables in production deployments
- Rotate your API keys regularly for security

### Error Handling

If the `WATSON_API_KEY` environment variable is not set, the application will:
- Display a clear error message
- Provide setup instructions
- Exit gracefully with status code 1

This prevents the application from running with invalid credentials and makes debugging easier.
