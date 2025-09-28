# Environment Setup Instructions

## Watson API Key Configuration

This project requires an IBM Watson Natural Language Understanding API key to function properly.

### Setting up the WATSON_API_KEY Environment Variable

1. **Obtain your Watson API Key**:
   - Sign up for IBM Cloud at https://cloud.ibm.com/
   - Create a Natural Language Understanding service instance
   - Copy your API key from the service credentials

2. **Set the environment variable**:

   **Linux/macOS:**
   ```bash
   export WATSON_API_KEY="your_api_key_here"
   ```

   **Windows:**
   ```cmd
   set WATSON_API_KEY=your_api_key_here
   ```

   **For persistent configuration, add to your shell profile:**
   ```bash
   echo 'export WATSON_API_KEY="your_api_key_here"' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Verify the setup**:
   ```bash
   echo $WATSON_API_KEY
   ```

### Running the Application

Once the environment variable is set, you can run the main analysis:

```bash
python explore_watson3.py
```

If the `WATSON_API_KEY` environment variable is not set, the script will display an error message and exit gracefully.

### Security Note

The Watson API key has been moved from hardcoded values to environment variables for security reasons. Never commit API keys or other sensitive credentials to version control.
