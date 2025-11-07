Simple CLI app for a quick LLM prompting evaluation. The goal was to learn a bit about prompt evaluation, different types of metrics, pipeline and implementation, as well as basic API calls and routing. This project uses HuggingFace models only.


To run the script, use the following command:
python src/main.py --input data/prompts.jsonl --model <Desired model from models.txt> --output results/output.json (or define your own output file path)


Optional commands:
--max-tokens <max_tokens>
--temperature <temperature>
--top-p <top_p>
--repetition-penalty <repetition_penalty>
--retries <retries>
--timeout <timeout>

All requirements found in requirements.txt. Run "pip install -r requirements.txt" to install all dependencies. 
It is recommended to install dependencies in a virtual environment (venv) by running "python -m venv venv" and then activating the virtual environment before running the script. This will ensure that all dependencies are installed in a separate environment and will not interfere with other Python projects on your system.

This is a script to generate responses from a list of prompts using a language model. It is not intended to be a production-ready solution. It is recommended to use a more robust solution for production use cases. This script is provided as is and is not guaranteed to work as expected. Use at your own risk.

