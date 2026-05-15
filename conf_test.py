# Run the test suite (no Ollama or mic needed)
pytest tests/ -v

# Quick Ollama check from Python
py -c "
from core.engine import GemmaEngine
e = GemmaEngine()
print(e.check_connectivity())
"