# Getting Started Guide

Welcome to the Agentic-Gen-AI-Showcase! This guide will help you get up and running quickly.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/FLotfiGit/Agentic-Gen-AI-Showcase.git
cd Agentic-Gen-AI-Showcase

# Install dependencies (only numpy is required for basic examples)
pip install numpy

# Or install all dependencies for full functionality
pip install -r requirements.txt
```

### 2. Run Examples

Each module comes with a complete example demonstrating its capabilities:

```bash
# Run all examples
python examples/example_reasoning_agent.py
python examples/example_rag_system.py
python examples/example_multimodal_agent.py
python examples/example_diffusion_generation.py
```

### 3. Explore the Modules

#### Reasoning Agent

```python
from modules.reasoning_agent import ReasoningAgent, Tool, get_example_tools

agent = ReasoningAgent(tools=get_example_tools())
result = agent.run("Solve a problem", "context")
```

#### RAG System

```python
from modules.rag_system import RAGSystem

rag = RAGSystem(chunk_size=500, top_k=3)
rag.ingest_documents([{'doc_id': 'doc1', 'text': 'content...'}])
result = rag.query("Your question?")
```

#### Multimodal Agent

```python
from modules.multimodal_agent import MultimodalAgent, ImageInput

agent = MultimodalAgent()
answer = agent.visual_question_answering(image, "What's in this image?")
```

#### Diffusion Model

```python
from modules.diffusion_generation import DiffusionModel, GenerationConfig

model = DiffusionModel()
config = GenerationConfig(prompt="A beautiful sunset")
result = model.generate(config)
```

## 📚 Learning Path

1. **Start with Examples**: Run the example scripts to see each module in action
2. **Read Documentation**: Check the detailed docs in the `docs/` directory
3. **Modify Examples**: Customize the examples for your use case
4. **Build Your Application**: Use the modules in your own projects

## 🎯 Common Use Cases

### Question Answering System
```python
# Combine RAG with reasoning agent
rag = RAGSystem()
rag.ingest_documents(knowledge_base)
# Use retrieved context with reasoning agent
```

### Visual Assistant
```python
# Use multimodal agent for visual tasks
agent = MultimodalAgent()
answer = agent.visual_question_answering(image, question)
```

### Content Generation
```python
# Generate images with diffusion models
pipeline = DiffusionPipeline()
images = pipeline.text_to_image("Your creative prompt")
```

## 🔧 Configuration

### Reasoning Agent
- `max_iterations`: Maximum reasoning steps (default: 10)
- `tools`: List of Tool objects

### RAG System
- `chunk_size`: Size of document chunks (default: 500)
- `chunk_overlap`: Overlap between chunks (default: 50)
- `top_k`: Number of documents to retrieve (default: 5)

### Multimodal Agent
- All components use mock implementations by default
- See documentation for production model integration

### Diffusion Model
- `num_inference_steps`: Number of denoising steps (default: 50)
- `guidance_scale`: Prompt adherence strength (default: 7.5)
- `width`, `height`: Image dimensions (default: 512x512)

## 🆘 Troubleshooting

### Import Errors
Make sure numpy is installed:
```bash
pip install numpy
```

### Module Not Found
Ensure you're running from the project root directory:
```bash
cd Agentic-Gen-AI-Showcase
python examples/example_reasoning_agent.py
```

### Production Use
Current implementations use mock models. For production:
1. Check the documentation for each module
2. Follow the integration guides for real models
3. Install additional dependencies as needed

## 📖 Next Steps

- Read the [README](README.md) for a complete overview
- Explore detailed documentation in [docs/](docs/)
- Study the [examples/](examples/) to understand usage patterns
- Check out the module implementations in [modules/](modules/)

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/FLotfiGit/Agentic-Gen-AI-Showcase/issues)
- **Discussions**: [GitHub Discussions](https://github.com/FLotfiGit/Agentic-Gen-AI-Showcase/discussions)

Happy coding! 🎉
