# Agentic-Gen-AI-Showcase

A comprehensive, modular framework showcasing the cutting edge of **Agentic AI** and **Generative AI**. This repository provides well-documented, self-contained implementations of key AI paradigms including reasoning-driven LLM agents, retrieval-augmented generation (RAG), multimodal vision-language understanding, and diffusion-based image generation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🎯 Overview

This framework demonstrates how to build intelligent AI systems that can:
- **Reason and Plan**: Use chain-of-thought reasoning to solve complex problems
- **Retrieve and Generate**: Combine knowledge retrieval with generation for accurate responses
- **See and Understand**: Process visual information and answer questions about images
- **Create and Synthesize**: Generate high-quality images from text descriptions

Each module is designed to be educational, extensible, and production-ready with clear documentation and examples.

## 🚀 Features

### 1. Reasoning-Driven LLM Agent (ReAct Pattern)
- **Chain-of-thought reasoning** with step-by-step problem solving
- **Tool use** with extensible tool framework
- **Self-reflection** and error correction
- **Planning and multi-step execution**
- Full ReAct (Reasoning + Acting) pattern implementation

### 2. Retrieval-Augmented Generation (RAG)
- **Document ingestion** with intelligent chunking
- **Vector embeddings** for semantic search
- **Similarity-based retrieval** with configurable top-k
- **Context-aware generation** for accurate Q&A
- Support for multiple vector stores

### 3. Multimodal Vision-Language Agent
- **Image understanding** and scene analysis
- **Visual Question Answering (VQA)**
- **Object detection** and localization
- **Image captioning** and description generation
- **Image comparison** for finding similarities/differences

### 4. Diffusion-Based Generative Models
- **Text-to-image generation**
- **Image-to-image transformation**
- **Inpainting** and outpainting
- **Multiple sampling methods** (DDPM, DDIM, Euler)
- **Quality presets** for different use cases

## 📁 Project Structure

```
Agentic-Gen-AI-Showcase/
├── modules/                          # Core AI modules
│   ├── reasoning_agent/              # ReAct pattern agent
│   │   ├── __init__.py
│   │   └── agent.py                  # Reasoning agent implementation
│   ├── rag_system/                   # RAG implementation
│   │   ├── __init__.py
│   │   └── rag.py                    # RAG system with vector store
│   ├── multimodal_agent/             # Vision-language agent
│   │   ├── __init__.py
│   │   └── multimodal.py             # Multimodal understanding
│   └── diffusion_generation/         # Diffusion models
│       ├── __init__.py
│       └── diffusion.py              # Image generation
├── examples/                         # Usage examples
│   ├── example_reasoning_agent.py
│   ├── example_rag_system.py
│   ├── example_multimodal_agent.py
│   └── example_diffusion_generation.py
├── docs/                             # Detailed documentation
│   ├── reasoning_agent.md
│   ├── rag_system.md
│   ├── multimodal_agent.md
│   └── diffusion_generation.md
├── requirements.txt                  # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/FLotfiGit/Agentic-Gen-AI-Showcase.git
cd Agentic-Gen-AI-Showcase

# Install dependencies
pip install -r requirements.txt
```

### Optional: Production Dependencies

For production use with real models, install additional dependencies:

```bash
# For RAG with real embeddings
pip install sentence-transformers faiss-cpu

# For multimodal with real models
pip install transformers torch torchvision pillow

# For diffusion models
pip install diffusers accelerate

# For LLM integration
pip install openai anthropic langchain
```

## 🎓 Quick Start

### 1. Reasoning Agent

```python
from modules.reasoning_agent import ReasoningAgent, Tool, get_example_tools

# Create agent with tools
agent = ReasoningAgent(tools=get_example_tools(), max_iterations=5)

# Solve a problem
result = agent.run(
    task="Calculate the result of (15 + 25) * 2",
    context="Use the calculator tool."
)

# Examine the reasoning process
for entry in result['history']:
    print(f"{entry['type']}: {entry.get('content', '')}")
```

### 2. RAG System

```python
from modules.rag_system import RAGSystem

# Initialize RAG
rag = RAGSystem(chunk_size=500, top_k=3)

# Ingest documents
documents = [
    {
        'doc_id': 'doc1',
        'text': 'Machine learning is a subset of AI...',
        'metadata': {'source': 'textbook'}
    }
]
rag.ingest_documents(documents)

# Query
result = rag.query("What is machine learning?")
print(result['answer'])
```

### 3. Multimodal Agent

```python
from modules.multimodal_agent import MultimodalAgent, ImageInput

# Initialize agent
agent = MultimodalAgent()

# Create image input
image = ImageInput(
    image_data=your_image,  # PIL Image or numpy array
    image_id="img_001",
    format="pil",
    metadata={}
)

# Ask questions about the image
answer = agent.visual_question_answering(
    image,
    "What objects are in this image?"
)
print(answer.text_response)
```

### 4. Diffusion Model

```python
from modules.diffusion_generation import DiffusionModel, GenerationConfig

# Create model
model = DiffusionModel()

# Configure generation
config = GenerationConfig(
    prompt="A serene lake at sunset with mountains",
    width=512,
    height=512,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42
)

# Generate image
result = model.generate(config)
print(f"Generated in {result.generation_time:.2f}s")
```

## 📚 Examples

Run the example scripts to see each module in action:

```bash
# Reasoning Agent examples
python examples/example_reasoning_agent.py

# RAG System examples
python examples/example_rag_system.py

# Multimodal Agent examples
python examples/example_multimodal_agent.py

# Diffusion Generation examples
python examples/example_diffusion_generation.py
```

## 📖 Documentation

Comprehensive documentation is available for each module:

- **[Reasoning Agent](docs/reasoning_agent.md)** - ReAct pattern, tool use, and planning
- **[RAG System](docs/rag_system.md)** - Document ingestion, retrieval, and generation
- **[Multimodal Agent](docs/multimodal_agent.md)** - Vision-language understanding and VQA
- **[Diffusion Generation](docs/diffusion_generation.md)** - Text-to-image generation and techniques

## 🎯 Use Cases

### Reasoning Agent
- Complex problem solving
- Task planning and execution
- Tool-augmented LLMs
- Multi-step reasoning
- Research assistants

### RAG System
- Knowledge-base Q&A
- Document search and retrieval
- Customer support
- Research assistance
- Context-aware chatbots

### Multimodal Agent
- Visual question answering
- Image understanding
- Accessibility (alt-text generation)
- Content moderation
- Visual search

### Diffusion Models
- Creative art generation
- Product design mockups
- Marketing content
- Concept art
- Style transfer

## 🔧 Customization

### Adding Custom Tools

```python
from modules.reasoning_agent import Tool

def my_custom_tool(param: str) -> str:
    # Your implementation
    return result

custom_tool = Tool(
    name="my_tool",
    description="What the tool does",
    function=my_custom_tool
)

agent.add_tool(custom_tool)
```

### Integrating Real Models

Each module includes documentation on integrating with production models:

- **LLMs**: OpenAI GPT-4, Anthropic Claude, open-source models
- **Embeddings**: OpenAI, Cohere, sentence-transformers
- **Vision Models**: CLIP, BLIP, YOLO, DETR
- **Diffusion Models**: Stable Diffusion, DALL-E, Midjourney APIs

See the documentation for integration examples.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This framework is inspired by and builds upon research in:
- ReAct: Reasoning and Acting in Language Models
- Retrieval-Augmented Generation (RAG)
- CLIP and multimodal learning
- Diffusion models and Stable Diffusion

## 📞 Contact

**Fatemeh Lotfi**
- GitHub: [@FLotfiGit](https://github.com/FLotfiGit)

## 🔮 Future Roadmap

- [ ] Integration with major LLM APIs (OpenAI, Anthropic)
- [ ] Production vector database support (Pinecone, Weaviate)
- [ ] Real vision model integration (GPT-4V, Claude 3)
- [ ] Stable Diffusion integration
- [ ] Multi-agent collaboration
- [ ] Video understanding
- [ ] 3D generation
- [ ] Web UI for demonstrations

---

**Star ⭐ this repository if you find it useful!**