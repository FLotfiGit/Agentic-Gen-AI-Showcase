# Agentic-Gen-AI-Showcase
A curated collection of experiments and frameworks exploring the intersection of **Agentic AI** and **Generative AI** — including reasoning-driven LLM agents, retrieval-augmented generation (RAG), multimodal synthesis, and autonomous decision-making systems.

<p align="left">
		<a href="https://github.com/FLotfiGit/Agentic-Gen-AI-Showcase/actions/workflows/ci.yml"><img src="https://github.com/FLotfiGit/Agentic-Gen-AI-Showcase/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
	<img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python"/>
	<img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
	<img src="https://img.shields.io/badge/Status-Experimental-orange" alt="Status"/>
</p>
<p align="left">
  <a href="https://github.com/FLotfiGit/Agentic-Gen-AI-Showcase/actions/workflows/notebook-ci.yml"><img src="https://github.com/FLotfiGit/Agentic-Gen-AI-Showcase/actions/workflows/notebook-ci.yml/badge.svg" alt="notebook-ci"/></a>
</p>
</p>

---

## 🚀 Overview

This repository serves as a hands-on showcase of modern **agentic and generative intelligence**:
- **Agentic Reasoning:** Multi-step planning and goal-oriented agents.
- **Generative Modeling:** Text, image, and multimodal generation.
- **RAG & Tool Use:** Integrating retrieval, APIs, and dynamic memory.
- **Evaluation:** Benchmarking performance and reasoning quality.

---

## 🧩 Repository Structure


<details>
<summary><strong>Repository Structure</strong></summary>

<table>
	<tr><th>Folder</th><th>Description</th></tr>
	<tr><td><code>agents/</code></td><td>LLM agent frameworks (planning, reasoning)</td></tr>
	<tr><td><code>generative_models/</code></td><td>Diffusion, transformers, multimodal demos</td></tr>
	<tr><td><code>rag_systems/</code></td><td>Retrieval-augmented pipelines</td></tr>
	<tr><td><code>multimodal/</code></td><td>CLIP, vision-language, audio-text agents</td></tr>
	<tr><td><code>evaluation/</code></td><td>Benchmarking and visualization tools</td></tr>
	<tr><td><code>docs/</code></td><td>Notes, papers, and documentation</td></tr>
</table>
</details>


---

## ⚙️ Tech Stack

- **Python 3.10+**, **PyTorch**
- **Hugging Face Transformers**
- **LangChain / LlamaIndex**
- **Diffusers / CLIP / BLIP2 / Flamingo**
- **FAISS / Chroma / Weaviate Retrieval**

---

## 🔬 Current Focus

- Building *self-improving* LLM agents through reflection loops.  
- Unifying multimodal generation and decision-making.  
- Exploring *reasoning + generation fusion* (JEPA-style training).  

---

## 🧠 Example Notebooks

| Notebook | Description |
|-----------|-------------|
| `agents/agent_reasoning.ipynb` | Basic reasoning agent using GPT + memory |
| `rag_systems/rag_loop.ipynb` | Retrieval-augmented generation pipeline |
| `multimodal/multimodal_agent.ipynb` | CLIP + LLM cooperative reasoning |
| `generative_models/diffusion_control.ipynb` | Generative image synthesis with feedback |

---

## 🏁 Quickstart

1. Clone and set up environment
	- Copy `.env.example` to `.env` and fill in keys (optional)
	- Run setup
		- Run setup

3. Developer tools
		- Install pre-commit and enable hooks:

			```bash
			pip install pre-commit
			pre-commit install
			pre-commit run --all-files
			```

		- CI will run pre-commit automatically on pushes and pull requests.
	Optional commands:
	- make setup
	- source .venv/bin/activate

2. Open notebooks in VS Code and run cells.

Outputs are saved to `./outputs/`.

---

## 🤝 Contributions & Contact

Suggestions and collaborations are welcome.  

**Authors:** *Fatemeh Lotfi* and *Hossein Rajoli*  
**Research Area:** AI-driven wireless systems, agentic learning, and generative intelligence.  

---

> “Agents that think before they generate — intelligence that adapts.”
