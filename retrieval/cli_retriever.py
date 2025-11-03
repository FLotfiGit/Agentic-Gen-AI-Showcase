"""Command-line tools to build or query a Retriever.

Usage examples:
  Build index from a text file (one doc per line):
    python retrieval/cli_retriever.py build --input data/corpus.txt --out outputs/myindex

  Query an existing index:
    python retrieval/cli_retriever.py query --index outputs/myindex --q "What is RAG?"
"""
import argparse
import json
from retrieval.retriever import Retriever
from pathlib import Path


def cmd_build(args):
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(args.input)
    texts = [l.strip() for l in in_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    r = Retriever(model_name=args.model)
    r.add_texts(texts)
    r.save(args.out)
    print(f"Built index with {len(texts)} documents and saved to {args.out}(.idx/.txt)")


def cmd_query(args):
    r = Retriever(model_name=args.model)
    r.load(args.index)
    res = r.query(args.q, k=args.k)
    out = {"query": args.q, "results": []}
    for dist, txt in res:
        out["results"].append({"distance": dist, "text": txt})
    p = Path("outputs")
    p.mkdir(parents=True, exist_ok=True)
    out_file = p / "retriever_results.jsonl"
    out_file.write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved query results to {out_file}")
    print(json.dumps(out, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    pbuild = sub.add_parser("build")
    pbuild.add_argument("--input", required=True, help="Text file with one document per line")
    pbuild.add_argument("--out", required=True, help="Base path to save index (no suffix)")
    pbuild.add_argument("--model", default="all-MiniLM-L6-v2")

    pq = sub.add_parser("query")
    pq.add_argument("--index", required=True, help="Base path to index (no suffix)")
    pq.add_argument("--q", required=True, help="Query string")
    pq.add_argument("--k", type=int, default=3)
    pq.add_argument("--model", default="all-MiniLM-L6-v2")

    args = parser.parse_args()
    if args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "query":
        cmd_query(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
