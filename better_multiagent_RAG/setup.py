"""
Setup Script — Cybersecurity RAG Database Initialisation

Ingests:
  1. CTI reports (text files in data/cti_reports/)
  2. Full MITRE ATT&CK technique catalogue
  3. Pre-parsed sample log events (from data/sample_logs/)

Run once before starting the CLI:
    python setup.py
"""
import os
import sys
from vector_db import VectorDBManager
from mitre_kb import MitreKnowledgeBase
from log_parser import LogParser


def ingest_cti_reports(db: VectorDBManager, reports_dir: str = "data/cti_reports"):
    """Load all .txt and .md files from the CTI reports folder."""
    if not os.path.isdir(reports_dir):
        print(f"[setup] CTI reports directory not found: {reports_dir} (skipping)")
        return 0

    files = [f for f in os.listdir(reports_dir) if f.endswith((".txt", ".md"))]
    if not files:
        print(f"[setup] No .txt/.md files in {reports_dir} (skipping)")
        return 0

    documents = []
    metadatas = []
    for fname in files:
        path = os.path.join(reports_dir, fname)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().strip()

        # Split large files into ~500-char chunks so embeddings are focused
        chunk_size = 500
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        for j, chunk in enumerate(chunks):
            if chunk.strip():
                documents.append(chunk)
                metadatas.append({
                    "type": "cti_report",
                    "source_file": fname,
                    "chunk": j,
                })

    if documents:
        db.add_documents(documents, metadatas)
        print(f"[setup] Ingested {len(documents)} CTI chunks from {len(files)} files.")
    return len(documents)


def ingest_mitre_techniques(db: VectorDBManager, stix_path: str = "enterprise-attack.json"):
    """Add all MITRE ATT&CK techniques to the vector DB."""
    kb = MitreKnowledgeBase(stix_path)
    documents, metadatas = kb.to_vector_db_documents()
    if documents:
        db.add_documents(documents, metadatas)
        print(f"[setup] Ingested {len(documents)} MITRE ATT&CK techniques.")
    return len(documents)


def ingest_sample_logs(db: VectorDBManager, logs_dir: str = "data/sample_logs"):
    """Parse and ingest sample log files."""
    if not os.path.isdir(logs_dir):
        print(f"[setup] Sample logs directory not found: {logs_dir} (skipping)")
        return 0

    files = [f for f in os.listdir(logs_dir) if f.endswith((".log", ".txt"))]
    if not files:
        print(f"[setup] No log files in {logs_dir} (skipping)")
        return 0

    parser = LogParser()
    all_docs = []
    all_meta = []

    for fname in files:
        path = os.path.join(logs_dir, fname)
        events = parser.parse_file(path)
        docs, metas = parser.events_to_documents(events)
        # Tag with source file
        for m in metas:
            m["source_file"] = fname
        all_docs.extend(docs)
        all_meta.extend(metas)
        summary = parser.summarize(events)
        print(f"[setup]   {fname}: {summary['suspicious_count']}/{summary['total_lines']} suspicious events")

    if all_docs:
        db.add_documents(all_docs, all_meta)
        print(f"[setup] Ingested {len(all_docs)} suspicious log events.")
    return len(all_docs)


def main():
    print("=" * 60)
    print("  Cybersecurity RAG — Database Setup")
    print("=" * 60)

    db = VectorDBManager(collection_name="cybersec_knowledge")
    starting_count = db.get_collection_count()
    print(f"[setup] Existing documents in DB: {starting_count}")

    if starting_count > 0:
        answer = input("[setup] DB already contains data. Re-ingest? (y/N): ").strip().lower()
        if answer != "y":
            print("[setup] Skipping ingestion.")
            print(f"[setup] DB ready with {starting_count} documents.")
            return True

    total = 0
    total += ingest_cti_reports(db)
    total += ingest_mitre_techniques(db)
    total += ingest_sample_logs(db)

    final_count = db.get_collection_count()
    print(f"\n[setup] Setup complete. Total documents in DB: {final_count} (+{final_count - starting_count} new)")

    if final_count == 0:
        print("\n[setup] WARNING: DB is empty!")
        print("  To add data:")
        print("  • Put CTI report .txt files in data/cti_reports/")
        print("  • Put log files (.log/.txt) in data/sample_logs/")
        print("  • Download enterprise-attack.json from github.com/mitre/cti")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)