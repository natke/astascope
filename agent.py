"""A tiny LangChain starter agent that touches most of LangChain's core features.

Run:  python agent.py            (reads ANTHROPIC_API_KEY from .env)
"""

from dotenv import load_dotenv

load_dotenv()  # load ANTHROPIC_API_KEY from .env before anything reads it

import ast
import json
import logging
import operator
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, PrivateAttr
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelRequest,
    SummarizationMiddleware,
    before_model,
    dynamic_prompt,
)
from langchain.tools import ToolRuntime, tool
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from rank_bm25 import BM25Okapi
from langgraph.types import Command

# --- 1. Chat model, with a callback that asks before each request ----------
class SendDeclined(Exception):
    """Raised when the user declines to send a request to Claude."""


class ConfirmBeforeSend(BaseCallbackHandler):
    """Ask before every request to Claude: agent turns, the explain chain and summaries."""

    raise_error = True  # let SendDeclined stop the run instead of only being logged

    def __init__(self):
        self.always = False

    def on_chat_model_start(self, serialized, messages, **kwargs):
        if self.always:
            return
        msgs = messages[0]
        print(f"  [ready to send {len(msgs)} messages; last ({msgs[-1].type}): {msgs[-1].text[:120]!r}]")
        answer = input("  Send to Claude? [y]es / [n]o / [a]lways: ").strip().lower()
        if answer == "a":
            self.always = True
        elif answer != "y":
            raise SendDeclined()


# Declining is expected, so don't let LangChain log it as a callback error.
logging.getLogger("langchain_core.callbacks.manager").addFilter(
    lambda record: "SendDeclined" not in record.getMessage())

model = ChatAnthropic(
    model="claude-opus-5-5",
    max_tokens=16000,
    output_config={"effort": "medium"},
    # Server-side fallback: if a safety classifier declines, the API retries on another model.
    betas=["server-side-fallback-2026-07-01"],
    model_kwargs={"fallbacks": "default"},
    callbacks=[ConfirmBeforeSend()],
)

# --- 2. Retrieval: documents + a custom retriever (no embeddings needed) ---
def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Retriever(BaseRetriever):
    """Keyword search over documents. Subclassing BaseRetriever makes it a Runnable."""

    docs: list[Document]
    k: int = 4
    _bm25: BM25Okapi = PrivateAttr()

    def model_post_init(self, __context) -> None:
        # Build the index once, not on every query.
        self._bm25 = BM25Okapi([_tokenize(d.page_content) for d in self.docs])

    @classmethod
    def from_documents(cls, docs: list[Document], k: int = 4) -> "BM25Retriever":
        return cls(docs=docs, k=k)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        return [self.docs[i] for i in ranked[: self.k] if scores[i] > 0]


DOCS_DIR = Path(__file__).parent / "docs"
DOC_SUFFIXES = {".md", ".txt", ".rst"}
CHUNK_CHARS = 1500


def load_docs(folder: Path) -> list[Document]:
    """Read the text files in folder (and its subfolders) and split them into chunks."""
    chunks = []
    for path in sorted(folder.rglob("*")) if folder.is_dir() else []:
        if path.suffix.lower() not in DOC_SUFFIXES or not path.is_file():
            continue
        source = str(path.relative_to(folder.parent))
        text = path.read_text(encoding="utf-8", errors="replace")
        # Group paragraphs into chunks of up to CHUNK_CHARS, so search
        # returns the relevant passage rather than a whole file.
        current = ""
        for para in re.split(r"\n\s*\n", text):
            if current and len(current) + len(para) > CHUNK_CHARS:
                chunks.append(Document(page_content=current.strip(), metadata={"source": source}))
                current = ""
            current += para + "\n\n"
        if current.strip():
            chunks.append(Document(page_content=current.strip(), metadata={"source": source}))
    return chunks


NOTES = [
    "LangChain agents are built with create_agent, which runs a model-and-tools loop on LangGraph.",
    "Middleware hooks into the agent loop: before_model, after_model, wrap_model_call, wrap_tool_call.",
    "A checkpointer gives the agent memory across turns, keyed by thread_id.",
    "LCEL composes runnables with the | operator, e.g. prompt | model | parser.",
]
DOC_CHUNKS = load_docs(DOCS_DIR)


def build_retriever() -> BM25Retriever:
    notes = [Document(page_content=n, metadata={"source": "notes"}) for n in NOTES]
    return BM25Retriever.from_documents(DOC_CHUNKS + notes)


retriever = build_retriever()


def reload_docs() -> None:
    """Re-read the docs folder and rebuild the search index."""
    global DOC_CHUNKS, retriever
    DOC_CHUNKS = load_docs(DOCS_DIR)
    retriever = build_retriever()


# --- 2b. Converting PDFs in papers/ to Markdown with marker-pdf ---------------
PAPERS_DIR = Path(__file__).parent / "papers"
PAPERS_OUT = DOCS_DIR / "papers"          # inside docs/, so the agent can search them
PAPERS_INDEX = PAPERS_OUT / "index.json"  # one entry per converted PDF
_marker_converter = None                  # loaded on first use; the models are large


def load_paper_index() -> dict[str, dict]:
    if not PAPERS_INDEX.exists():
        return {}
    return {e["filename"]: e for e in json.loads(PAPERS_INDEX.read_text(encoding="utf-8"))}


def save_paper_index(index: dict[str, dict]) -> None:
    PAPERS_OUT.mkdir(parents=True, exist_ok=True)
    entries = sorted(index.values(), key=lambda e: e["filename"].lower())
    PAPERS_INDEX.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _paper_title(markdown: str, pdf: Path) -> str:
    """Use the first heading in the converted text; fall back to the file name."""
    for line in markdown.splitlines():
        heading = re.match(r"#{1,3}\s+(.+)", line.strip())
        if heading:
            title = re.sub(r"[*_`]", "", heading.group(1)).strip()
            if len(title) > 15:  # skip short headings like "Article" or "Abstract"
                return title
    return pdf.stem


def convert_papers() -> None:
    """Convert new PDFs in papers/ to Markdown (with images) in docs/papers/."""
    global _marker_converter
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"), key=lambda p: p.name.lower()) if PAPERS_DIR.is_dir() else []
    if not pdfs:
        print(f"  No PDFs found in {PAPERS_DIR}")
        return
    index = load_paper_index()
    # A paper counts as converted if the index has it with the same file size.
    todo = [p for p in pdfs
            if index.get(p.name, {}).get("pdf_size_bytes") != p.stat().st_size]
    for p in pdfs:
        if p not in todo:
            print(f"  skip  {p.name} (already converted)")
    if not todo:
        print("  Nothing new to convert.")
        return

    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import save_output, text_from_rendered
    except ImportError:
        print("  marker-pdf is not installed. Run: pip install marker-pdf")
        return
    if _marker_converter is None:
        print("  Loading marker models (the first run downloads them)...")
        _marker_converter = PdfConverter(artifact_dict=create_model_dict())

    converted = 0
    for i, pdf in enumerate(todo, 1):
        print(f"  [{i}/{len(todo)}] converting {pdf.name} ...")
        out_dir = PAPERS_OUT / pdf.stem
        try:
            rendered = _marker_converter(str(pdf))
            out_dir.mkdir(parents=True, exist_ok=True)
            save_output(rendered, str(out_dir), pdf.stem)  # writes <stem>.md, images, metadata
            markdown, _, images = text_from_rendered(rendered)
        except Exception as e:
            print(f"        failed: {e}")
            continue
        index[pdf.name] = {
            "filename": pdf.name,
            "title": _paper_title(markdown, pdf),
            "pdf_size_bytes": pdf.stat().st_size,
            "date_converted": datetime.now().isoformat(timespec="seconds"),
            "output": str((out_dir / f"{pdf.stem}.md").relative_to(DOCS_DIR.parent)),
            "images": len(images),
        }
        save_paper_index(index)  # save after each paper so progress isn't lost
        converted += 1
        print(f"        done: {index[pdf.name]['title']!r} ({len(images)} images)")

    reload_docs()
    print(f"  Converted {converted} of {len(todo)} paper(s). Index: {PAPERS_INDEX.relative_to(DOCS_DIR.parent)}")


# --- 2c. Combining the reference lists of all converted papers ---------------
REFERENCES_FILE = PAPERS_OUT / "references.md"

# "References" as a heading, or in bold inside a paragraph (as in Lancet papers).
_REF_HEADING = re.compile(
    r"^#{1,6}\s*\**\s*(references|bibliography|literature cited)\s*\**\s*$"
    r"|\*\*(references|bibliography)\*\*", re.I | re.M)
# A reference number ("12" or "12.") at the start of a line or after the end of
# the previous reference, followed by the first author or title.
_REF_NUM = re.compile(r"(?:^|(?<=[.)\]]\s))(\d{1,3})\.?\s+(?=[\[\"'A-Za-zÀ-ɏ])", re.M)
# The authors part ends in initials ("Flegal KM"), "Jr"/"Sr", or "et al".
_AUTHORS = re.compile(r"(\bet al|\b[A-Z][A-Z-]{0,4}( (Jr|Sr|II|III))?)$")


def _reference_section(markdown: str) -> str:
    """The text after the last References heading, up to the next heading."""
    matches = list(_REF_HEADING.finditer(markdown))
    if not matches:
        return ""
    rest = markdown[matches[-1].end():]
    end = re.search(r"(?m)^#{1,6}\s", rest)
    return rest[: end.start()] if end else rest


def _clean_markdown(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)  # marker's <span id="page-..."> anchors
    # [text](url) -> text; the url may contain one level of parentheses
    text = re.sub(r"\[([^\]]*)\]\((?:[^()]|\([^()]*\))*\)", r"\1", text)
    text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)  # list markers
    text = text.replace("*", "")
    text = re.sub(r"\\([()\[\]_#.-])", r"\1", text)  # markdown escapes like \( in DOIs
    return re.sub(r"[ \t]+", " ", text).strip()  # keep line breaks: they mark new references


def split_references(markdown: str) -> list[tuple[int, str]]:
    """Return (number, text) for each numbered reference in a converted paper."""
    text = _clean_markdown(_reference_section(markdown))
    hits = [(int(m.group(1)), m.start(), m.end()) for m in _REF_NUM.finditer(text)]
    # Follow the numbering 1, 2, 3, ... so numbers inside a reference (volumes,
    # pages) aren't mistaken for new references. Allow one or two to be missing.
    chosen, expected, pos = [], 1, 0
    while True:
        hit = None
        for want in (expected, expected + 1, expected + 2):
            hit = next((h for h in hits if h[0] == want and h[1] >= pos), None)
            if hit:
                break
        if not hit:
            break
        chosen.append(hit)
        expected, pos = hit[0] + 1, hit[2]
    refs = []
    for k, (n, _, start) in enumerate(chosen):
        stop = chosen[k + 1][1] if k + 1 < len(chosen) else len(text)
        ref = re.sub(r"\s+", " ", text[start:stop]).strip()
        ref = re.split(r"\s(?:Copyright ©|©\s?\d{4})", ref)[0].strip()  # page footer after the last one
        refs.append((n, ref))
    return refs


def _reference_key(ref: str) -> str:
    """Key for spotting the same reference in two papers: its title, letters and digits only."""
    parts = re.split(r"[.?]\s+", ref, maxsplit=2)
    if len(parts) > 1 and _AUTHORS.search(parts[0]):
        title = parts[1]               # "Authors. Title. Journal ..."
    else:
        title = " ".join(parts[:2])    # an organisation or no authors
    key = re.sub(r"[^a-z0-9]+", "", title.lower())
    return key if len(key) >= 15 else re.sub(r"[^a-z0-9]+", "", ref.lower())


# One person in a Vancouver-style author list: "Flegal KM", "Kim B-Y", "D'Agostino RB Sr", "di Palmo E".
_PERSON = re.compile(r"^\S.{0,40}\s[A-Z][A-Z-]{0,4}(\s(Jr|Sr|II|III))?$")


def _reference_authors(ref: str) -> tuple[list[str], bool]:
    """The people listed as authors of a reference, and whether the list ends in "et al"."""
    first = re.split(r"[.?]\s+", ref, maxsplit=1)[0]
    if not _AUTHORS.search(first):
        return [], False  # an organisation, or no authors
    names = [n.strip() for n in first.split(",")]
    et_al = bool(names) and names[-1] == "et al"
    return [n for n in names if _PERSON.match(n)], et_al


def _author_key(name: str) -> str:
    """Match "Wu Y-K" with "Wu YK" and "Mannan H." with "Mannan H"."""
    return re.sub(r"[-.]", "", name).lower()


def author_stats(references: list[str]) -> dict:
    """Count unique authors over a list of (already de-duplicated) references."""
    refs_by_author: dict[str, int] = Counter()
    spelling: dict[str, str] = {}
    with_people = truncated = 0
    for ref in references:
        names, et_al = _reference_authors(ref)
        with_people += bool(names)
        truncated += et_al
        for key in {_author_key(n): n for n in names}:  # count each person once per reference
            refs_by_author[key] += 1
        for n in names:
            spelling.setdefault(_author_key(n), n)
    # Same surname, where one set of initials starts the other ("Puhl R" / "Puhl RM"):
    # often the same person, but not always, so list them instead of merging.
    by_surname: dict[str, list[str]] = {}
    for key in sorted(refs_by_author):
        surname, initials = key.rsplit(" ", 1)
        by_surname.setdefault(surname, []).append(initials)
    possible_same = []
    for surname, inits in by_surname.items():
        group = sorted({i for i in inits for j in inits if i != j and (i.startswith(j) or j.startswith(i))})
        if group:
            possible_same.append([spelling[f"{surname} {i}"] for i in group])
    return {
        "possible_same": possible_same,
        "unique": len(refs_by_author),
        "listings": sum(refs_by_author.values()),
        "with_people": with_people,
        "truncated": truncated,
        "single_reference": sum(c == 1 for c in refs_by_author.values()),
        "top": [(spelling[k], c) for k, c in sorted(refs_by_author.items(), key=lambda kc: (-kc[1], kc[0]))[:20]],
    }


def collect_references() -> None:
    """Write the combined, de-duplicated references of all converted papers to references.md."""
    papers = [e for e in load_paper_index().values() if (DOCS_DIR.parent / e["output"]).exists()]
    if not papers:
        print("  No converted papers found. Run /convert first.")
        return
    merged: dict[str, dict] = {}  # key -> {"text": ..., "cited_by": [(title, number)]}
    for paper in sorted(papers, key=lambda e: e["title"].lower()):
        refs = split_references((DOCS_DIR.parent / paper["output"]).read_text(encoding="utf-8"))
        print(f"  {len(refs):4} references  {paper['title'][:70]}")
        for n, ref in refs:
            entry = merged.setdefault(_reference_key(ref), {"text": ref, "cited_by": []})
            if len(ref) > len(entry["text"]):  # keep the fullest version, e.g. all authors
                entry["text"] = ref
            entry["cited_by"].append((paper["title"], n))

    total = sum(len(e["cited_by"]) for e in merged.values())
    shared = sum(len({t for t, _ in e["cited_by"]}) > 1 for e in merged.values())
    entries = sorted(merged.values(), key=lambda e: e["text"].lower())
    authors = author_stats([e["text"] for e in entries])
    lines = [
        "# References from all papers",
        "",
        f"{len(merged)} unique references from {len(papers)} papers ({total} in total, "
        f"{shared} cited by more than one paper). Generated {datetime.now():%Y-%m-%d %H:%M} by /references.",
        "",
        "## Authors",
        "",
        f"- **{authors['unique']} unique authors** across the {authors['with_people']} references "
        f"that name people as authors ({authors['listings']} author listings in total).",
        f"- {authors['single_reference']} authors appear on only one reference.",
        f"- {len(merged) - authors['with_people']} references have an organisation as author, "
        "or no author, and aren't counted.",
        f"- {authors['truncated']} references shorten their author list to \"et al\", so authors "
        "hidden by \"et al\" aren't counted.",
        *([f"- Counted separately but possibly the same person (same surname, shorter initials): "
           + "; ".join(" / ".join(g) for g in authors["possible_same"]) + "."]
          if authors["possible_same"] else []),
        "",
        "Authors on the most references:",
        "",
        "| Author | References |",
        "|---|---|",
        *[f"| {name} | {count} |" for name, count in authors["top"]],
        "",
        "## References",
        "",
    ]
    for i, entry in enumerate(entries, 1):
        cited = "; ".join(f"{title} (ref {n})" for title, n in entry["cited_by"])
        lines += [f"{i}. {entry['text']}", f"   - Cited by: {cited}", ""]
    REFERENCES_FILE.write_text("\n".join(lines), encoding="utf-8")
    reload_docs()  # make the combined list searchable too
    print(f"  {len(merged)} unique references ({total} in total, {shared} shared).")
    print(f"  {authors['unique']} unique authors; most references: "
          + ", ".join(f"{name} ({count})" for name, count in authors["top"][:5]))
    print(f"  Saved to {REFERENCES_FILE.relative_to(DOCS_DIR.parent)}")


# --- 3. An LCEL chain used inside a tool -----------------------------------
eli5_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Explain things to a five-year-old in two sentences."),
        ("human", "{topic}"),
    ])
    | model
    | StrOutputParser()
)


# --- 3b. One-line summaries of the converted papers --------------------------
SUMMARY_CHARS = 60_000  # send at most this much of each paper (title, abstract, main text)

summary_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Summarize the research paper in one sentence of at most 30 words: "
                   "what was studied and the main finding or conclusion. "
                   "Reply with the sentence only."),
        ("human", "Title: {title}\n\n{text}"),
    ])
    | model
    | StrOutputParser()
)


def summarize_papers() -> None:
    """Print a one-line summary of each converted paper, asking Claude for any not yet summarized."""
    index = load_paper_index()
    papers = [e for e in index.values() if (DOCS_DIR.parent / e["output"]).exists()]
    if not papers:
        print("  No converted papers found. Run /convert first.")
        return
    for paper in sorted(papers, key=lambda e: e["title"].lower()):
        if not paper.get("summary"):
            markdown = (DOCS_DIR.parent / paper["output"]).read_text(encoding="utf-8")
            refs = list(_REF_HEADING.finditer(markdown))
            body = markdown[: refs[-1].start()] if refs else markdown  # drop the reference list
            try:
                summary = summary_chain.invoke({"title": paper["title"], "text": body[:SUMMARY_CHARS]})
            except SendDeclined:
                print(f"  skip  {paper['title']} (not sent)")
                continue
            paper["summary"] = " ".join(summary.split())
            save_paper_index(index)  # cache it; /convert replaces the entry, so a reconverted paper is summarized again
        print(f"  - {paper['title']}: {paper['summary']}")


# --- 4. Runtime context (per-invocation data the tools and prompt can read) --
@dataclass
class Context:
    user_name: str


# --- 5. Tools ----------------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression like '2 * (3 + 4) ** 2'."""
    return str(_eval(ast.parse(expression, mode="eval").body))


@tool
def current_time() -> str:
    """Get the current local date and time."""
    return datetime.now().isoformat(timespec="seconds")


@tool
def search_notes(query: str) -> str:
    """Search the knowledge base: the documents in the docs folder plus saved notes.
    Uses keyword matching, so include the specific words likely to appear in the text."""
    results = retriever.invoke(query)
    if not results:
        return "No matches."
    return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in results)


@tool
def explain_simply(topic: str) -> str:
    """Explain a topic in very simple terms."""
    return eli5_chain.invoke({"topic": topic})


@tool
def save_note(text: str, runtime: ToolRuntime[Context]) -> str:
    """Save a note to the knowledge base. Requires human approval."""
    NOTES.append(f"{runtime.context.user_name}: {text}")
    global retriever
    retriever = build_retriever()
    return "Saved."


# --- 6. Middleware -----------------------------------------------------------
@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
    return (f"You are a friendly assistant helping {request.runtime.context.user_name}. "
            "Use tools when they help. For questions about the user's documents, "
            "use search_notes and cite the [source] file. Keep answers short.")


@before_model
def log_turn(state, runtime) -> None:
    print(f"  [model call #{sum(m.type == 'ai' for m in state['messages']) + 1}]")


# --- 7. Structured output ----------------------------------------------------
class Answer(BaseModel):
    answer: str = Field(description="The reply to the user")
    tools_used: list[str] = Field(description="Names of tools used this turn")


# --- 8. Assemble the agent ---------------------------------------------------
agent = create_agent(
    model,
    tools=[calculator, current_time, search_notes, explain_simply, save_note],
    middleware=[
        personalized_prompt,
        log_turn,
        HumanInTheLoopMiddleware(interrupt_on={"save_note": True}),
        SummarizationMiddleware(model, trigger=("messages", 40), keep=("messages", 10)),
        ModelCallLimitMiddleware(run_limit=10),
    ],
    response_format=Answer,
    context_schema=Context,
    # Allow only our own Answer type (plus LangGraph's built-in safe types) to be
    # loaded back from saved state.
    checkpointer=InMemorySaver(serde=JsonPlusSerializer(
        allowed_msgpack_modules=[(Answer.__module__, Answer.__name__)])),
)


# --- 9. Chat loop: streaming, memory (thread_id), and interrupt/resume -------
def run(payload, config, context):
    for chunk in agent.stream(payload, config, context=context, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                for action in update[0].value["action_requests"]:
                    ok = input(f"  Approve {action['name']}({action['args']})? [y/N] ").lower() == "y"
                    decision = {"type": "approve"} if ok else {"type": "reject", "message": "User declined."}
                    return run(Command(resume={"decisions": [decision]}), config, context)
            for msg in (update or {}).get("messages", []):
                for call in getattr(msg, "tool_calls", []):
                    print(f"  -> {call['name']}({call['args']})")
    return agent.get_state(config).values.get("structured_response")


# --- 10. Commands: handled locally; only /summarize sends anything to Claude --
HELP = """  /help     show this list
  /notes    list the notes in the knowledge base
  /docs     list the files loaded from the docs folder
  /convert  convert new PDFs in the papers folder to text (skips ones already done)
  /references  combine the reference lists of all converted papers, without duplicates
  /summarize   one-line summary of each converted paper (asks Claude; results are cached)
  /history  show this conversation's messages
  /new      start a fresh conversation
  /exit     quit (also /quit, Ctrl-C or Ctrl-D)"""


def handle_command(cmd, config):
    """Run a /command. Returns False when the user wants to quit."""
    if cmd in ("/exit", "/quit"):
        return False
    if cmd == "/help":
        print(HELP)
    elif cmd == "/notes":
        print("\n".join(f"  - {n}" for n in NOTES))
    elif cmd == "/docs":
        counts = Counter(d.metadata["source"] for d in DOC_CHUNKS)
        if not counts:
            print(f"  No documents loaded. Put .md, .txt or .rst files in {DOCS_DIR}")
        for source, n in counts.items():
            print(f"  - {source} ({n} chunk{'s' if n != 1 else ''})")
    elif cmd == "/convert":
        convert_papers()
    elif cmd == "/references":
        collect_references()
    elif cmd == "/summarize":
        summarize_papers()
    elif cmd == "/history":
        messages = agent.get_state(config).values.get("messages", [])
        if not messages:
            print("  (empty)")
        for m in messages:
            calls = ", ".join(c["name"] for c in getattr(m, "tool_calls", []))
            print(f"  {m.type}: {m.text[:200] or f'(calls {calls})'}")
    elif cmd == "/new":
        thread = config["configurable"]["thread_id"]
        config["configurable"]["thread_id"] = str(int(thread) + 1)
        print("  Started a new conversation.")
    else:
        print(f"  Unknown command {cmd}. Type /help for the list.")
    return True


def main():
    context = Context(user_name=input("Your name: ").strip() or "friend")
    config = {"configurable": {"thread_id": "1"}}
    files = len({d.metadata["source"] for d in DOC_CHUNKS})
    print(f"Loaded {files} documents ({len(DOC_CHUNKS)} chunks) from {DOCS_DIR.name}/.")
    print("Ask me anything. Type /help for commands.")
    while True:
        text = input("\nyou> ").strip()
        if not text:
            continue
        if text.startswith("/"):
            if not handle_command(text.split()[0].lower(), config):
                break
            continue
        try:
            result = run({"messages": [{"role": "user", "content": text}]}, config, context)
        except SendDeclined:
            print("  Cancelled. Nothing more was sent to Claude.")
            continue
        if result:
            print(f"bot> {result.answer}\n     (tools: {', '.join(result.tools_used) or 'none'})")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        pass
