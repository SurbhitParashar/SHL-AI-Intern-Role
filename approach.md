# Approach

## Design

The service is a local-first FastAPI application with two endpoints: `GET /health` and `POST /chat`. The chat endpoint is stateless. Each request contains the complete message history, and the agent rebuilds the working context from that history on every call. This keeps refinement behavior simple and robust: if a user says “actually add personality tests,” the new shortlist is generated from the original role context plus the latest constraint.

The runtime never depends on live SHL network access. It loads `data/shl_catalog.json`, a cached Individual Test Solutions catalog. A separate `scripts/scrape_catalog.py` script can refresh the cache from SHL product pages, but deployment remains fast and deterministic.

## Retrieval and Ranking

Catalog items are represented with name, URL, test type, description, skills, job family, and optional metadata. Retrieval uses tokenization plus BM25-style lexical scoring over name, description, skills, job family, and type labels. The ranker adds explicit boosts for common hiring signals such as Java, Python, SQL, developer, stakeholder communication, graduate, manager, personality, cognitive ability, and situational judgment.

The recommender returns only items loaded from the cached catalog. It never invents names or URLs. When a requested type is clear, such as personality or cognitive ability, the system first selects matching test types and then fills the remaining shortlist with supporting assessments from the same role context.

## Conversation Policy

The agent has four main behaviors. It clarifies vague requests before recommending, recommends once role or skill context is present, refines by reconstructing context from full stateless history, and compares named assessments using only cached catalog fields. It refuses prompt-injection attempts and topics outside SHL assessment selection, including legal advice and general hiring advice.

The implementation avoids requiring an LLM key. This reduces latency and deployment risk under the evaluator’s 30-second timeout. An LLM could be added later only to polish response wording; ranking and catalog grounding should remain deterministic.

## Evaluation

I included tests for the hard evaluator constraints: exact `/health` schema, strict chat response shape, no recommendations on vague input, 1-10 recommendations for concrete hiring requests, catalog-backed URLs only, refinement with changed constraints, grounded comparison, and refusal of off-topic or prompt-injection input.

What did not work well for this style of assignment was an LLM-first design. It would make phrasing easier, but it increases hallucination risk, requires secrets during deployment, and makes behavior harder to defend in an interview. The current design favors predictable correctness and explainable trade-offs.
