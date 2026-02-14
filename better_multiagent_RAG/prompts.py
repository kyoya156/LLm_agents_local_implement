retriever_intent_prompt = """Analyze this query and classify its intent in one word:
Query: "{query}"

Is this asking for:
- DEFINITION (what is X?)
- EXPLANATION (how/why does X work?)
- COMPARISON (difference between X and Y?)
- FACT (specific factual information?)
- LIST (give me examples/types of X?)

Respond with just ONE word: DEFINITION, EXPLANATION, COMPARISON, FACT, or LIST."""

analyzer_verification_prompt = """You are a critical fact-checker. Given these documents and a query, extract ONLY the verified facts that directly answer the query.

Query: {query}
Intent: {query_intent}

Documents:
{docs_text}

Extract 2-3 specific, verified facts that answer the query. List them as bullet points.
If documents don't contain enough information, say "INSUFFICIENT_DATA".
Be critical - only include facts you're confident about."""

answer_generator_prompt = """You are a writer creating an answer for a user. You receive a query, its intent, a quality score, verified facts, and supporting context from documents.

Query: {query}
Query Type: {query_intent}
Writing Style: {answer_style}
Quality Score: {quality_score:.2f}

Verified Facts:
{facts_text}

Supporting Context:
{docs_context}

Write a {answer_style} answer that:
1. Directly answers the question
2. Uses the verified facts
3. Is 2-4 sentences long
4. Is friendly and easy to understand

When there is insufficient data to answer, respond with a sentence stating that.
Answer:"""