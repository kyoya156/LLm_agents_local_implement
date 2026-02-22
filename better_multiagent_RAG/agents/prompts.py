retriever_decision_prompt = """Query: {query}
                            Memory Context: {memory_context}

                            THOUGHT: Can this be answered from memory, or do I need to search the knowledge base?
                            Respond with just: use_memory or search_knowledge"""

retriever_intent_prompt = """Analyze this query and classify its intent in one word:
                            Query: "{query}"

                            Respond with just ONE word: DEFINITION, EXPLANATION, COMPARISON, FACT, LIST OR PERSONAL."""


#                            Is this asking for:
#                            - DEFINITION (what is X?)
#                            - EXPLANATION (how/why does X work?)
#                            - COMPARISON (difference between X and Y?)
#                            - FACT (specific factual information?)
#                            - LIST (give me examples/types of X?)
#                            - PERSONAL (asking about user preferences or past interactions?)


analyzer_verification_prompt = """Query: {query}
                                Documents:
                                {docs_text}

                                Extract 2-3 verified facts that answer this query, be precise.
                                If insufficient data, respond with: INSUFFICIENT_DATA"""

answer_generator_prompt = """Query: {query}
                            Query Type: {query_intent}
                            Writing Style: {answer_style}
                            Quality Score: {quality_score:.2f}

                            Memory Context (use this to personalize):
                            {memory_context}

                            Verified Facts:
                            {facts_text}

                            IMPORTANT: Only use information explicitly stated in the verified facts above.
                            Do not infer, calculate, or assume any information not directly provided.
                            If the user shares personal details, acknowledge them warmly but do not use them to make calculations.
                            ...

                            Write an answer (2-4 sentences) that:
                            1. Answers the question directly
                            2. Uses verified facts
                            3. Considers the memory context about the user if relevant"""

summarize_prompt = """Summarize this conversation in 2-3 sentences. Focus on:
                    1. Main topics discussed
                    2. Key information provided
                    3. User's apparent interests

                    Conversation:
                    {history_text}

                    Summary:"""

planner_prompt = """Use the ReAct pattern to plan how to answer this query.

                    Query: "{query}"

                    Memory Context:
                    {memory_context}

                    Provide your reasoning in this format:
                    THOUGHT: What do I understand about this query? What's my goal?
                    ACTION: What specific action should I take? (search_knowledge / use_memory / ask_clarification)
                    PLAN: Step-by-step plan to answer this query

                    Respond in this exact format."""

extract_facts_prompt = """From this conversation, extract any facts about the user:
                        User: {query}
                        Assistant: {final_answer}

                        If there are facts to learn, respond with: FACT: [fact] 
                        And categorize the fact CATEGORY: [category]
                        Otherwise respond with: NONE"""