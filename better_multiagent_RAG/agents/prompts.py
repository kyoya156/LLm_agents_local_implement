"""
Cybersecurity-adapted prompts for all agents.
"""

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
planner_prompt = """You are a senior SOC analyst using the ReAct framework. Analyze this security query.

                    Query: "{query}"

                    Prior Incident Context:
                    {memory_context}

                    THOUGHT: What type of security event or question is this? What do I already know from context?
                    ACTION: Choose ONE — log_analysis / threat_intel_lookup / incident_response / anomaly_detection / general_query
                    SEVERITY: Choose ONE — CRITICAL / HIGH / MEDIUM / LOW / UNKNOWN
                    PLAN: Write 3-5 numbered steps to investigate this query

                    Respond in this EXACT format (no extra text):
                    THOUGHT: <your reasoning>
                    ACTION: <one action keyword>
                    SEVERITY: <one severity keyword>
                    PLAN: <numbered steps>"""


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
retriever_intent_prompt = """You are a cybersecurity query classifier.

                            Query: "{query}"

                            Classify the query intent as ONE of:
                            - LOG_ANALYSIS     (user provides or asks about log entries)
                            - THREAT_INTEL     (asking about known attack techniques, malware, CVEs)
                            - INCIDENT_RESPONSE (asking what to do about an attack)
                            - ANOMALY_DETECTION (asking if something is suspicious or normal)
                            - GENERAL          (general cybersecurity question)

                            Respond with just ONE phrase from the list above."""

retriever_decision_prompt = """Query: {query}
                                Prior Incident Context: {memory_context}

                                Can this query be fully answered from the prior context, or do we need to search for new information?
                                Respond with just: use_memory OR search_knowledge"""


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------
analyzer_verification_prompt = """You are a threat analyst. Review the following evidence carefully.

                                Original Query / Alert: {query}

                                Retrieved Evidence (logs + CTI):
                                {docs_text}

                                Your tasks:
                                1. List 2-5 concrete facts that are DIRECTLY supported by the evidence above.
                                2. If the evidence is insufficient, respond with: INSUFFICIENT_DATA

                                Format each fact as a bullet starting with '-'.
                                Do NOT infer or assume information not present in the evidence."""

analyzer_mitre_prompt = """You are a cybersecurity expert. Given these detected attack flags and MITRE ATT&CK data:

                                Detected Flags: {flags}
                                MITRE Mapping: {mitre_summary}

                                Summarize in 2-3 sentences:
                                1. What attack stage this likely represents
                                2. What the attacker's probable goal is
                                3. The single most important defensive action to take right now"""


# ---------------------------------------------------------------------------
# Answer Generator
# ---------------------------------------------------------------------------
answer_generator_prompt = """You are a professional cybersecurity analyst writing an incident report.

                            Original Query: {query}
                            Query Type: {query_intent}
                            Overall Severity: {severity}
                            Quality of Evidence: {quality_score:.2f}/1.00

                            PRIOR INCIDENT CONTEXT:
                            {memory_context}

                            Verified Facts:
                            {facts_text}

                            MITRE ATT&CK Assessment:
                            {mitre_assessment}

                            IMPORTANT RULES:
                            - If Query Type is INCIDENT_RESPONSE: write actionable step-by-step response guidance. 
                            Reference the prior incident context to give specific advice (e.g. mention actual IPs to block).
                            Do NOT re-analyze the logs. Do NOT repeat the attack timeline. Focus ONLY on what to DO next.
                            - If Query Type is LOG_ANALYSIS: write a full threat assessment of what happened.
                            - Only use MITRE technique names and descriptions from the MITRE Assessment section above.
                            NEVER invent or guess technique names.

                            Write a structured report with:
                            1. SUMMARY (1-2 sentences, tailored to query type)
                            2. THREAT ASSESSMENT (current attack stage based on ALL known context)
                            3. MITRE ATT&CK CONTEXT (only techniques confirmed in the assessment above)
                            4. RECOMMENDED ACTIONS (specific, actionable, reference actual IPs/users from context)"""


# ---------------------------------------------------------------------------
# Memory Agent
# ---------------------------------------------------------------------------
extract_incident_facts_prompt = """From this security interaction, extract key incident facts to remember:

                                Analyst Query: {query}
                                System Response: {final_answer}

                                Extract facts in this format (or respond NONE if nothing noteworthy):
                                FACT: <concise fact about the incident, attacker, or system>
                                CATEGORY: <one of: attacker_ip / affected_user / attack_technique / timeline / system / recommendation>"""

summarize_prompt = """You are maintaining a running cybersecurity incident timeline.

                    {history_text}

                    Produce an UPDATED cumulative incident summary that:
                    1. Preserves ALL findings from the prior summary (if any)
                    2. Adds new findings from the new messages
                    3. Always includes: attacker IPs, usernames, attack techniques, timeline of events, current attack stage
                    4. Notes if the attack appears to be continuing or escalating

                    Keep it under 10 sentences. Be specific — include actual IPs, usernames, and technique names.

                    Updated Incident Timeline:"""


# ---------------------------------------------------------------------------
# Planner (kept for backwards compat alias)
# ---------------------------------------------------------------------------
planner_prompt_v1 = planner_prompt