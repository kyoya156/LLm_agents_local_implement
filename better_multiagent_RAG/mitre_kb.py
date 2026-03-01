"""
MITRE ATT&CK Knowledge Base
Loads and indexes the full MITRE ATT&CK enterprise framework from a local STIX JSON file.
Download the file from: https://github.com/mitre/cti/raw/master/enterprise-attack/enterprise-attack.json
"""
import json
import os
from typing import Dict, List, Optional


# Maps log pattern flags (from log_parser.py) to MITRE technique IDs
FLAG_TO_TECHNIQUE = {
    "brute_force":           "T1110",
    "port_scan":             "T1046",
    "privilege_escalation":  "T1548",
    "lateral_movement":      "T1021",
    "data_exfiltration":     "T1041",
    "new_cron_job":          "T1053",
    "new_user_created":      "T1136",
    "suspicious_process":    "T1059",
    "firewall_disabled":     "T1562",
    "log_cleared":           "T1070",
    "dns_anomaly":           "T1071",
    "http_anomaly":          "T1071",
}

# Tactic ordering — used to predict the NEXT likely phase after a detected tactic
TACTIC_CHAIN = [
    "reconnaissance",
    "resource-development",
    "initial-access",
    "execution",
    "persistence",
    "privilege-escalation",
    "defense-evasion",
    "credential-access",
    "discovery",
    "lateral-movement",
    "collection",
    "command-and-control",
    "exfiltration",
    "impact",
]


class MitreKnowledgeBase:
    """
    Loads the full MITRE ATT&CK enterprise framework and provides
    lookup, keyword search, and next-phase prediction.
    """

    def __init__(self, stix_file: str = "enterprise-attack.json"):
        self.stix_file = stix_file
        self.techniques: Dict[str, Dict] = {}   # keyed by ATT&CK ID e.g. "T1110"
        self.available = False

        if os.path.exists(stix_file):
            self._load(stix_file)
        else:
            print(f"[MitreKB] WARNING: STIX file not found at '{stix_file}'.")
            print("[MitreKB] Using built-in fallback index (limited coverage).")
            self._load_fallback()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, path: str):
        """Parse the STIX 2.0 bundle and index all techniques."""
        print(f"[MitreKB] Loading MITRE ATT&CK from {path}...")
        with open(path, "r", encoding="utf-8") as f:
            bundle = json.load(f)

        count = 0
        for obj in bundle.get("objects", []):
            if obj.get("type") != "attack-pattern":
                continue
            if obj.get("x_mitre_deprecated", False):
                continue

            # Extract the ATT&CK ID (e.g. T1110, T1110.001)
            attack_id = ""
            url = ""
            for ref in obj.get("external_references", []):
                if ref.get("source_name") == "mitre-attack":
                    attack_id = ref.get("external_id", "")
                    url = ref.get("url", "")
                    break

            if not attack_id.startswith("T"):
                continue

            tactics = [
                phase["phase_name"]
                for phase in obj.get("kill_chain_phases", [])
                if phase.get("kill_chain_name") == "mitre-attack"
            ]

            self.techniques[attack_id] = {
                "id": attack_id,
                "name": obj.get("name", ""),
                "tactics": tactics,
                "description": obj.get("description", "")[:500],
                "detection": obj.get("x_mitre_detection", "No detection guidance available."),
                "platforms": obj.get("x_mitre_platforms", []),
                "data_sources": obj.get("x_mitre_data_sources", []),
                "url": url,
                "is_subtechnique": "." in attack_id,
            }
            count += 1

        print(f"[MitreKB] Loaded {count} techniques.")
        self.available = True

    def _load_fallback(self):
        """Hard-coded index for the most common techniques when the STIX file is absent."""
        fallback = [
            {
                "id": "T1110", "name": "Brute Force",
                "tactics": ["credential-access"],
                "description": "Adversaries may use brute force techniques to gain access.",
                "detection": "Monitor authentication logs for repeated failures from a single source.",
                "platforms": ["Linux", "Windows", "macOS"], "data_sources": ["Authentication logs"],
                "url": "https://attack.mitre.org/techniques/T1110", "is_subtechnique": False,
            },
            {
                "id": "T1046", "name": "Network Service Discovery",
                "tactics": ["discovery"],
                "description": "Adversaries may attempt to get a listing of services running on remote hosts.",
                "detection": "Monitor for port scanning activity.",
                "platforms": ["Linux", "Windows", "macOS"], "data_sources": ["Network traffic"],
                "url": "https://attack.mitre.org/techniques/T1046", "is_subtechnique": False,
            },
            {
                "id": "T1548", "name": "Abuse Elevation Control Mechanism",
                "tactics": ["privilege-escalation", "defense-evasion"],
                "description": "Adversaries may circumvent mechanisms designed to control elevated privileges.",
                "detection": "Monitor for sudo/su failures and unusual privilege changes.",
                "platforms": ["Linux", "Windows", "macOS"], "data_sources": ["Process logs", "Authentication logs"],
                "url": "https://attack.mitre.org/techniques/T1548", "is_subtechnique": False,
            },
            {
                "id": "T1021", "name": "Remote Services",
                "tactics": ["lateral-movement"],
                "description": "Adversaries may use valid accounts to log into a service.",
                "detection": "Monitor for remote login events between internal systems.",
                "platforms": ["Linux", "Windows", "macOS"], "data_sources": ["Authentication logs", "Network traffic"],
                "url": "https://attack.mitre.org/techniques/T1021", "is_subtechnique": False,
            },
            {
                "id": "T1041", "name": "Exfiltration Over C2 Channel",
                "tactics": ["exfiltration"],
                "description": "Adversaries may steal data by exfiltrating it over an existing C2 channel.",
                "detection": "Monitor for unusually large outbound transfers.",
                "platforms": ["Linux", "Windows", "macOS"], "data_sources": ["Network traffic"],
                "url": "https://attack.mitre.org/techniques/T1041", "is_subtechnique": False,
            },
            {
                "id": "T1053", "name": "Scheduled Task/Job",
                "tactics": ["persistence", "privilege-escalation", "execution"],
                "description": "Adversaries may abuse task scheduling to facilitate execution or persistence.",
                "detection": "Monitor scheduled task creation and modification.",
                "platforms": ["Linux", "Windows", "macOS"], "data_sources": ["Process logs"],
                "url": "https://attack.mitre.org/techniques/T1053", "is_subtechnique": False,
            },
            {
                "id": "T1136", "name": "Create Account",
                "tactics": ["persistence"],
                "description": "Adversaries may create an account to maintain access.",
                "detection": "Monitor for new user account creation.",
                "platforms": ["Linux", "Windows", "macOS"], "data_sources": ["Authentication logs"],
                "url": "https://attack.mitre.org/techniques/T1136", "is_subtechnique": False,
            },
            {
                "id": "T1059", "name": "Command and Scripting Interpreter",
                "tactics": ["execution"],
                "description": "Adversaries may abuse command and script interpreters to execute commands.",
                "detection": "Monitor for suspicious command-line activity.",
                "platforms": ["Linux", "Windows", "macOS"], "data_sources": ["Process logs"],
                "url": "https://attack.mitre.org/techniques/T1059", "is_subtechnique": False,
            },
            {
                "id": "T1562", "name": "Impair Defenses",
                "tactics": ["defense-evasion"],
                "description": "Adversaries may maliciously modify components of a system to hinder defenses.",
                "detection": "Monitor for disabling of security tools and firewall rules.",
                "platforms": ["Linux", "Windows", "macOS"], "data_sources": ["Process logs", "Firewall logs"],
                "url": "https://attack.mitre.org/techniques/T1562", "is_subtechnique": False,
            },
            {
                "id": "T1070", "name": "Indicator Removal",
                "tactics": ["defense-evasion"],
                "description": "Adversaries may delete or alter artifacts to remove evidence of their presence.",
                "detection": "Monitor for log clearing and artifact deletion.",
                "platforms": ["Linux", "Windows", "macOS"], "data_sources": ["Process logs"],
                "url": "https://attack.mitre.org/techniques/T1070", "is_subtechnique": False,
            },
            {
                "id": "T1071", "name": "Application Layer Protocol",
                "tactics": ["command-and-control"],
                "description": "Adversaries may communicate using OSI application layer protocols.",
                "detection": "Monitor for anomalous DNS/HTTP traffic patterns.",
                "platforms": ["Linux", "Windows", "macOS"], "data_sources": ["Network traffic"],
                "url": "https://attack.mitre.org/techniques/T1071", "is_subtechnique": False,
            },
        ]
        for t in fallback:
            self.techniques[t["id"]] = t
        self.available = True
        print(f"[MitreKB] Fallback index loaded ({len(self.techniques)} techniques).")

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_technique(self, technique_id: str) -> Optional[Dict]:
        """Return full details for a technique ID like 'T1110'."""
        return self.techniques.get(technique_id.upper())

    def get_techniques_for_flags(self, flags: List[str]) -> List[Dict]:
        """Given a list of log-parser flags, return matching MITRE technique dicts."""
        results = []
        seen = set()
        for flag in flags:
            tid = FLAG_TO_TECHNIQUE.get(flag)
            if tid and tid not in seen:
                technique = self.get_technique(tid)
                if technique:
                    results.append(technique)
                    seen.add(tid)
        return results

    def search_by_keyword(self, keyword: str, limit: int = 5) -> List[Dict]:
        """Find techniques whose name or description contains the keyword."""
        keyword = keyword.lower()
        matches = []
        for technique in self.techniques.values():
            if (keyword in technique["name"].lower() or
                    keyword in technique["description"].lower()):
                matches.append(technique)
        return matches[:limit]

    def get_techniques_by_tactic(self, tactic: str) -> List[Dict]:
        """Return all techniques that belong to a given tactic."""
        tactic = tactic.lower().replace(" ", "-")
        return [t for t in self.techniques.values() if tactic in t["tactics"]]

    def predict_next_phases(self, detected_tactics: List[str]) -> List[str]:
        """
        Given a list of already-detected tactics, return the next likely
        phases in the attack chain (up to 3).
        """
        if not detected_tactics:
            return []

        # Find the furthest tactic index in the chain
        indices = []
        for tactic in detected_tactics:
            tactic_norm = tactic.lower().replace(" ", "-")
            if tactic_norm in TACTIC_CHAIN:
                indices.append(TACTIC_CHAIN.index(tactic_norm))

        if not indices:
            return []

        max_idx = max(indices)
        next_phases = TACTIC_CHAIN[max_idx + 1: max_idx + 4]
        return next_phases

    def get_mitre_summary_for_flags(self, flags: List[str]) -> str:
        """
        Build a human-readable MITRE summary string for a set of log flags.
        Used by the analyzer to include in state.
        """
        techniques = self.get_techniques_for_flags(flags)
        if not techniques:
            return "No MITRE ATT&CK techniques mapped."

        lines = []
        all_tactics = []
        for t in techniques:
            lines.append(
                f"  • {t['id']} - {t['name']} "
                f"[{', '.join(t['tactics'])}]\n"
                f"    Detection: {t['detection'][:150]}"
            )
            all_tactics.extend(t["tactics"])

        next_phases = self.predict_next_phases(all_tactics)
        summary = "MITRE ATT&CK Mapping:\n" + "\n".join(lines)
        if next_phases:
            summary += f"\n\nPredicted Next Attack Phases: {', '.join(next_phases)}"

        return summary

    def to_vector_db_documents(self) -> tuple:
        """
        Return (documents, metadatas) ready to be ingested into ChromaDB.
        Each technique becomes a searchable document.
        """
        documents = []
        metadatas = []
        for tid, t in self.techniques.items():
            doc = (
                f"MITRE ATT&CK Technique: {t['name']} ({tid})\n"
                f"Tactics: {', '.join(t['tactics'])}\n"
                f"Platforms: {', '.join(t['platforms'])}\n"
                f"Description: {t['description']}\n"
                f"Detection: {t['detection']}\n"
                f"Data Sources: {', '.join(t['data_sources'])}"
            )
            meta = {
                "type": "mitre_technique",
                "technique_id": tid,
                "technique_name": t["name"],
                "tactics": ", ".join(t["tactics"]),
                "url": t["url"],
            }
            documents.append(doc)
            metadatas.append(meta)
        return documents, metadatas


if __name__ == "__main__":
    kb = MitreKnowledgeBase()  # uses fallback if no STIX file
    print(kb.get_mitre_summary_for_flags(["brute_force", "lateral_movement"]))
    print("\nNext phases after credential-access:")
    print(kb.predict_next_phases(["credential-access"]))