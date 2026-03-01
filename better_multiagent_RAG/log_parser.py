"""
Log Parser
Parses raw log text (syslog, auth.log, Windows Event, etc.) into
structured security events and flags suspicious patterns.
"""
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Pattern definitions
# Each entry: (flag_name, compiled_regex, severity)
# ---------------------------------------------------------------------------
SUSPICIOUS_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    ("brute_force",          re.compile(r"failed\s+password|authentication\s+failure|invalid\s+user|failed\s+login", re.I), "HIGH"),
    ("port_scan",            re.compile(r"SYN\s+INVALID|nmap|port\s+scan|connection\s+refused.*repeated", re.I),           "MEDIUM"),
    ("privilege_escalation", re.compile(r"sudo.*FAILED|sudo.*incorrect|su.*FAILED|permission\s+denied.*sudo", re.I),        "HIGH"),
    ("lateral_movement",     re.compile(r"Accepted\s+(password|publickey).*from\s+(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)", re.I), "MEDIUM"),
    ("data_exfiltration",    re.compile(r"bytes_out[=:\s]+([5-9]\d{6}|\d{7,})|UPLOAD.*\d{7,}", re.I),                     "CRITICAL"),
    ("new_cron_job",         re.compile(r"crontab|new\s+cron|CRON.*REPLACE|/etc/cron", re.I),                              "MEDIUM"),
    ("new_user_created",     re.compile(r"useradd|adduser|new\s+user|user\s+added", re.I),                                 "HIGH"),
    ("suspicious_process",   re.compile(r"eval\(|base64_decode|/bin/sh\s+-c|cmd\.exe\s+/c|powershell.*-enc", re.I),       "HIGH"),
    ("firewall_disabled",    re.compile(r"iptables.*FLUSH|ufw\s+disable|firewall.*stop|netfilter.*disabled", re.I),        "CRITICAL"),
    ("log_cleared",          re.compile(r"truncat.*log|>.*auth\.log|rm.*\.log|clear.*event\s+log", re.I),                  "CRITICAL"),
    ("dns_anomaly",          re.compile(r"NXDOMAIN.*\d{3,}|DNS.*tunnel|long\s+dns\s+query", re.I),                        "MEDIUM"),
    ("http_anomaly",         re.compile(r"(GET|POST).*(\.php\?.*=http|cmd=|exec=|;ls|;cat\s+/etc)", re.I),                "HIGH"),
]

# Severity ordering for sorting
SEVERITY_RANK = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}

# Regex for common timestamp formats
TIMESTAMP_PATTERNS = [
    re.compile(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})"),          # ISO 8601
    re.compile(r"([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})"),     # syslog: Jan  5 12:34:56
    re.compile(r"(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2})"),             # Apache: 01/Jan/2024:12:00:00
]

IP_PATTERN = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b")
USER_PATTERN = re.compile(r"user[=:\s]+(\w+)|for\s+(\w+)\s+from|invalid\s+user\s+(\w+)", re.I)


def _extract_timestamp(line: str) -> str:
    for pattern in TIMESTAMP_PATTERNS:
        m = pattern.search(line)
        if m:
            return m.group(1)
    return "unknown"


def _extract_ips(line: str) -> List[str]:
    return list(set(IP_PATTERN.findall(line)))


def _extract_user(line: str) -> Optional[str]:
    m = USER_PATTERN.search(line)
    if m:
        return next((g for g in m.groups() if g), None)
    return None


def _highest_severity(severities: List[str]) -> str:
    if not severities:
        return "LOW"
    return max(severities, key=lambda s: SEVERITY_RANK.get(s, 0))


class LogParser:
    """
    Parse raw log strings into structured security event dicts.
    """

    def parse_lines(self, log_text: str) -> List[Dict]:
        """
        Parse a multi-line log string.
        Returns a list of event dicts, one per non-empty line.
        """
        events = []
        for line in log_text.splitlines():
            line = line.strip()
            if not line:
                continue
            events.append(self._parse_line(line))
        return events

    def parse_file(self, filepath: str) -> List[Dict]:
        """Parse a log file from disk."""
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return self.parse_lines(f.read())

    def _parse_line(self, line: str) -> Dict:
        flags = []
        severities = []

        for flag_name, pattern, severity in SUSPICIOUS_PATTERNS:
            if pattern.search(line):
                flags.append(flag_name)
                severities.append(severity)

        return {
            "raw": line,
            "timestamp": _extract_timestamp(line),
            "ips": _extract_ips(line),
            "user": _extract_user(line),
            "flags": flags,
            "severity": _highest_severity(severities) if flags else "LOW",
            "is_suspicious": len(flags) > 0,
        }

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def get_suspicious_events(self, events: List[Dict]) -> List[Dict]:
        """Filter to only events that triggered at least one flag."""
        return [e for e in events if e["is_suspicious"]]

    def summarize(self, events: List[Dict]) -> Dict:
        """
        Return a high-level summary dict for a batch of parsed events.
        """
        suspicious = self.get_suspicious_events(events)
        flag_counts: Dict[str, int] = {}
        ip_counts: Dict[str, int] = {}
        severities: Dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}

        for e in events:
            severities[e["severity"]] = severities.get(e["severity"], 0) + 1
            for flag in e["flags"]:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
            for ip in e["ips"]:
                ip_counts[ip] = ip_counts.get(ip, 0) + 1

        top_flags = sorted(flag_counts.items(), key=lambda x: x[1], reverse=True)
        top_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_lines": len(events),
            "suspicious_count": len(suspicious),
            "severity_breakdown": severities,
            "top_flags": top_flags,
            "top_source_ips": top_ips,
        }

    def events_to_documents(self, events: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """
        Convert parsed events into (documents, metadatas) for ChromaDB ingestion.
        Only suspicious events are included.
        """
        documents = []
        metadatas = []
        for e in self.get_suspicious_events(events):
            doc = (
                f"Security Event [{e['severity']}]\n"
                f"Timestamp: {e['timestamp']}\n"
                f"Source IPs: {', '.join(e['ips']) or 'unknown'}\n"
                f"User: {e['user'] or 'unknown'}\n"
                f"Flags: {', '.join(e['flags'])}\n"
                f"Raw: {e['raw'][:300]}"
            )
            meta = {
                "type": "log_event",
                "severity": e["severity"],
                "flags": ", ".join(e["flags"]),
                "timestamp": e["timestamp"],
                "ips": ", ".join(e["ips"]),
            }
            documents.append(doc)
            metadatas.append(meta)
        return documents, metadatas


if __name__ == "__main__":
    sample = """
Jan  5 12:01:03 server sshd[1234]: Failed password for root from 192.168.1.105 port 22 ssh2
Jan  5 12:01:05 server sshd[1234]: Failed password for root from 192.168.1.105 port 22 ssh2
Jan  5 12:01:07 server sshd[1234]: Failed password for invalid user admin from 10.0.0.50 port 4321
Jan  5 12:05:00 server sudo: user1 : incorrect password ; TTY=pts/0 ; PWD=/home/user1
Jan  5 12:10:00 server sshd[5678]: Accepted password for deploy from 192.168.1.200 port 22
Jan  5 12:15:00 server CRON[9999]: new cron entry added by root
"""
    parser = LogParser()
    events = parser.parse_lines(sample)
    summary = parser.summarize(events)

    print("=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n=== Suspicious Events ===")
    for e in parser.get_suspicious_events(events):
        print(f"  [{e['severity']}] {e['flags']} | {e['raw'][:80]}")