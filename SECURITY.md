# Security Policy

## Supported Versions

Private Agent is currently maintained by a single developer with assistance from AI-based development tools.

Security fixes are generally applied only to the latest released version.

| Version | Supported |
|----------|-----------|
| Latest Release | ✅ |
| Older Releases | ❌ |
| Development Branches | ❌ |

Users should always upgrade to the latest available release before reporting security issues.

---

## Reporting a Vulnerability

If you discover a security vulnerability, please report it privately.

### Preferred Method

Create a GitHub Security Advisory:

- GitHub → Security → Report a Vulnerability

### Alternative

Open an issue marked as:

```
[SECURITY] Brief description
```

**Do not include exploit details, secrets, credentials, or proof-of-concept payloads in a public issue.**

Instead, provide:

- A description of the issue
- Affected version
- Steps to reproduce
- Impact assessment
- Suggested remediation (optional)

---

## Response Expectations

Private Agent is maintained by a single contributor.

As a result:

- There are no guaranteed response times.
- Security reports are reviewed on a best-effort basis.
- Critical issues will be prioritized whenever possible.
- Fixes may be delayed based on maintainer availability.

Target response goals:

| Severity | Initial Response Goal |
|----------|----------------------|
| Critical | Within 7 days |
| High | Within 14 days |
| Medium | Within 30 days |
| Low | As time permits |

These targets are goals rather than service-level agreements.

---

## Security Scope

Private Agent is a local-first AI assistant that may interact with:

- Local files
- Local repositories
- Local databases
- Local AI runtimes
- User-configured APIs
- Model Context Protocol (MCP) servers
- External AI providers

Examples include:

- Ollama
- OpenAI-compatible endpoints
- LM Studio
- Azure OpenAI
- Anthropic
- Google Gemini
- Custom MCP integrations

Security reports involving these areas are in scope.

---

## Out of Scope

The following are generally considered out of scope:

- Self-inflicted exposure of API keys or credentials
- Vulnerabilities in third-party software
- Vulnerabilities already fixed in upstream dependencies
- Local denial-of-service attacks requiring system-level access
- Issues requiring modification of source code by the attacker
- Social engineering attacks
- Unsupported versions

---

## Security Model

Private Agent is designed with a local-first philosophy.

The project attempts to:

- Minimize unnecessary network access
- Avoid collecting telemetry by default
- Keep sensitive data on the user's machine whenever possible
- Require explicit user actions before potentially dangerous operations
- Support sandboxed execution where available

However:

### No Security Guarantees

Private Agent is an open-source project maintained by a single individual and assisted by AI-generated code contributions.

While reasonable efforts are made to improve security, the software:

- Has not undergone formal security certification
- Has not undergone professional penetration testing
- Has not been formally audited
- May contain undiscovered vulnerabilities
- May contain AI-generated implementation defects

Users should evaluate suitability for their environment before use.

---

## AI-Generated Code Notice

Portions of this project may be:

- Authored with AI assistance
- Refactored with AI assistance
- Reviewed with AI assistance

All code is reviewed prior to inclusion where practical, but AI-generated code can introduce:

- Logic flaws
- Security vulnerabilities
- Incorrect assumptions
- Incomplete validation

Users should exercise the same caution they would with any open-source project.

---

## User Responsibilities

Before using Private Agent in production or sensitive environments:

- Review the source code.
- Restrict file-system permissions where possible.
- Use least-privilege API keys.
- Run models and services in isolated environments when appropriate.
- Keep dependencies updated.
- Validate outputs before acting on them.
- Avoid exposing confidential or regulated data unless you understand the associated risks.

Recommended environments:

- Python virtual environments
- Docker containers
- Sandbox or VM-based development environments
- Non-administrative operating system accounts

---

## Dependency Security

Dependency security can be evaluated using tools such as:

```bash
pip-audit
```

and

```bash
bandit -r src
```

Users and contributors are encouraged to report vulnerable dependencies discovered through security scanning.

---

## Disclosure Policy

Reported vulnerabilities will be handled according to the following process:

1. Vulnerability received.
2. Validity assessed.
3. Severity evaluated.
4. Remediation developed.
5. Fix released.
6. Public disclosure made after a fix is available when appropriate.

The maintainer reserves the right to adjust this process based on issue severity and available time.

---

## Safe Harbor

Good-faith security research conducted to identify vulnerabilities in Private Agent is welcomed.

Please:

- Avoid accessing data belonging to others.
- Avoid disrupting systems or services.
- Avoid privacy violations.
- Avoid destructive testing on machines you do not own.

Researchers acting in good faith and following responsible disclosure practices are appreciated.

---

## Contact

For security concerns, please use GitHub Security Advisories whenever possible.

Thank you for helping improve the security of Private Agent.
