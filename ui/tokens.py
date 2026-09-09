"""Central design tokens for the RAG-4i Streamlit frontend.

Single source of truth (design-brief discipline: no invented values at
usage sites). Light theme is default; DARK overrides apply under
prefers-color-scheme. All text/background pairs target WCAG AA (4.5:1).
"""

TOKENS = {
    "color": {
        "background": "#FFFFFF",
        "surface": "#F8FAFC",
        "surface-elevated": "#FFFFFF",
        "text": "#0F172A",
        "text-secondary": "#475569",
        "text-muted": "#64748B",
        "border": "#E2E8F0",
        "border-strong": "#CBD5E1",
        "accent": "#5F8575",
        "accent-hover": "#4C6E60",
        "accent-soft": "#EDF3F0",
        "accent-text": "#2F5233",
        "success": "#2F7D4F",
        "success-soft": "#EAF4EE",
        "warning": "#8A5A00",
        "warning-soft": "#FBF3DF",
        "error": "#B42318",
        "error-soft": "#FDECEA",
        "user-bubble": "#EFF4F1",
    },
    "dark": {
        "background": "#0F172A",
        "surface": "#1E293B",
        "surface-elevated": "#243349",
        "text": "#F1F5F9",
        "text-secondary": "#C3CEDD",
        "text-muted": "#94A3B8",
        "border": "#334155",
        "border-strong": "#475569",
        "accent": "#9DBFA9",
        "accent-hover": "#B4D2BE",
        "accent-soft": "#1D2B24",
        "accent-text": "#C4DDCC",
        "success": "#7BC79A",
        "success-soft": "#16281E",
        "warning": "#E3B341",
        "warning-soft": "#2C2410",
        "error": "#F08A80",
        "error-soft": "#2E1512",
        "user-bubble": "#1E2A33",
    },
    "space": {
        "xs": "0.25rem",
        "sm": "0.5rem",
        "md": "1rem",
        "lg": "1.5rem",
        "xl": "2.5rem",
        "section": "3.5rem",
    },
    "radius": {
        "sm": "6px",
        "md": "10px",
        "lg": "16px",
        "pill": "999px",
    },
    "type": {
        "family": '"Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
        "mono": '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace',
        "base-size": "1rem",
        "base-leading": "1.65",
        "measure": "46rem",
        "h1-size": "1.75rem",
        "h1-weight": "650",
        "h2-size": "1.2rem",
        "h2-weight": "600",
    },
    "shadow": {
        "sm": "0 1px 2px rgba(15, 23, 42, 0.05)",
        "md": "0 4px 12px rgba(15, 23, 42, 0.07)",
    },
    "motion": {
        "fast": "120ms ease-out",
        "base": "180ms ease-out",
    },
}


def css_variables(mode: str = "auto") -> str:
    """Render :root vars (light) plus dark-scheme overrides.

    mode "dark" forces dark values (used by the in-app theme toggle);
    any other value keeps the OS-preference media query default.
    """
    lines = [":root {"]
    for group in ("color", "space", "radius", "type", "shadow", "motion"):
        for key, value in TOKENS[group].items():
            lines.append(f"  --rag-{group}-{key}: {value};")
    lines.append("}")
    if mode == "dark":
        lines.append(":root {")
        for key, value in TOKENS["dark"].items():
            lines.append(f"  --rag-color-{key}: {value};")
        lines.append("}")
    else:
        lines.append("@media (prefers-color-scheme: dark) {")
        lines.append("  :root {")
        for key, value in TOKENS["dark"].items():
            lines.append(f"  --rag-color-{key}: {value};")
        lines.append("  }")
        lines.append("}")
    return "\n".join(lines)
