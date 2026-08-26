// Renders assistant Markdown into sanitized, syntax-highlighted HTML.
// Self-contained (no CDN): marked + highlight.js + katex + DOMPurify are all
// bundled by Vite, consistent with SAGE's zero-egress requirement.

import DOMPurify from "dompurify";
import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import css from "highlight.js/lib/languages/css";
import json from "highlight.js/lib/languages/json";
import plaintext from "highlight.js/lib/languages/plaintext";
import python from "highlight.js/lib/languages/python";
import sql from "highlight.js/lib/languages/sql";
import typescript from "highlight.js/lib/languages/typescript";
import xml from "highlight.js/lib/languages/xml";
import yaml from "highlight.js/lib/languages/yaml";
import { Marked } from "marked";
import { markedHighlight } from "marked-highlight";
import markedKatex from "marked-katex-extension";

// `hljs/lib/core` (vs. the full bundle) only knows languages explicitly
// registered here — "plaintext" is the fallback used below for anything the
// model tags with an unrecognized/no language, so it has to be registered
// too, or that fallback itself throws instead of falling back.
hljs.registerLanguage("plaintext", plaintext);
hljs.registerLanguage("python", python);
hljs.registerLanguage("javascript", typescript); // close enough highlighting, avoids a second grammar
hljs.registerLanguage("typescript", typescript);
hljs.registerLanguage("bash", bash);
hljs.registerLanguage("shell", bash);
hljs.registerLanguage("json", json);
hljs.registerLanguage("yaml", yaml);
hljs.registerLanguage("html", xml);
hljs.registerLanguage("xml", xml);
hljs.registerLanguage("css", css);
hljs.registerLanguage("sql", sql);

export { hljs };

// A thrown exception here happens inside a Vue `computed()` (Markdown.vue),
// which surfaces as a render-flush error — that can abort the rest of that
// same reactive flush, including unrelated DOM updates queued in the same
// tick (e.g. the Send button's `:disabled` binding), making the whole
// Console look frozen from one bad code fence. Caught live: a fenced block
// with no language (or one hljs doesn't have registered) hit exactly this,
// because the "plaintext" fallback below wasn't itself a registered
// language. Fixed at the source (registered it), but every hljs call here
// is now also defensively wrapped so a future unknown-language edge case
// degrades to plain unhighlighted text instead of breaking the page again.
function escapeHtml(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function safeHighlight(code: string, lang: string): string {
  try {
    const language = hljs.getLanguage(lang) ? lang : "plaintext";
    return hljs.highlight(code, { language }).value;
  } catch {
    return escapeHtml(code);
  }
}

const marked = new Marked(
  markedHighlight({
    langPrefix: "hljs language-",
    highlight: safeHighlight,
  }),
  // $$...$$ (block) and $...$ (inline) rendered locally via KaTeX — no CDN,
  // matches how the agent is now prompted to write math.
  markedKatex({ throwOnError: false }),
);
marked.setOptions({ breaks: true, gfm: true });

// KaTeX emits MathML (<math>, <semantics>, <mrow>, ...) and sometimes SVG for
// certain glyphs; DOMPurify's default allowlist covers both namespaces
// already, this just keeps the handful of KaTeX-specific attributes/classes
// it relies on for layout (e.g. `mathvariant`, `aria-hidden`) from being
// stripped as unrecognized.
const KATEX_EXTRA_ATTR = ["mathvariant", "mathsize", "mathbackground", "columnalign", "aria-hidden", "encoding"];

// marked-katex-extension only recognizes $...$/$$...$$. Despite the system
// prompt asking for that, models trained on ChatGPT-style output reliably
// reach for \(...\)/\[...\] instead (confirmed live: gpt-oss wrote the
// quadratic formula as "\[ ... \]", never $$...$$) — without this, that
// LaTeX rendered as inert escaped text. Converted to $ delimiters before
// parsing, skipping fenced/inline code spans so a literal "\(" in a code
// sample isn't touched.
const FENCE_OR_INLINE_CODE = /(```[\s\S]*?```|`[^`\n]*`)/g;

function normalizeLatexDelimiters(text: string): string {
  return text
    .split(FENCE_OR_INLINE_CODE)
    .map((part, i) =>
      i % 2 === 1
        ? part
        : part.replace(/\\\[([\s\S]+?)\\\]/g, (_m, inner) => `$$${inner}$$`).replace(/\\\(([\s\S]+?)\\\)/g, (_m, inner) => `$${inner}$`),
    )
    .join("");
}

/** Full Markdown -> sanitized HTML, for a finished assistant message. */
export function renderMarkdown(text: string): string {
  try {
    const raw = marked.parse(normalizeLatexDelimiters(text), { async: false }) as string;
    return DOMPurify.sanitize(raw, { ADD_ATTR: ["target", ...KATEX_EXTRA_ATTR] });
  } catch {
    // Whatever broke, showing the raw text beats throwing inside a Vue
    // computed and taking the rest of that render flush down with it.
    return `<p>${escapeHtml(text)}</p>`;
  }
}

/** Highlights a standalone code snippet (for the code-viewer popup). */
export function highlightCode(code: string, language: string): string {
  return safeHighlight(code, language);
}
