import { bundledLanguages, createHighlighter, type BuiltinLanguage, type Highlighter } from 'shiki';

type HighlighterListener = () => void;

const listeners = new Set<HighlighterListener>();
const languageAliases: Record<string, string> = {
    js: 'javascript',
    jsx: 'jsx',
    ts: 'typescript',
    tsx: 'tsx',
    py: 'python',
    sh: 'bash',
    shell: 'bash',
    yml: 'yaml',
    md: 'markdown',
    tex: 'latex',
};

let highlighter: Highlighter | null = null;
let initStarted = false;

function escapeHtml(value: string): string {
    return value
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function currentTheme() {
    if (typeof document !== 'undefined' && document.body.classList.contains('light-mode')) {
        return 'github-light';
    }
    return 'github-dark';
}

async function bootHighlighter() {
    if (initStarted) {
        return;
    }

    initStarted = true;

    try {
        highlighter = await createHighlighter({
            themes: ['github-dark', 'github-light'],
            langs: ['plaintext', 'javascript', 'jsx', 'typescript', 'tsx', 'python', 'bash', 'markdown', 'json', 'yaml', 'html', 'css', 'latex'],
        });

        listeners.forEach((listener) => listener());
    } catch (error) {
        console.error('Failed to initialize Shiki:', error);
    } finally {
        listeners.clear();
    }
}

bootHighlighter();

export function onHighlighterReady(listener: HighlighterListener) {
    if (highlighter) {
        listener();
        return () => undefined;
    }

    listeners.add(listener);
    void bootHighlighter();

    return () => {
        listeners.delete(listener);
    };
}

export function isHighlighterReady() {
    return Boolean(highlighter);
}

export function resolveLanguage(language: string) {
    const normalized = language.trim().toLowerCase();
    if (!normalized) {
        return 'plaintext';
    }

    return languageAliases[normalized] || normalized;
}

export function isLanguageSupported(language: string) {
    const resolved = resolveLanguage(language);
    if (resolved === 'plaintext') {
        return true;
    }

    return Object.prototype.hasOwnProperty.call(bundledLanguages, resolved);
}

export function highlightCodeSync(code: string, language: string) {
    const resolved = resolveLanguage(language);

    if (!highlighter || !isLanguageSupported(resolved) || resolved === 'plaintext') {
        return escapeHtml(code);
    }

    try {
        return highlighter.codeToHtml(code, {
            lang: resolved as BuiltinLanguage,
            theme: currentTheme(),
        });
    } catch (error) {
        console.error('Shiki highlighting failed:', error);
        return escapeHtml(code);
    }
}
