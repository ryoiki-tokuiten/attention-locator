import React, { startTransition, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react';

import { GlobalImagePreviewModal, RenderMathMarkdown } from '../RenderMathMarkdown';
import type { HealthPayload, SemanticWeight } from './types/semantic';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';
const INITIAL_MARKDOWN = `# Semantic Markdown Renderer

This prototype highlights the sentences that attract the most inter-sentence attention.

## Why this matters

Readers should be able to spot the core argument immediately, even when a document is dense and technical.

- Important claims should surface visually.
- Secondary explanations should stay readable without overpowering the page.
- Code and math should still render cleanly without semantic overlays.

> The goal is not just pretty markdown. The goal is markdown that helps people focus.

Inline math still works: $ISA(S_a, S_b) = \\max_{h \\in H} \\beta_h(S_a, S_b)$.

\`\`\`python
def semantic_weight(score: float) -> str:
    return "high" if score > 0.75 else "medium"
\`\`\`
`;

function sentenceCount(content: string) {
    const matches = content.match(/[.!?](?=\s|$)/g);
    return matches ? matches.length : 0;
}

function statusLabel(health: HealthPayload | null, isAnalyzing: boolean, analysisError: string | null) {
    if (analysisError) {
        return 'Semantic fallback';
    }
    if (isAnalyzing) {
        return 'Analyzing';
    }
    if (health?.ready) {
        return 'Model ready';
    }
    return 'Waiting for backend';
}

const ATTENTION_LEGEND_ITEMS = [
    {
        label: 'Lower attention',
        description: 'supporting or contextual text',
        className: 'attention-stop-cool',
    },
    {
        label: 'Medium attention',
        description: 'semantically connected text',
        className: 'attention-stop-mid',
    },
    {
        label: 'Higher attention',
        description: 'most salient model focus',
        className: 'attention-stop-hot',
    },
] as const;

const App: React.FC = () => {
    const [content, setContent] = useState(INITIAL_MARKDOWN);
    const [semanticWeights, setSemanticWeights] = useState<SemanticWeight[]>([]);
    const [health, setHealth] = useState<HealthPayload | null>(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [analysisError, setAnalysisError] = useState<string | null>(null);
    const deferredContent = useDeferredValue(content);
    const latestRequestRef = useRef(0);

    useEffect(() => {
        let isMounted = true;

        const loadHealth = async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/health`);
                if (!response.ok) {
                    throw new Error(`Backend health check failed with status ${response.status}`);
                }

                const payload = await response.json() as HealthPayload;
                if (isMounted) {
                    setHealth(payload);
                }
            } catch (error) {
                if (isMounted) {
                    setHealth({
                        ready: false,
                        modelName: 'prajjwal1/bert-tiny',
                        device: 'unavailable',
                        error: error instanceof Error ? error.message : 'Unable to reach backend.',
                    });
                }
            }
        };

        void loadHealth();

        return () => {
            isMounted = false;
        };
    }, []);

    useEffect(() => {
        if (!deferredContent.trim()) {
            startTransition(() => {
                setSemanticWeights([]);
                setAnalysisError(null);
            });
            setIsAnalyzing(false);
            return;
        }

        const controller = new AbortController();
        const requestId = latestRequestRef.current + 1;
        latestRequestRef.current = requestId;

        const timeoutId = window.setTimeout(async () => {
            setIsAnalyzing(true);
            setAnalysisError(null);

            try {
                const response = await fetch(`${API_BASE_URL}/analyze`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: deferredContent }),
                    signal: controller.signal,
                });

                if (!response.ok) {
                    const errorBody = await response.json().catch(() => null) as { detail?: string } | null;
                    throw new Error(errorBody?.detail || `Analysis failed with status ${response.status}`);
                }

                const payload = await response.json() as SemanticWeight[];
                if (latestRequestRef.current === requestId) {
                    startTransition(() => {
                        setSemanticWeights(payload);
                        setAnalysisError(null);
                    });
                }
            } catch (error) {
                if (controller.signal.aborted) {
                    return;
                }

                if (latestRequestRef.current === requestId) {
                    startTransition(() => {
                        setSemanticWeights([]);
                        setAnalysisError(error instanceof Error ? error.message : 'Analysis failed.');
                    });
                }
            } finally {
                if (!controller.signal.aborted && latestRequestRef.current === requestId) {
                    setIsAnalyzing(false);
                }
            }
        }, 800);

        return () => {
            controller.abort();
            window.clearTimeout(timeoutId);
        };
    }, [deferredContent]);

    const stats = useMemo(() => {
        const words = content.trim() ? content.trim().split(/\s+/).length : 0;
        const sentences = sentenceCount(content);
        const highlighted = semanticWeights.filter((weight) => weight.band !== 'low').length;

        return { words, sentences, highlighted };
    }, [content, semanticWeights]);

    const headerStatus = statusLabel(health, isAnalyzing, analysisError);

    return (
        <div className="app-shell">
            <div className="app-backdrop app-backdrop-primary" />
            <div className="app-backdrop app-backdrop-secondary" />

            <main className="workspace">
                <header className="hero">
                    <div>
                        <p className="eyebrow">Local sentence attention</p>
                        <h1>Semantic Attention-Weighted Markdown Renderer</h1>
                        <p className="hero-copy">
                            Your custom markdown renderer stays intact, while a lightweight local model decides which sentences deserve extra visual emphasis.
                        </p>
                    </div>

                    <div className="hero-metrics">
                        <div className="metric-card">
                            <span className="metric-label">Status</span>
                            <strong>{headerStatus}</strong>
                            <small>{health?.modelName ?? 'prajjwal1/bert-tiny'} on {health?.device ?? 'pending'}</small>
                        </div>
                        <div className="metric-card">
                            <span className="metric-label">Document</span>
                            <strong>{stats.words} words</strong>
                            <small>{stats.sentences} detected sentences</small>
                        </div>
                        <div className="metric-card">
                            <span className="metric-label">Highlighted</span>
                            <strong>{stats.highlighted}</strong>
                            <small>medium or high semantic bands</small>
                        </div>
                    </div>
                </header>

                <section className="panel-grid">
                    <section className="editor-panel glass-panel">
                        <div className="panel-header">
                            <div>
                                <p className="panel-label">Markdown input</p>
                                <h2>Compose or paste a document</h2>
                            </div>
                            <span className="panel-pill">{isAnalyzing ? 'Debounced analysis in progress' : 'Instant render, delayed weighting'}</span>
                        </div>

                        <textarea
                            className="editor-textarea"
                            value={content}
                            onChange={(event) => setContent(event.target.value)}
                            spellCheck={false}
                            placeholder="Paste markdown here..."
                        />
                    </section>

                    <section className="preview-panel glass-panel">
                        <div className="panel-header">
                            <div>
                                <p className="panel-label">Semantic preview</p>
                                <h2>Attention-aware rendering</h2>
                            </div>
                            <span className={`panel-pill ${analysisError ? 'panel-pill-warning' : ''}`}>
                                {analysisError ? 'Backend unavailable, rendering without weights' : `${semanticWeights.length} sentence weights loaded`}
                            </span>
                        </div>

                        {analysisError ? (
                            <div className="status-banner">
                                <strong>Semantic analysis is unavailable.</strong>
                                <span>{analysisError}</span>
                            </div>
                        ) : null}

                        {health?.error && !health.ready ? (
                            <div className="status-banner">
                                <strong>Backend health warning.</strong>
                                <span>{health.error}</span>
                            </div>
                        ) : null}

                        <div className="attention-legend" aria-label="Attention color legend">
                            <div className="attention-legend-bar" />
                            <div className="attention-legend-labels">
                                {ATTENTION_LEGEND_ITEMS.map((item) => (
                                    <div key={item.label} className="attention-legend-item">
                                        <span className={`attention-stop ${item.className}`} />
                                        <div>
                                            <strong>{item.label}</strong>
                                            <small>{item.description}</small>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="preview-surface">
                            <RenderMathMarkdown content={content} semanticWeights={semanticWeights} />
                        </div>
                    </section>
                </section>
            </main>

            <GlobalImagePreviewModal />
        </div>
    );
};

export default App;
