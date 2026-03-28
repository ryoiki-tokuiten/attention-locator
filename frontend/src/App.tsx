import React, { startTransition, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react';

import { GlobalImagePreviewModal, RenderMathMarkdown } from '../RenderMathMarkdown';
import AttentionExplorer, { type AttentionTab } from './AttentionExplorer';
import type {
    DocumentAttentionPayload,
    HealthPayload,
    SemanticBand,
    SemanticWeight,
    SentencePairAttentionPayload,
} from './types/semantic';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';
const INITIAL_MARKDOWN = `# SAVIS Attention Markdown

This prototype renders markdown with sentence-level attention emphasis and lets you inspect the underlying attention patterns.

## What changed

- The semantic renderer still highlights salient sentences.
- Sentence attention now comes from a full-context causal model pass.
- You can switch from the rendered view into sentence-level and token-level attention inspection.

> The goal is not just pretty markdown. The goal is seeing how the model routes attention through the document.

Inline math still works: $ISA(S_a, S_b) = \\max_{h \\in H} \\beta_h(S_a, S_b)$.
`;

const VIEW_TABS: Array<{ id: AttentionTab; label: string }> = [
    { id: 'rendered', label: 'Rendered' },
    { id: 'sentence', label: 'Sentence view' },
    { id: 'token', label: 'Token view' },
];

function clamp(value: number, min = 0, max = 1) {
    return Math.min(max, Math.max(min, value));
}

function asFiniteNumber(value: unknown, fallback = 0) {
    const numeric = typeof value === 'number' ? value : Number(value);
    return Number.isFinite(numeric) ? numeric : fallback;
}

function asString(value: unknown, fallback = '') {
    return typeof value === 'string' ? value : fallback;
}

function asStringArray(value: unknown): string[] {
    if (!Array.isArray(value)) {
        return [];
    }
    return value.map((item) => asString(item)).filter(Boolean);
}

function asSemanticBand(value: unknown): SemanticBand {
    return value === 'high' || value === 'medium' || value === 'low' ? value : 'medium';
}

function sanitizeSemanticWeight(item: unknown, fallbackIndex: number): SemanticWeight {
    const candidate = item && typeof item === 'object' ? item as Record<string, unknown> : {};
    const sentence = asString(candidate.sentence);
    const normalizedSentence = asString(candidate.normalizedSentence, sentence.trim().toLowerCase());

    return {
        index: asFiniteNumber(candidate.index, fallbackIndex),
        sentence,
        normalizedSentence,
        weight: clamp(asFiniteNumber(candidate.weight, 0.5)),
        band: asSemanticBand(candidate.band),
    };
}

function sanitizeSemanticWeights(payload: unknown): SemanticWeight[] {
    if (!Array.isArray(payload)) {
        return [];
    }

    return payload.map((item, index) => sanitizeSemanticWeight(item, index));
}

function sanitizeAttentionMatrix(payload: unknown) {
    if (!Array.isArray(payload)) {
        return [];
    }

    return payload.map((row) => (
        Array.isArray(row)
            ? row.map((value) => clamp(asFiniteNumber(value, 0)))
            : []
    ));
}

function sanitizeDocumentAttentionPayload(payload: unknown): DocumentAttentionPayload {
    const candidate = payload && typeof payload === 'object' ? payload as Record<string, unknown> : {};
    const sentenceWeights = sanitizeSemanticWeights(candidate.sentenceWeights);

    return {
        analysisText: asString(candidate.analysisText),
        tokenCount: asFiniteNumber(candidate.tokenCount, 0),
        sentenceCount: asFiniteNumber(candidate.sentenceCount, sentenceWeights.length),
        sentenceWeights,
        sentenceAttention: sanitizeAttentionMatrix(candidate.sentenceAttention),
        warnings: asStringArray(candidate.warnings),
    };
}

function sanitizeSentencePairAttentionPayload(payload: unknown): SentencePairAttentionPayload {
    const candidate = payload && typeof payload === 'object' ? payload as Record<string, unknown> : {};

    return {
        sentenceXIndex: asFiniteNumber(candidate.sentenceXIndex, 0),
        sentenceYIndex: asFiniteNumber(candidate.sentenceYIndex, 0),
        sentenceX: asString(candidate.sentenceX),
        sentenceY: asString(candidate.sentenceY),
        columnTokens: asStringArray(candidate.columnTokens),
        rowTokens: asStringArray(candidate.rowTokens),
        attention: sanitizeAttentionMatrix(candidate.attention),
        warnings: asStringArray(candidate.warnings),
    };
}

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

function previewPillLabel(
    activeTab: AttentionTab,
    semanticWeights: SemanticWeight[],
    documentAttention: DocumentAttentionPayload | null,
    pairAttention: SentencePairAttentionPayload | null,
    analysisError: string | null,
    attentionError: string | null,
    isLoadingDocumentAttention: boolean,
    isLoadingPairAttention: boolean,
) {
    if (activeTab === 'rendered') {
        return analysisError ? 'Backend unavailable, rendering without weights' : `${semanticWeights.length} sentence weights loaded`;
    }

    if (activeTab === 'sentence') {
        if (attentionError) {
            return 'Sentence map unavailable';
        }
        if (isLoadingDocumentAttention) {
            return 'Computing full-document attention';
        }
        return documentAttention ? `${documentAttention.sentenceCount} sentences in focus` : 'Sentence map ready when requested';
    }

    if (attentionError) {
        return 'Token view unavailable';
    }
    if (isLoadingPairAttention) {
        return 'Loading sentence pair tokens';
    }
    return pairAttention ? 'Inspecting selected sentence pair' : 'Choose a sentence pair first';
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
    const [activeTab, setActiveTab] = useState<AttentionTab>('rendered');
    const [documentAttention, setDocumentAttention] = useState<DocumentAttentionPayload | null>(null);
    const [pairAttention, setPairAttention] = useState<SentencePairAttentionPayload | null>(null);
    const [attentionError, setAttentionError] = useState<string | null>(null);
    const [isLoadingDocumentAttention, setIsLoadingDocumentAttention] = useState(false);
    const [isLoadingPairAttention, setIsLoadingPairAttention] = useState(false);
    const [selectedPair, setSelectedPair] = useState<{ sentenceXIndex: number; sentenceYIndex: number } | null>(null);
    const deferredContent = useDeferredValue(content);
    const latestRequestRef = useRef(0);
    const latestDocumentRequestRef = useRef(0);
    const latestPairRequestRef = useRef(0);

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
                        modelName: 'local model',
                        device: 'unavailable',
                        maxInputTokens: null,
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
        startTransition(() => {
            setDocumentAttention(null);
            setPairAttention(null);
            setAttentionError(null);
            setSelectedPair(null);
        });
    }, [deferredContent]);

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

                const payload = sanitizeSemanticWeights(await response.json());
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

    useEffect(() => {
        if (activeTab === 'rendered' || !deferredContent.trim()) {
            setIsLoadingDocumentAttention(false);
            return;
        }

        const controller = new AbortController();
        const requestId = latestDocumentRequestRef.current + 1;
        latestDocumentRequestRef.current = requestId;

        const timeoutId = window.setTimeout(async () => {
            setIsLoadingDocumentAttention(true);
            setAttentionError(null);

            try {
                const response = await fetch(`${API_BASE_URL}/attention/document`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: deferredContent }),
                    signal: controller.signal,
                });

                if (!response.ok) {
                    const errorBody = await response.json().catch(() => null) as { detail?: string } | null;
                    throw new Error(errorBody?.detail || `Attention analysis failed with status ${response.status}`);
                }

                const payload = sanitizeDocumentAttentionPayload(await response.json());
                if (latestDocumentRequestRef.current === requestId) {
                    startTransition(() => {
                        setDocumentAttention(payload);
                        setAttentionError(null);
                    });
                }
            } catch (error) {
                if (controller.signal.aborted) {
                    return;
                }

                if (latestDocumentRequestRef.current === requestId) {
                    startTransition(() => {
                        setDocumentAttention(null);
                        setAttentionError(error instanceof Error ? error.message : 'Attention analysis failed.');
                    });
                }
            } finally {
                if (!controller.signal.aborted && latestDocumentRequestRef.current === requestId) {
                    setIsLoadingDocumentAttention(false);
                }
            }
        }, 250);

        return () => {
            controller.abort();
            window.clearTimeout(timeoutId);
        };
    }, [activeTab, deferredContent]);

    useEffect(() => {
        if (activeTab !== 'token' || !selectedPair || !deferredContent.trim()) {
            setIsLoadingPairAttention(false);
            return;
        }

        const controller = new AbortController();
        const requestId = latestPairRequestRef.current + 1;
        latestPairRequestRef.current = requestId;

        const timeoutId = window.setTimeout(async () => {
            setIsLoadingPairAttention(true);
            setAttentionError(null);

            try {
                const response = await fetch(`${API_BASE_URL}/attention/pair`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: deferredContent,
                        sentenceXIndex: selectedPair.sentenceXIndex,
                        sentenceYIndex: selectedPair.sentenceYIndex,
                    }),
                    signal: controller.signal,
                });

                if (!response.ok) {
                    const errorBody = await response.json().catch(() => null) as { detail?: string } | null;
                    throw new Error(errorBody?.detail || `Token attention analysis failed with status ${response.status}`);
                }

                const payload = sanitizeSentencePairAttentionPayload(await response.json());
                if (latestPairRequestRef.current === requestId) {
                    startTransition(() => {
                        setPairAttention(payload);
                        setAttentionError(null);
                    });
                }
            } catch (error) {
                if (controller.signal.aborted) {
                    return;
                }

                if (latestPairRequestRef.current === requestId) {
                    startTransition(() => {
                        setPairAttention(null);
                        setAttentionError(error instanceof Error ? error.message : 'Token attention analysis failed.');
                    });
                }
            } finally {
                if (!controller.signal.aborted && latestPairRequestRef.current === requestId) {
                    setIsLoadingPairAttention(false);
                }
            }
        }, 150);

        return () => {
            controller.abort();
            window.clearTimeout(timeoutId);
        };
    }, [activeTab, deferredContent, selectedPair]);

    const stats = useMemo(() => {
        const words = content.trim() ? content.trim().split(/\s+/).length : 0;
        const sentences = sentenceCount(content);
        const highlighted = semanticWeights.filter((weight) => weight.band !== 'low').length;

        return { words, sentences, highlighted };
    }, [content, semanticWeights]);

    const headerStatus = statusLabel(health, isAnalyzing, analysisError);

    const handleSelectPair = ({ sentenceXIndex, sentenceYIndex }: { sentenceXIndex: number; sentenceYIndex: number }) => {
        startTransition(() => {
            setSelectedPair({ sentenceXIndex, sentenceYIndex });
            setPairAttention(null);
            setActiveTab('token');
        });
    };

    const currentPreviewPillLabel = previewPillLabel(
        activeTab,
        semanticWeights,
        documentAttention,
        pairAttention,
        analysisError,
        attentionError,
        isLoadingDocumentAttention,
        isLoadingPairAttention,
    );

    return (
        <div className="app-shell">
            <div className="app-backdrop app-backdrop-primary" />
            <div className="app-backdrop app-backdrop-secondary" />

            <main className="workspace">
                <header className="hero">
                    <div>
                        <p className="eyebrow">Local sentence attention</p>
                        <h1>SAVIS Attention Markdown</h1>
                        <p className="hero-copy">
                            The markdown renderer stays intact, while a full-context local causal model powers both sentence-level emphasis and a closer look at sentence and token attention patterns.
                        </p>
                    </div>

                    <div className="hero-metrics">
                        <div className="metric-card">
                            <span className="metric-label">Status</span>
                            <strong>{headerStatus}</strong>
                            <small>
                                {health?.modelName ?? 'local model'} on {health?.device ?? 'pending'}
                            </small>
                        </div>
                        <div className="metric-card">
                            <span className="metric-label">Document</span>
                            <strong>{stats.words} words</strong>
                            <small>{stats.sentences} detected sentences</small>
                        </div>
                        <div className="metric-card">
                            <span className="metric-label">Attention cap</span>
                            <strong>{health?.maxInputTokens ?? 'unknown'} tokens</strong>
                            <small>{stats.highlighted} highlighted sentences</small>
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
                                <h2>Rendered text, sentence map, and token explorer</h2>
                            </div>
                            <span className={`panel-pill ${(analysisError || attentionError) ? 'panel-pill-warning' : ''}`}>
                                {currentPreviewPillLabel}
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

                        <div className="preview-tab-bar" role="tablist" aria-label="Attention explorer views">
                            {VIEW_TABS.map((tab) => (
                                <button
                                    key={tab.id}
                                    type="button"
                                    role="tab"
                                    aria-selected={activeTab === tab.id}
                                    className={`preview-tab ${activeTab === tab.id ? 'preview-tab-active' : ''}`}
                                    onClick={() => setActiveTab(tab.id)}
                                >
                                    {tab.label}
                                </button>
                            ))}
                        </div>

                        <div className="preview-surface">
                            {activeTab === 'rendered' ? (
                                <RenderMathMarkdown content={content} semanticWeights={semanticWeights} />
                            ) : (
                                <AttentionExplorer
                                    activeTab={activeTab}
                                    documentAttention={documentAttention}
                                    pairAttention={pairAttention}
                                    isLoadingDocument={isLoadingDocumentAttention}
                                    isLoadingPair={isLoadingPairAttention}
                                    error={attentionError}
                                    onSelectPair={handleSelectPair}
                                />
                            )}
                        </div>
                    </section>
                </section>
            </main>

            <GlobalImagePreviewModal />
        </div>
    );
};

export default App;
