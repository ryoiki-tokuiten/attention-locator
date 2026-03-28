import React, { startTransition, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react';
import { FileText, GitBranch, Globe, LoaderCircle, PanelLeftClose, PanelLeftOpen, SlidersHorizontal, SwatchBook, Target } from 'lucide-react';

import { GlobalImagePreviewModal, RenderMathMarkdown, type AttentionArcConnection } from '../RenderMathMarkdown';
import AttentionExplorer, { attentionColorForWeight, type AttentionTab } from './AttentionExplorer';
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

type RenderArcMode = 'off' | 'manual' | 'focused' | 'global';

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
        sourceStart: candidate.sourceStart == null ? null : asFiniteNumber(candidate.sourceStart, 0),
        sourceEnd: candidate.sourceEnd == null ? null : asFiniteNumber(candidate.sourceEnd, 0),
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

function previewPillLabel(
    activeTab: AttentionTab,
    renderArcMode: RenderArcMode,
    semanticWeights: SemanticWeight[],
    documentAttention: DocumentAttentionPayload | null,
    pairAttention: SentencePairAttentionPayload | null,
    analysisError: string | null,
    attentionError: string | null,
    isLoadingDocumentAttention: boolean,
    isLoadingPairAttention: boolean,
) {
    if (activeTab === 'rendered') {
        if (renderArcMode !== 'off') {
            if (attentionError && !documentAttention) {
                return 'Arc overlay unavailable';
            }
            if (isLoadingDocumentAttention) {
                return 'Building sentence arcs';
            }
            return documentAttention ? `Arc graph ready · ${documentAttention.sentenceCount} sentences` : 'Arc overlay ready on demand';
        }
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

function previewActivityState(
    activeTab: AttentionTab,
    isInputPending: boolean,
    isAnalyzing: boolean,
    isLoadingDocumentAttention: boolean,
    isLoadingPairAttention: boolean,
    renderArcMode: RenderArcMode,
) {
    if (isInputPending) {
        return {
            kicker: 'Live typing',
            title: 'Preparing the next semantic pass',
            description: 'Your text is already visible while the updated sentence weights queue in the background.',
        };
    }

    if (isAnalyzing) {
        return {
            kicker: 'Semantic refresh',
            title: 'Recomputing sentence emphasis',
            description: 'We are refreshing the rendered pill intensities so the preview reflects the latest wording.',
        };
    }

    if (activeTab === 'rendered' && renderArcMode !== 'off' && isLoadingDocumentAttention) {
        return {
            kicker: 'Arc overlay',
            title: 'Building sentence-to-sentence citations',
            description: 'The full-document sentence graph is loading so the rendered preview can draw later-to-earlier arcs.',
        };
    }

    if (activeTab === 'sentence' && isLoadingDocumentAttention) {
        return {
            kicker: 'Attention map',
            title: 'Building the sentence matrix',
            description: 'The model is integrating full-document attention so the sentence view can open with fresh scores.',
        };
    }

    if (activeTab === 'token' && isLoadingPairAttention) {
        return {
            kicker: 'Token drill-down',
            title: 'Resolving pair-level attention',
            description: 'We are extracting token interactions for the selected sentence pair.',
        };
    }

    return null;
}

function clampTopK(value: number) {
    return Math.max(1, Math.min(15, Math.round(value)));
}

function clampLongDistanceFactor(value: number, maxFactor: number) {
    return Math.max(1, Math.min(maxFactor, Math.round(value)));
}

function arcConnectionId(sourceIndex: number, targetIndex: number) {
    return `${sourceIndex}:${targetIndex}`;
}

function buildArcConnection(sourceIndex: number, targetIndex: number, score: number): AttentionArcConnection | null {
    const normalizedScore = clamp(asFiniteNumber(score, 0));

    if (sourceIndex === targetIndex || normalizedScore <= 0) {
        return null;
    }

    return {
        id: arcConnectionId(sourceIndex, targetIndex),
        sourceIndex,
        targetIndex,
        score: normalizedScore,
        color: attentionColorForWeight(normalizedScore),
        label: `${sourceIndex} -> ${targetIndex} · ${normalizedScore.toFixed(3)}`,
    };
}

interface LongDistanceOptions {
    enabled: boolean;
    minimumGap: number;
}

function passesLongDistanceFilter(
    sourceIndex: number,
    targetIndex: number,
    options: LongDistanceOptions,
) {
    if (!options.enabled) {
        return true;
    }

    return (sourceIndex - targetIndex) > options.minimumGap;
}

function sortedPastConnections(
    sentenceAttention: number[][],
    sourceIndex: number,
    longDistanceOptions: LongDistanceOptions,
) {
    const row = sentenceAttention[sourceIndex] ?? [];

    return row
        .map((score, targetIndex) => buildArcConnection(sourceIndex, targetIndex, score))
        .filter((connection): connection is AttentionArcConnection => (
            Boolean(
                connection &&
                connection.targetIndex < sourceIndex &&
                passesLongDistanceFilter(connection.sourceIndex, connection.targetIndex, longDistanceOptions),
            )
        ))
        .sort((left, right) => (
            right.score - left.score ||
            left.targetIndex - right.targetIndex ||
            left.sourceIndex - right.sourceIndex
        ));
}

function topGlobalConnections(
    sentenceAttention: number[][],
    topK: number,
    longDistanceOptions: LongDistanceOptions,
) {
    const connections: AttentionArcConnection[] = [];

    sentenceAttention.forEach((row, sourceIndex) => {
        row.forEach((score, targetIndex) => {
            const connection = buildArcConnection(sourceIndex, targetIndex, score);
            if (
                connection &&
                targetIndex < sourceIndex &&
                passesLongDistanceFilter(connection.sourceIndex, connection.targetIndex, longDistanceOptions)
            ) {
                connections.push(connection);
            }
        });
    });

    return connections
        .sort((left, right) => (
            right.score - left.score ||
            left.sourceIndex - right.sourceIndex ||
            left.targetIndex - right.targetIndex
        ))
        .slice(0, clampTopK(topK));
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
    const [isEditorCollapsed, setIsEditorCollapsed] = useState(false);
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
    const [renderArcMode, setRenderArcMode] = useState<RenderArcMode>('off');
    const [manualArcSelection, setManualArcSelection] = useState<number[]>([]);
    const [focusedArcSentence, setFocusedArcSentence] = useState<number | null>(null);
    const [focusedArcTopK, setFocusedArcTopK] = useState(5);
    const [globalArcTopK, setGlobalArcTopK] = useState(5);
    const [focusedLongDistanceEnabled, setFocusedLongDistanceEnabled] = useState(false);
    const [globalLongDistanceEnabled, setGlobalLongDistanceEnabled] = useState(false);
    const [focusedLongDistanceFactor, setFocusedLongDistanceFactor] = useState(5);
    const [globalLongDistanceFactor, setGlobalLongDistanceFactor] = useState(5);
    const [selectedArcId, setSelectedArcId] = useState<string | null>(null);
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
            setManualArcSelection([]);
            setFocusedArcSentence(null);
            setSelectedArcId(null);
        });
    }, [deferredContent]);

    useEffect(() => {
        setSelectedArcId(null);

        if (renderArcMode === 'off') {
            setManualArcSelection([]);
            setFocusedArcSentence(null);
            return;
        }

        if (renderArcMode === 'manual') {
            setFocusedArcSentence(null);
            return;
        }

        setManualArcSelection([]);

        if (renderArcMode === 'global') {
            setFocusedArcSentence(null);
        }
    }, [renderArcMode]);

    const maxLongDistanceFactor = useMemo(() => {
        if (!documentAttention) {
            return 5;
        }

        return Math.max(1, documentAttention.sentenceCount - 1);
    }, [documentAttention]);

    useEffect(() => {
        setFocusedLongDistanceFactor((currentValue) => clampLongDistanceFactor(currentValue, maxLongDistanceFactor));
        setGlobalLongDistanceFactor((currentValue) => clampLongDistanceFactor(currentValue, maxLongDistanceFactor));
    }, [maxLongDistanceFactor]);

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

    const shouldLoadDocumentAttention = Boolean(deferredContent.trim()) && (activeTab !== 'rendered' || renderArcMode !== 'off');

    useEffect(() => {
        if (!shouldLoadDocumentAttention) {
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
    }, [activeTab, deferredContent, renderArcMode, shouldLoadDocumentAttention]);

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

        return { words };
    }, [content]);

    const isInputPending = content !== deferredContent;
    const previewActivity = previewActivityState(
        activeTab,
        isInputPending,
        isAnalyzing,
        isLoadingDocumentAttention,
        isLoadingPairAttention,
        renderArcMode,
    );

    const handleSelectPair = ({ sentenceXIndex, sentenceYIndex }: { sentenceXIndex: number; sentenceYIndex: number }) => {
        startTransition(() => {
            setSelectedPair({ sentenceXIndex, sentenceYIndex });
            setPairAttention(null);
            setActiveTab('token');
        });
    };

    const currentPreviewPillLabel = previewPillLabel(
        activeTab,
        renderArcMode,
        semanticWeights,
        documentAttention,
        pairAttention,
        analysisError,
        attentionError,
        isLoadingDocumentAttention,
        isLoadingPairAttention,
    );

    const focusedLongDistanceOptions = useMemo<LongDistanceOptions>(() => ({
        enabled: focusedLongDistanceEnabled,
        minimumGap: clampLongDistanceFactor(focusedLongDistanceFactor, maxLongDistanceFactor),
    }), [focusedLongDistanceEnabled, focusedLongDistanceFactor, maxLongDistanceFactor]);

    const globalLongDistanceOptions = useMemo<LongDistanceOptions>(() => ({
        enabled: globalLongDistanceEnabled,
        minimumGap: clampLongDistanceFactor(globalLongDistanceFactor, maxLongDistanceFactor),
    }), [globalLongDistanceEnabled, globalLongDistanceFactor, maxLongDistanceFactor]);

    const renderArcConnections = useMemo(() => {
        if (!documentAttention || renderArcMode === 'off') {
            return [];
        }

        if (renderArcMode === 'manual') {
            if (manualArcSelection.length < 2) {
                return [];
            }

            const sourceIndex = Math.max(...manualArcSelection);
            const targetIndex = Math.min(...manualArcSelection);
            const connection = buildArcConnection(
                sourceIndex,
                targetIndex,
                documentAttention.sentenceAttention[sourceIndex]?.[targetIndex],
            );

            return connection ? [connection] : [];
        }

        if (renderArcMode === 'focused') {
            if (focusedArcSentence == null) {
                return [];
            }

            return sortedPastConnections(
                documentAttention.sentenceAttention,
                focusedArcSentence,
                focusedLongDistanceOptions,
            )
                .slice(0, clampTopK(focusedArcTopK));
        }

        return topGlobalConnections(
            documentAttention.sentenceAttention,
            globalArcTopK,
            globalLongDistanceOptions,
        );
    }, [
        documentAttention,
        focusedArcSentence,
        focusedLongDistanceOptions,
        focusedArcTopK,
        globalLongDistanceOptions,
        globalArcTopK,
        manualArcSelection,
        renderArcMode,
    ]);

    const activeRenderArc = useMemo(() => (
        renderArcConnections.find((connection) => connection.id === selectedArcId) ?? null
    ), [renderArcConnections, selectedArcId]);

    const renderArcSelectedSentences = useMemo(() => {
        const selected = new Set<number>();

        if (renderArcMode === 'manual') {
            manualArcSelection.forEach((sentenceIndex) => selected.add(sentenceIndex));
        }

        if (renderArcMode === 'focused' && focusedArcSentence != null) {
            selected.add(focusedArcSentence);
        }

        if (activeRenderArc) {
            selected.add(activeRenderArc.sourceIndex);
            selected.add(activeRenderArc.targetIndex);
        }

        return [...selected];
    }, [activeRenderArc, focusedArcSentence, manualArcSelection, renderArcMode]);

    const renderArcHelper = useMemo(() => {
        if (renderArcMode === 'off') {
            return 'Turn on arcs to trace how later sentences cite earlier ones inside the rendered preview.';
        }

        if (attentionError && !documentAttention) {
            return attentionError;
        }

        if (isLoadingDocumentAttention && !documentAttention) {
            return 'Loading the full-document sentence graph for arc overlays...';
        }

        if (!documentAttention) {
            return 'Arc overlays will appear once the document sentence graph is ready.';
        }

        if (renderArcMode === 'manual') {
            if (!manualArcSelection.length) {
                return 'Click two different sentences. We will draw one later-to-earlier citation arc between them.';
            }
            if (manualArcSelection.length === 1) {
                return `Sentence ${manualArcSelection[0]} selected. Choose one more sentence to complete the arc.`;
            }
            return 'Two sentences selected. Click the arc itself if you want to reveal its score.';
        }

        if (renderArcMode === 'focused') {
            if (focusedArcSentence == null) {
                return 'Click a later sentence to reveal its strongest past citations.';
            }
            if (!renderArcConnections.length) {
                if (focusedLongDistanceOptions.enabled) {
                    return `Sentence ${focusedArcSentence} has no citations beyond a ${focusedLongDistanceOptions.minimumGap}-sentence gap.`;
                }
                return `Sentence ${focusedArcSentence} has no earlier citations to draw yet.`;
            }
            if (focusedLongDistanceOptions.enabled) {
                return `Showing the top ${renderArcConnections.length} long-distance citations for sentence ${focusedArcSentence}, excluding gaps of ${focusedLongDistanceOptions.minimumGap} sentences or less.`;
            }
            return `Showing the top ${renderArcConnections.length} past citations for sentence ${focusedArcSentence}.`;
        }

        if (!renderArcConnections.length) {
            if (globalLongDistanceOptions.enabled) {
                return `No global citations remain beyond a ${globalLongDistanceOptions.minimumGap}-sentence gap.`;
            }
            return 'No global later-to-earlier citation arcs are available for this document yet.';
        }

        if (globalLongDistanceOptions.enabled) {
            return `Showing the strongest ${renderArcConnections.length} long-distance citation arcs, excluding gaps of ${globalLongDistanceOptions.minimumGap} sentences or less.`;
        }

        return `Showing the strongest ${renderArcConnections.length} later-to-earlier citation arcs in the document.`;
    }, [
        attentionError,
        documentAttention,
        focusedArcSentence,
        focusedLongDistanceOptions,
        focusedArcTopK,
        globalLongDistanceOptions,
        isLoadingDocumentAttention,
        manualArcSelection,
        renderArcConnections,
        renderArcMode,
    ]);

    const handleRenderedSentenceClick = (sentenceIndex: number) => {
        if (renderArcMode === 'manual') {
            setSelectedArcId(null);
            setManualArcSelection((currentSelection) => {
                if (currentSelection.includes(sentenceIndex)) {
                    return currentSelection.filter((index) => index !== sentenceIndex);
                }

                if (currentSelection.length >= 2) {
                    return [currentSelection[currentSelection.length - 1], sentenceIndex];
                }

                return [...currentSelection, sentenceIndex];
            });
            return;
        }

        if (renderArcMode === 'focused') {
            setSelectedArcId(null);
            setFocusedArcSentence((currentSentence) => (
                currentSentence === sentenceIndex ? null : sentenceIndex
            ));
        }
    };

    const currentArcTopK = renderArcMode === 'focused' ? clampTopK(focusedArcTopK) : clampTopK(globalArcTopK);
    const isCurrentLongDistanceEnabled = renderArcMode === 'focused'
        ? focusedLongDistanceEnabled
        : globalLongDistanceEnabled;
    const currentLongDistanceFactor = renderArcMode === 'focused'
        ? focusedLongDistanceOptions.minimumGap
        : globalLongDistanceOptions.minimumGap;

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
                            A preview-first markdown workspace for seeing where the model concentrates, with sentence and token attention tucked behind the same reading surface.
                        </p>
                    </div>

                    <div className="hero-aside">
                        <aside className="legend-showcase" aria-label="Attention color legend">
                            <div className="legend-showcase-header">
                                <div>
                                    <p className="panel-label">Attention spectrum</p>
                                    <h2>Legend</h2>
                                </div>
                                <SwatchBook size={18} />
                            </div>
                            <div className="attention-legend-bar" />
                            <div className="legend-showcase-grid">
                                {ATTENTION_LEGEND_ITEMS.map((item) => (
                                    <div key={item.label} className="legend-showcase-item">
                                        <span className={`attention-stop ${item.className}`} />
                                        <div>
                                            <strong>{item.label}</strong>
                                            <small>{item.description}</small>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </aside>
                    </div>
                </header>

                <section className={`panel-grid ${isEditorCollapsed ? 'panel-grid-collapsed' : ''}`}>
                    <section className={`editor-panel glass-panel ${isEditorCollapsed ? 'editor-panel-collapsed' : ''}`}>
                        <button
                            type="button"
                            className="editor-collapse-toggle"
                            onClick={() => setIsEditorCollapsed((value) => !value)}
                            aria-label={isEditorCollapsed ? 'Expand editor panel' : 'Collapse editor panel'}
                            title={isEditorCollapsed ? 'Expand editor panel' : 'Collapse editor panel'}
                        >
                            {isEditorCollapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
                        </button>

                        {isEditorCollapsed ? (
                            <div className="editor-collapsed-strip">
                                <FileText size={18} />
                                <span>Editor</span>
                                <small>{stats.words} words</small>
                            </div>
                        ) : (
                            <>
                                <div className="panel-header editor-panel-header">
                                    <div>
                                        <p className="panel-label">Markdown input</p>
                                        <h2>Compose or paste a document</h2>
                                    </div>
                                    <span className="panel-pill panel-pill-quiet">
                                        {isInputPending || isAnalyzing ? 'Live analysis syncing' : 'Preview-first editing'}
                                    </span>
                                </div>

                                <textarea
                                    className="editor-textarea"
                                    value={content}
                                    onChange={(event) => setContent(event.target.value)}
                                    spellCheck={false}
                                    placeholder="Paste markdown here..."
                                />
                            </>
                        )}
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

                        {previewActivity ? (
                            <div className="preview-activity-card" aria-live="polite">
                                <div className="preview-activity-header">
                                    <div className="preview-activity-copy">
                                        <span className="preview-activity-kicker">{previewActivity.kicker}</span>
                                        <strong>{previewActivity.title}</strong>
                                        <span>{previewActivity.description}</span>
                                    </div>
                                    <LoaderCircle size={18} className="spin-icon" />
                                </div>
                                <div className="preview-activity-track">
                                    <span className="preview-activity-bar" />
                                </div>
                            </div>
                        ) : null}

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
                                <div className="rendered-preview-stack">
                                    <section className="arc-toolbar" aria-label="Sentence arc controls">
                                        <div className="arc-toolbar-header">
                                            <div>
                                                <p className="panel-label">Arc overlay</p>
                                                <h3>Later-to-earlier sentence citations</h3>
                                            </div>
                                            <GitBranch size={18} />
                                        </div>

                                        <div className="arc-mode-group" role="group" aria-label="Arc overlay modes">
                                            <button
                                                type="button"
                                                className={`arc-mode-button ${renderArcMode === 'manual' ? 'arc-mode-button-active' : ''}`}
                                                onClick={() => setRenderArcMode('manual')}
                                            >
                                                <GitBranch size={16} />
                                                <span className="arc-mode-button-label">Pair select</span>
                                            </button>
                                            <button
                                                type="button"
                                                className={`arc-mode-button ${renderArcMode === 'focused' ? 'arc-mode-button-active' : ''}`}
                                                onClick={() => setRenderArcMode('focused')}
                                            >
                                                <Target size={16} />
                                                <span className="arc-mode-button-label">Top-k from sentence</span>
                                            </button>
                                            <button
                                                type="button"
                                                className={`arc-mode-button ${renderArcMode === 'global' ? 'arc-mode-button-active' : ''}`}
                                                onClick={() => setRenderArcMode('global')}
                                            >
                                                <Globe size={16} />
                                                <span className="arc-mode-button-label">Global top-k</span>
                                            </button>
                                            <button
                                                type="button"
                                                className={`arc-mode-button ${renderArcMode === 'off' ? 'arc-mode-button-active arc-mode-button-muted' : 'arc-mode-button-muted'}`}
                                                onClick={() => setRenderArcMode('off')}
                                            >
                                                <span className="arc-mode-button-label">Off</span>
                                            </button>
                                        </div>

                                        {renderArcMode === 'focused' || renderArcMode === 'global' ? (
                                            <>
                                                <label className="arc-slider-card">
                                                    <div className="arc-slider-heading">
                                                        <span>
                                                            <SlidersHorizontal size={15} />
                                                            {renderArcMode === 'focused' ? 'Top-k past citations' : 'Top-k global arcs'}
                                                        </span>
                                                        <strong>{currentArcTopK}</strong>
                                                    </div>
                                                    <input
                                                        className="arc-slider"
                                                        type="range"
                                                        min={1}
                                                        max={15}
                                                        step={1}
                                                        value={currentArcTopK}
                                                        onChange={(event) => {
                                                            const nextValue = clampTopK(Number(event.target.value));
                                                            if (renderArcMode === 'focused') {
                                                                setFocusedArcTopK(nextValue);
                                                                return;
                                                            }
                                                            setGlobalArcTopK(nextValue);
                                                        }}
                                                    />
                                                    <div className="arc-slider-scale" aria-hidden="true">
                                                        <span>1</span>
                                                        <span>15</span>
                                                    </div>
                                                </label>

                                                <section className="arc-filter-card" aria-label="Long distance relationship filter">
                                                    <div className="arc-filter-toggle">
                                                        <div className="arc-filter-copy">
                                                            <strong>Long distance relationship factor</strong>
                                                        </div>
                                                        <button
                                                            type="button"
                                                            role="switch"
                                                            aria-checked={isCurrentLongDistanceEnabled}
                                                            className={`arc-filter-switch ${isCurrentLongDistanceEnabled ? 'arc-filter-switch-active' : ''}`}
                                                            onClick={() => {
                                                                if (renderArcMode === 'focused') {
                                                                    setFocusedLongDistanceEnabled((currentValue) => !currentValue);
                                                                    return;
                                                                }
                                                                setGlobalLongDistanceEnabled((currentValue) => !currentValue);
                                                            }}
                                                        >
                                                            <span className="arc-filter-switch-thumb" />
                                                        </button>
                                                    </div>

                                                    <label className={`arc-filter-slider-wrap ${isCurrentLongDistanceEnabled ? '' : 'arc-filter-slider-wrap-disabled'}`}>
                                                        <div className="arc-slider-heading">
                                                            <strong>{currentLongDistanceFactor}</strong>
                                                        </div>
                                                        <input
                                                            className="arc-slider"
                                                            type="range"
                                                            min={1}
                                                            max={maxLongDistanceFactor}
                                                            step={1}
                                                            disabled={!isCurrentLongDistanceEnabled}
                                                            value={currentLongDistanceFactor}
                                                            onChange={(event) => {
                                                                const nextValue = clampLongDistanceFactor(Number(event.target.value), maxLongDistanceFactor);
                                                                if (renderArcMode === 'focused') {
                                                                    setFocusedLongDistanceFactor(nextValue);
                                                                    return;
                                                                }
                                                                setGlobalLongDistanceFactor(nextValue);
                                                            }}
                                                        />
                                                        <div className="arc-slider-scale" aria-hidden="true">
                                                            <span>1</span>
                                                            <span>{maxLongDistanceFactor}</span>
                                                        </div>
                                                    </label>
                                                </section>
                                            </>
                                        ) : null}

                                        <div className="arc-toolbar-footer">
                                            <div className="arc-toolbar-copy">
                                                <span>{renderArcHelper}</span>
                                                {activeRenderArc ? (
                                                    <div className="arc-selection-banner" role="status" aria-live="polite">
                                                        <span className="arc-selection-dot" style={{ backgroundColor: activeRenderArc.color }} />
                                                        <strong>Selected arc</strong>
                                                        <span>{activeRenderArc.label}</span>
                                                    </div>
                                                ) : null}
                                            </div>
                                            {(renderArcMode === 'manual' && manualArcSelection.length > 0) || (renderArcMode === 'focused' && focusedArcSentence != null) || activeRenderArc ? (
                                                <button
                                                    type="button"
                                                    className="arc-clear-button"
                                                    onClick={() => {
                                                        setManualArcSelection([]);
                                                        setFocusedArcSentence(null);
                                                        setSelectedArcId(null);
                                                    }}
                                                >
                                                    Clear selection
                                                </button>
                                            ) : null}
                                        </div>
                                    </section>

                                    <RenderMathMarkdown
                                        content={content}
                                        semanticWeights={semanticWeights}
                                        sentenceArcConnections={renderArcConnections}
                                        selectedSentenceIndices={renderArcSelectedSentences}
                                        activeArcId={selectedArcId}
                                        isSentenceSelectionEnabled={renderArcMode === 'manual' || renderArcMode === 'focused'}
                                        onSentenceClick={(renderArcMode === 'manual' || renderArcMode === 'focused') ? handleRenderedSentenceClick : undefined}
                                        onArcClick={(arcId) => setSelectedArcId((currentId) => currentId === arcId ? null : arcId)}
                                    />
                                </div>
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
