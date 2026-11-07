import React, { startTransition, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react';
import { GitBranch, Globe, LoaderCircle, SlidersHorizontal, SwatchBook, Target } from 'lucide-react';

import { Render, type AttentionArcConnection } from '../render';
import AttentionExplorer, { type AttentionTab } from './AttentionExplorer';
import { asFiniteNumber, attentionColorForWeight, clamp } from './utils';
import type {
    DeltaVisualizationMode,
    DeltaVisualizationPayload,
    DocumentAttentionPayload,
    HealthPayload,
    SemanticBand,
    SemanticWeight,
    SentencePairAttentionPayload,
} from './types/semantic';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';
const INITIAL_MARKDOWN = `Attention-Locator Workspace.

This prototype renders plain text with sentence-level attention emphasis and lets you inspect the underlying attention patterns.

Sentence attention comes from a full-context causal model pass. You can type or paste directly inside this surface, draw sentence-to-sentence attention arcs, or switch into sentence-level and token-level attention inspection.`;

async function postJson(path: string, body: unknown, signal: AbortSignal, errorLabel: string): Promise<unknown> {
    const response = await fetch(`${API_BASE_URL}${path}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
        signal,
    });

    if (!response.ok) {
        const errorBody = await response.json().catch(() => null) as { detail?: string } | null;
        throw new Error(errorBody?.detail || `${errorLabel} with status ${response.status}`);
    }

    return response.json();
}

const VIEW_TABS: Array<{ id: AttentionTab; label: string }> = [
    { id: 'rendered', label: 'Rendered' },
    { id: 'sentence', label: 'Sentence View' },
    { id: 'token', label: 'Token View' },
];

const RENDER_SURFACE_MODES = [
    { id: 'semantic', label: 'Semantic Surface' },
    { id: 'delta', label: 'Embeddings Delta' },
] as const;

const DELTA_RENDER_MODES: Array<{ id: DeltaVisualizationMode; label: string; icon: React.ReactNode; description: string }> = [
    {
        id: 'local-global',
        label: 'Local vs global',
        icon: <Globe size={16} strokeWidth={1.5} />,
        description: 'Shows whether a sentence is shaped more by nearby structure or longer-range context',
    },
    {
        id: 'flow-map',
        label: 'Flow map',
        icon: <GitBranch size={16} strokeWidth={1.5} />,
        description: 'Draws sentence-to-sentence write paths to trace where context flows from',
    },
    {
        id: 'composition',
        label: 'Composition',
        icon: <SwatchBook size={16} strokeWidth={1.5} />,
        description: 'Colors each sentence by the upstream sentences that contributed most to it',
    },
    {
        id: 'drift',
        label: 'Drift',
        icon: <Target size={16} strokeWidth={1.5} />,
        description: 'Highlights sentences whose representations change most across transformer depth',
    },
] as const;

type RenderSurfaceMode = typeof RENDER_SURFACE_MODES[number]['id'];

type RenderArcMode = 'off' | 'manual' | 'focused' | 'global';

function asString(value: unknown, fallback = '') {
    return typeof value === 'string' ? value : fallback;
}

function asStringArray(value: unknown): string[] {
    if (!Array.isArray(value)) {
        return [];
    }
    return value.map((item) => asString(item)).filter(Boolean);
}

function asRecord(value: unknown): Record<string, unknown> {
    return value && typeof value === 'object' ? value as Record<string, unknown> : {};
}

function asNullableFiniteNumber(value: unknown): number | null {
    return value == null ? null : asFiniteNumber(value, 0);
}

function asDistanceBucket(value: unknown): 'local' | 'medium' | 'global' {
    return value === 'local' || value === 'medium' || value === 'global' ? value : 'local';
}

function asSemanticBand(value: unknown): SemanticBand {
    return value === 'high' || value === 'medium' || value === 'low' ? value : 'medium';
}

function sanitizeSemanticWeight(item: unknown, fallbackIndex: number): SemanticWeight {
    const candidate = asRecord(item);
    const sentence = asString(candidate.sentence);
    const normalizedSentence = asString(candidate.normalizedSentence, sentence.trim().toLowerCase());

    return {
        index: asFiniteNumber(candidate.index, fallbackIndex),
        sentence,
        normalizedSentence,
        weight: clamp(asFiniteNumber(candidate.weight, 0.5)),
        band: asSemanticBand(candidate.band),
        sourceStart: asNullableFiniteNumber(candidate.sourceStart),
        sourceEnd: asNullableFiniteNumber(candidate.sourceEnd),
    };
}

function sanitizeSemanticWeights(payload: unknown): SemanticWeight[] {
    if (!Array.isArray(payload)) {
        return [];
    }

    return payload.map((item, index) => sanitizeSemanticWeight(item, index));
}

function sanitizeNumericMatrix(payload: unknown) {
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
    const candidate = asRecord(payload);
    const sentenceWeights = sanitizeSemanticWeights(candidate.sentenceWeights);

    return {
        analysisText: asString(candidate.analysisText),
        tokenCount: asFiniteNumber(candidate.tokenCount, 0),
        sentenceCount: asFiniteNumber(candidate.sentenceCount, sentenceWeights.length),
        sentenceWeights,
        sentenceAttention: sanitizeNumericMatrix(candidate.sentenceAttention),
        warnings: asStringArray(candidate.warnings),
    };
}

function sanitizeSentencePairAttentionPayload(payload: unknown): SentencePairAttentionPayload {
    const candidate = asRecord(payload);

    return {
        sentenceXIndex: asFiniteNumber(candidate.sentenceXIndex, 0),
        sentenceYIndex: asFiniteNumber(candidate.sentenceYIndex, 0),
        sentenceX: asString(candidate.sentenceX),
        sentenceY: asString(candidate.sentenceY),
        columnTokens: asStringArray(candidate.columnTokens),
        rowTokens: asStringArray(candidate.rowTokens),
        attention: sanitizeNumericMatrix(candidate.attention),
        warnings: asStringArray(candidate.warnings),
    };
}

function asDeltaMode(value: unknown): DeltaVisualizationMode {
    return value === 'flow-map' || value === 'composition' || value === 'drift' || value === 'local-global'
        ? value
        : 'local-global';
}

function sanitizeDeltaVisualizationPayload(payload: unknown): DeltaVisualizationPayload {
    const candidate = asRecord(payload);
    const sentences = Array.isArray(candidate.sentences) ? candidate.sentences : [];
    const globalTopEdges = Array.isArray(candidate.globalTopEdges) ? candidate.globalTopEdges : [];

    return {
        analysisText: asString(candidate.analysisText),
        tokenCount: asFiniteNumber(candidate.tokenCount, 0),
        sentenceCount: asFiniteNumber(candidate.sentenceCount, 0),
        layerCount: asFiniteNumber(candidate.layerCount, 0),
        headCount: asFiniteNumber(candidate.headCount, 0),
        hiddenSize: asFiniteNumber(candidate.hiddenSize, 0),
        defaultMode: asDeltaMode(candidate.defaultMode),
        sentences: sentences.map((item, index) => {
            const sentenceCandidate = asRecord(item);
            const topSources = Array.isArray(sentenceCandidate.topSources) ? sentenceCandidate.topSources : [];
            return {
                index: asFiniteNumber(sentenceCandidate.index, index),
                sentence: asString(sentenceCandidate.sentence),
                displaySentence: asString(sentenceCandidate.displaySentence, asString(sentenceCandidate.sentence)),
                sourceStart: asNullableFiniteNumber(sentenceCandidate.sourceStart),
                sourceEnd: asNullableFiniteNumber(sentenceCandidate.sourceEnd),
                visible: Boolean(sentenceCandidate.visible),
                incomingFlow: asFiniteNumber(sentenceCandidate.incomingFlow, 0),
                incomingFlowNorm: clamp(asFiniteNumber(sentenceCandidate.incomingFlowNorm, 0)),
                localShare: clamp(asFiniteNumber(sentenceCandidate.localShare, 0)),
                mediumShare: clamp(asFiniteNumber(sentenceCandidate.mediumShare, 0)),
                globalShare: clamp(asFiniteNumber(sentenceCandidate.globalShare, 0)),
                drift: asFiniteNumber(sentenceCandidate.drift, 0),
                driftNorm: clamp(asFiniteNumber(sentenceCandidate.driftNorm, 0)),
                peakLayer: asFiniteNumber(sentenceCandidate.peakLayer, 0),
                peakLayerRatio: clamp(asFiniteNumber(sentenceCandidate.peakLayerRatio, 0)),
                topSources: topSources.map((source, sourceIndex) => {
                    const sourceCandidate = asRecord(source);

                    return {
                        sourceIndex: asFiniteNumber(sourceCandidate.sourceIndex, sourceIndex),
                        score: asFiniteNumber(sourceCandidate.score, 0),
                        scoreNorm: clamp(asFiniteNumber(sourceCandidate.scoreNorm, 0)),
                        share: clamp(asFiniteNumber(sourceCandidate.share, 0)),
                        distance: asFiniteNumber(sourceCandidate.distance, 0),
                        distanceBucket: asDistanceBucket(sourceCandidate.distanceBucket),
                        sentence: asString(sourceCandidate.sentence),
                        displaySentence: asString(sourceCandidate.displaySentence, asString(sourceCandidate.sentence)),
                        sourceStart: asNullableFiniteNumber(sourceCandidate.sourceStart),
                        sourceEnd: asNullableFiniteNumber(sourceCandidate.sourceEnd),
                        visible: Boolean(sourceCandidate.visible),
                    };
                }),
            };
        }),
        driftByLayer: sanitizeNumericMatrix(candidate.driftByLayer),
        globalTopEdges: globalTopEdges.map((edge, edgeIndex) => {
            const edgeCandidate = asRecord(edge);

            return {
                id: asString(edgeCandidate.id, `edge-${edgeIndex}`),
                sourceIndex: asFiniteNumber(edgeCandidate.sourceIndex, 0),
                targetIndex: asFiniteNumber(edgeCandidate.targetIndex, 0),
                score: asFiniteNumber(edgeCandidate.score, 0),
                scoreNorm: clamp(asFiniteNumber(edgeCandidate.scoreNorm, 0)),
                distance: asFiniteNumber(edgeCandidate.distance, 0),
                distanceBucket: asDistanceBucket(edgeCandidate.distanceBucket),
                color: asString(edgeCandidate.color, attentionColorForWeight(asFiniteNumber(edgeCandidate.scoreNorm, 0))),
            };
        }),
        warnings: asStringArray(candidate.warnings),
    };
}

function previewActivityState(
    activeTab: AttentionTab,
    renderSurfaceMode: RenderSurfaceMode,
    isAnalyzing: boolean,
    isLoadingDocumentAttention: boolean,
    isLoadingPairAttention: boolean,
    isLoadingDeltaVisualization: boolean,
    renderArcMode: RenderArcMode,
) {
    if (isAnalyzing) {
        return {
            kicker: 'Semantic refresh',
            title: 'Recomputing sentence emphasis',
        };
    }

    if (activeTab === 'rendered' && renderSurfaceMode === 'delta' && isLoadingDeltaVisualization) {
        return {
            kicker: 'Delta capture',
            title: 'Building sentence flow surface',
        };
    }

    if (activeTab === 'rendered' && renderArcMode !== 'off' && isLoadingDocumentAttention) {
        return {
            kicker: 'Arc overlay',
            title: 'Building sentence arcs',
        };
    }

    if (activeTab === 'sentence' && isLoadingDocumentAttention) {
        return {
            kicker: 'Attention map',
            title: 'Building sentence matrix',
        };
    }

    if (activeTab === 'token' && isLoadingPairAttention) {
        return {
            kicker: 'Token drill-down',
            title: 'Resolving pair-level attention',
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

    return (targetIndex - sourceIndex) > options.minimumGap;
}

function sortedPastConnections(
    sentenceAttention: number[][],
    targetIndex: number,
    longDistanceOptions: LongDistanceOptions,
) {
    const row = sentenceAttention[targetIndex] ?? [];

    return row
        .map((score, sourceIndex) => buildArcConnection(sourceIndex, targetIndex, score))
        .filter((connection): connection is AttentionArcConnection => (
            Boolean(
                connection &&
                connection.sourceIndex < targetIndex &&
                passesLongDistanceFilter(connection.sourceIndex, connection.targetIndex, longDistanceOptions),
            )
        ))
        .sort((left, right) => (
            right.score - left.score ||
            left.sourceIndex - right.sourceIndex ||
            left.targetIndex - right.targetIndex
        ));
}

function topGlobalConnections(
    sentenceAttention: number[][],
    topK: number,
    longDistanceOptions: LongDistanceOptions,
) {
    const connections: AttentionArcConnection[] = [];

    sentenceAttention.forEach((row, targetIndex) => {
        row.forEach((score, sourceIndex) => {
            const connection = buildArcConnection(sourceIndex, targetIndex, score);
            if (
                connection &&
                sourceIndex < targetIndex &&
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
        description: 'Supporting context',
        className: 'attention-stop-cool',
    },
    {
        label: 'Medium attention',
        description: 'Connected structure',
        className: 'attention-stop-mid',
    },
    {
        label: 'Higher attention',
        description: 'Primary model focus',
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
    const [renderSurfaceMode, setRenderSurfaceMode] = useState<RenderSurfaceMode>('semantic');
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
    const [deltaVisualization, setDeltaVisualization] = useState<DeltaVisualizationPayload | null>(null);
    const [deltaVisualizationError, setDeltaVisualizationError] = useState<string | null>(null);
    const [isLoadingDeltaVisualization, setIsLoadingDeltaVisualization] = useState(false);
    const [deltaVisualizationMode, setDeltaVisualizationMode] = useState<DeltaVisualizationMode>('local-global');
    const [selectedDeltaSentenceIndex, setSelectedDeltaSentenceIndex] = useState<number | null>(null);
    const [selectedDeltaEdgeId, setSelectedDeltaEdgeId] = useState<string | null>(null);
    const deferredContent = useDeferredValue(content);
    const latestRequestRef = useRef(0);
    const latestDocumentRequestRef = useRef(0);
    const latestPairRequestRef = useRef(0);
    const latestDeltaVisualizationRequestRef = useRef(0);
    const lastDocumentTextRef = useRef<string | null>(null);
    const lastDeltaTextRef = useRef<string | null>(null);

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
                setDocumentAttention(null);
                setPairAttention(null);
                setAttentionError(null);
                setSelectedPair(null);
                setManualArcSelection([]);
                setFocusedArcSentence(null);
                setSelectedArcId(null);
                setDeltaVisualization(null);
                setDeltaVisualizationError(null);
                setSelectedDeltaSentenceIndex(null);
                setSelectedDeltaEdgeId(null);
            });
        }
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
                const payload = sanitizeSemanticWeights(await postJson(
                    '/analyze',
                    { text: deferredContent },
                    controller.signal,
                    'Analysis failed',
                ));
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

    const shouldLoadDocumentAttention = Boolean(deferredContent.trim()) && (
        activeTab !== 'rendered' ||
        (renderSurfaceMode === 'semantic' && renderArcMode !== 'off')
    );

    useEffect(() => {
        if (!shouldLoadDocumentAttention) {
            setIsLoadingDocumentAttention(false);
            return;
        }

        const controller = new AbortController();
        const requestId = latestDocumentRequestRef.current + 1;
        latestDocumentRequestRef.current = requestId;
        const isTextEdit = lastDocumentTextRef.current !== null && lastDocumentTextRef.current !== deferredContent;
        const debounceMs = isTextEdit ? 750 : 200;

        const timeoutId = window.setTimeout(async () => {
            setIsLoadingDocumentAttention(true);
            setAttentionError(null);

            try {
                const payload = sanitizeDocumentAttentionPayload(await postJson(
                    '/attention/document',
                    { text: deferredContent },
                    controller.signal,
                    'Attention analysis failed',
                ));
                if (latestDocumentRequestRef.current === requestId) {
                    lastDocumentTextRef.current = deferredContent;
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
        }, debounceMs);

        return () => {
            controller.abort();
            window.clearTimeout(timeoutId);
        };
    }, [activeTab, deferredContent, renderArcMode, renderSurfaceMode, shouldLoadDocumentAttention]);

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
                const payload = sanitizeSentencePairAttentionPayload(await postJson(
                    '/attention/pair',
                    {
                        text: deferredContent,
                        sentenceXIndex: selectedPair.sentenceXIndex,
                        sentenceYIndex: selectedPair.sentenceYIndex,
                    },
                    controller.signal,
                    'Token attention analysis failed',
                ));
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

    useEffect(() => {
        if (activeTab !== 'rendered' || renderSurfaceMode !== 'delta' || !deferredContent.trim()) {
            setIsLoadingDeltaVisualization(false);
            return;
        }

        const controller = new AbortController();
        const requestId = latestDeltaVisualizationRequestRef.current + 1;
        latestDeltaVisualizationRequestRef.current = requestId;
        const isTextEdit = lastDeltaTextRef.current !== null && lastDeltaTextRef.current !== deferredContent;
        const debounceMs = isTextEdit ? 750 : 200;

        const timeoutId = window.setTimeout(async () => {
            setIsLoadingDeltaVisualization(true);
            setDeltaVisualizationError(null);

            try {
                const payload = sanitizeDeltaVisualizationPayload(await postJson(
                    '/delta/visualization',
                    { text: deferredContent },
                    controller.signal,
                    'Delta visualization failed',
                ));
                if (latestDeltaVisualizationRequestRef.current === requestId) {
                    lastDeltaTextRef.current = deferredContent;
                    startTransition(() => {
                        setDeltaVisualization(payload);
                        setDeltaVisualizationError(null);
                    });
                }
            } catch (error) {
                if (controller.signal.aborted) {
                    return;
                }

                if (latestDeltaVisualizationRequestRef.current === requestId) {
                    startTransition(() => {
                        setDeltaVisualization(null);
                        setSelectedDeltaSentenceIndex(null);
                        setSelectedDeltaEdgeId(null);
                        setDeltaVisualizationError(error instanceof Error ? error.message : 'Delta visualization failed.');
                    });
                }
            } finally {
                if (!controller.signal.aborted && latestDeltaVisualizationRequestRef.current === requestId) {
                    setIsLoadingDeltaVisualization(false);
                }
            }
        }, debounceMs);

        return () => {
            controller.abort();
            window.clearTimeout(timeoutId);
        };
    }, [activeTab, deferredContent, renderSurfaceMode]);

    const previewActivity = previewActivityState(
        activeTab,
        renderSurfaceMode,
        isAnalyzing,
        isLoadingDocumentAttention,
        isLoadingPairAttention,
        isLoadingDeltaVisualization,
        renderArcMode,
    );

    const handleSelectPair = ({ sentenceXIndex, sentenceYIndex }: { sentenceXIndex: number; sentenceYIndex: number }) => {
        startTransition(() => {
            setSelectedPair({ sentenceXIndex, sentenceYIndex });
            setPairAttention(null);
            setActiveTab('token');
        });
    };

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

            const sourceIndex = Math.min(...manualArcSelection);
            const targetIndex = Math.max(...manualArcSelection);
            const connection = buildArcConnection(
                sourceIndex,
                targetIndex,
                documentAttention.sentenceAttention[targetIndex]?.[sourceIndex],
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
                return 'Click two different sentences. We will draw one earlier-to-later attention arc between them.';
            }
            if (manualArcSelection.length === 1) {
                return `Sentence ${manualArcSelection[0]} selected. Choose one more sentence to complete the arc.`;
            }
            return 'Two sentences selected. Click the arc itself if you want to reveal its score.';
        }

        if (renderArcMode === 'focused') {
            if (focusedArcSentence == null) {
                return 'Click a later sentence to reveal its strongest incoming sources.';
            }
            if (!renderArcConnections.length) {
                if (focusedLongDistanceOptions.enabled) {
                    return `Sentence ${focusedArcSentence} has no incoming sources beyond a ${focusedLongDistanceOptions.minimumGap}-sentence gap.`;
                }
                return `Sentence ${focusedArcSentence} has no earlier sources to draw yet.`;
            }
            if (focusedLongDistanceOptions.enabled) {
                return `Showing the top ${renderArcConnections.length} long-distance sources flowing into sentence ${focusedArcSentence}, excluding gaps of ${focusedLongDistanceOptions.minimumGap} sentences or less.`;
            }
            return `Showing the top ${renderArcConnections.length} earlier sources flowing into sentence ${focusedArcSentence}.`;
        }

        if (!renderArcConnections.length) {
            if (globalLongDistanceOptions.enabled) {
                return `No global arcs remain beyond a ${globalLongDistanceOptions.minimumGap}-sentence gap.`;
            }
            return 'No global earlier-to-later attention arcs are available for this document yet.';
        }

        if (globalLongDistanceOptions.enabled) {
            return `Showing the strongest ${renderArcConnections.length} long-distance attention arcs, excluding gaps of ${globalLongDistanceOptions.minimumGap} sentences or less.`;
        }

        return `Showing the strongest ${renderArcConnections.length} earlier-to-later attention arcs in the document.`;
    }, [
        attentionError,
        documentAttention,
        focusedArcSentence,
        focusedLongDistanceOptions,
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

    const selectedDeltaSentence = useMemo(() => (
        selectedDeltaSentenceIndex == null
            ? null
            : (deltaVisualization?.sentences.find((sentence) => sentence.index === selectedDeltaSentenceIndex) ?? null)
    ), [deltaVisualization, selectedDeltaSentenceIndex]);

    const deltaConnections = useMemo(() => {
        if (!deltaVisualization) {
            return [];
        }

        if (deltaVisualizationMode === 'flow-map' || deltaVisualizationMode === 'composition') {
            if (selectedDeltaSentence) {
                return selectedDeltaSentence.topSources
                    .filter((source) => source.visible && selectedDeltaSentence.visible)
                    .map((source) => ({
                        id: `${source.sourceIndex}:${selectedDeltaSentence.index}`,
                        sourceIndex: source.sourceIndex,
                        targetIndex: selectedDeltaSentence.index,
                        score: source.scoreNorm,
                        color: attentionColorForWeight(source.scoreNorm),
                        label: `${source.displaySentence} -> ${selectedDeltaSentence.displaySentence} · ${source.score.toFixed(3)}`,
                    }))
                    .sort((left, right) => right.score - left.score);
            }

            return deltaVisualization.globalTopEdges.map((edge) => ({
                id: edge.id,
                sourceIndex: edge.sourceIndex,
                targetIndex: edge.targetIndex,
                score: edge.scoreNorm,
                color: edge.color,
                label: `${edge.sourceIndex} -> ${edge.targetIndex} · ${edge.score.toFixed(3)}`,
            }));
        }

        return [];
    }, [deltaVisualization, deltaVisualizationMode, selectedDeltaSentence]);

    const activeDeltaArc = useMemo(() => (
        deltaConnections.find((connection) => connection.id === selectedDeltaEdgeId) ?? null
    ), [deltaConnections, selectedDeltaEdgeId]);

    const selectedDeltaIndices = useMemo(() => {
        const selected = new Set<number>();

        if (selectedDeltaSentenceIndex != null) {
            selected.add(selectedDeltaSentenceIndex);
        }

        deltaConnections.forEach((connection) => {
            if (selectedDeltaSentenceIndex != null && connection.targetIndex === selectedDeltaSentenceIndex) {
                selected.add(connection.sourceIndex);
                selected.add(connection.targetIndex);
            }
        });

        if (activeDeltaArc) {
            selected.add(activeDeltaArc.sourceIndex);
            selected.add(activeDeltaArc.targetIndex);
        }

        return [...selected];
    }, [activeDeltaArc, deltaConnections, selectedDeltaSentenceIndex]);

    const deltaInspectorSources = selectedDeltaSentence?.topSources.filter((source) => source.visible).slice(0, 4) ?? [];

    const deltaHelperText = useMemo(() => {
        if (deltaVisualizationError && !deltaVisualization) {
            return deltaVisualizationError;
        }

        if (isLoadingDeltaVisualization && !deltaVisualization) {
            return 'Capturing delta writes and deriving the current sentence flow surface...';
        }

        if (!deltaVisualization) {
            return 'Open the delta surface to derive sentence flow, composition, drift, and context routing directly from the model writes.';
        }

        if (deltaVisualizationMode === 'local-global') {
            return 'Each visible sentence is filled by its local, medium-range, and long-range write mix. Click a sentence to inspect its strongest contributors.';
        }

        if (deltaVisualizationMode === 'drift') {
            return 'Sentences with stronger cross-layer updates glow harder. Click any sentence to inspect where its rewrite pressure peaks.';
        }

        if (selectedDeltaSentence) {
            return `Showing the strongest source sentences writing into ${selectedDeltaSentence.displaySentence}.`;
        }

        return 'Click a sentence to trace its strongest upstream source sentences across the rendered document.';
    }, [
        deltaVisualization,
        deltaVisualizationError,
        deltaVisualizationMode,
        isLoadingDeltaVisualization,
        selectedDeltaSentence,
    ]);

    const handleRenderedDeltaSentenceClick = (sentenceIndex: number) => {
        setSelectedDeltaEdgeId(null);
        setSelectedDeltaSentenceIndex((currentIndex) => currentIndex === sentenceIndex ? null : sentenceIndex);
    };

    return (
        <div className="app-shell">
            <main className="workspace">
                <header className="hero">
                    <div className="hero-brand">
                        <h1>Attention-Locator</h1>
                        <p className="hero-subtitle">Inspect how a causal language model routes sentence and token attention across text</p>
                    </div>

                    <div className="hero-aside">
                        <aside className="legend-showcase" aria-label="Attention color legend">
                            <div className="legend-showcase-header">
                                <span className="panel-label">Attention spectrum</span>
                                <div className="attention-legend-bar" />
                            </div>
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

                <section className="preview-panel glass-panel">
                    <div className="preview-tab-bar" role="tablist" aria-label="Attention explorer views">
                        <div className="preview-tab-group preview-tab-group-pinned" role="group" aria-label="Rendered surface mode options">
                            {RENDER_SURFACE_MODES.map((mode) => (
                                <button
                                    key={mode.id}
                                    type="button"
                                    aria-pressed={renderSurfaceMode === mode.id}
                                    className={`preview-tab preview-tab-pinned ${renderSurfaceMode === mode.id ? 'preview-tab-pinned-active' : ''}`}
                                    onClick={() => {
                                        setRenderSurfaceMode(mode.id);
                                        if (activeTab !== 'rendered') {
                                            setActiveTab('rendered');
                                        }
                                    }}
                                >
                                    {mode.label}
                                </button>
                            ))}
                        </div>

                        <span className="preview-tab-divider" aria-hidden="true" />

                        <div className="preview-tab-group">
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

                        {previewActivity ? (
                            <div className="preview-tab-status" aria-live="polite">
                                <LoaderCircle size={14} strokeWidth={1.75} className="spin-icon" />
                                <span>{previewActivity.title}</span>
                            </div>
                        ) : null}
                    </div>

                    {analysisError ? (
                        <div className="status-banner">
                            <strong>Semantic analysis is unavailable</strong>
                            <span>{analysisError}</span>
                        </div>
                    ) : null}

                    {health?.error && !health.ready ? (
                        <div className="status-banner">
                            <strong>Backend health warning</strong>
                            <span>{health.error}</span>
                        </div>
                    ) : null}

                    <div className="preview-surface">
                            {activeTab === 'rendered' ? (
                                <div className="rendered-preview-stack">
                                    {renderSurfaceMode === 'semantic' ? (
                                        <>
                                            <section className="arc-toolbar" aria-label="Sentence arc controls">
                                                <div className="arc-mode-group" role="group" aria-label="Arc overlay modes">
                                                    <button
                                                        type="button"
                                                        className={`arc-mode-button ${renderArcMode === 'manual' ? 'arc-mode-button-active' : ''}`}
                                                        onClick={() => setRenderArcMode('manual')}
                                                    >
                                                        <GitBranch size={16} strokeWidth={1.5} />
                                                        <span className="arc-mode-button-label">Pair select</span>
                                                    </button>
                                                    <button
                                                        type="button"
                                                        className={`arc-mode-button ${renderArcMode === 'focused' ? 'arc-mode-button-active' : ''}`}
                                                        onClick={() => setRenderArcMode('focused')}
                                                    >
                                                        <Target size={16} strokeWidth={1.5} />
                                                        <span className="arc-mode-button-label">Top-k from sentence</span>
                                                    </button>
                                                    <button
                                                        type="button"
                                                        className={`arc-mode-button ${renderArcMode === 'global' ? 'arc-mode-button-active' : ''}`}
                                                        onClick={() => setRenderArcMode('global')}
                                                    >
                                                        <Globe size={16} strokeWidth={1.5} />
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
                                                    <div className="arc-controls-row">
                                                        <label className="arc-slider-card">
                                                            <div className="arc-slider-heading">
                                                                <span>
                                                                    <SlidersHorizontal size={16} strokeWidth={1.5} />
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
                                                                <div className="arc-filter-actions">
                                                                    <strong className="arc-filter-value">{currentLongDistanceFactor}</strong>
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
                                                            </div>

                                                            <label className={`arc-filter-slider-wrap ${isCurrentLongDistanceEnabled ? '' : 'arc-filter-slider-wrap-disabled'}`}>
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
                                                    </div>
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

                                            <Render
                                                content={content}
                                                onChange={setContent}
                                                placeholder="Paste markdown here..."
                                                semanticWeights={semanticWeights}
                                                sentenceArcConnections={renderArcConnections}
                                                selectedSentenceIndices={renderArcSelectedSentences}
                                                activeArcId={selectedArcId}
                                                isSentenceSelectionEnabled={renderArcMode === 'manual' || renderArcMode === 'focused'}
                                                onSentenceClick={(renderArcMode === 'manual' || renderArcMode === 'focused') ? handleRenderedSentenceClick : undefined}
                                                onArcClick={(arcId) => setSelectedArcId((currentId) => currentId === arcId ? null : arcId)}
                                            />
                                        </>
                                    ) : (
                                        <>
                                            <section className="delta-toolbar" aria-label="Delta embedding visualization controls">
                                                <div className="delta-toolbar-header">
                                                    <div>
                                                        <p className="panel-label">Delta visualization</p>
                                                        <h3>Sentence flow through residual writes</h3>
                                                    </div>
                                                </div>

                                                <div className="delta-mode-grid" role="group" aria-label="Delta visualization modes">
                                                    {DELTA_RENDER_MODES.map((mode) => (
                                                        <button
                                                            key={mode.id}
                                                            type="button"
                                                            className={`delta-mode-button ${deltaVisualizationMode === mode.id ? 'delta-mode-button-active' : ''}`}
                                                            onClick={() => {
                                                                setDeltaVisualizationMode(mode.id);
                                                                setSelectedDeltaEdgeId(null);
                                                            }}
                                                        >
                                                            {mode.icon}
                                                            <span>{mode.label}</span>
                                                            <small>{mode.description}</small>
                                                        </button>
                                                    ))}
                                                </div>

                                                <div className="delta-toolbar-footer">
                                                    <div className="arc-toolbar-copy">
                                                        <span>{deltaHelperText}</span>
                                                        {activeDeltaArc ? (
                                                            <div className="arc-selection-banner" role="status" aria-live="polite">
                                                                <span className="arc-selection-dot" style={{ backgroundColor: activeDeltaArc.color }} />
                                                                <strong>Selected flow</strong>
                                                                <span>{activeDeltaArc.label}</span>
                                                            </div>
                                                        ) : null}
                                                    </div>
                                                    {(selectedDeltaSentenceIndex != null || selectedDeltaEdgeId) ? (
                                                        <button
                                                            type="button"
                                                            className="arc-clear-button"
                                                            onClick={() => {
                                                                setSelectedDeltaSentenceIndex(null);
                                                                setSelectedDeltaEdgeId(null);
                                                            }}
                                                        >
                                                            Clear selection
                                                        </button>
                                                    ) : null}
                                                </div>

                                                {(deltaVisualization?.warnings.length || selectedDeltaSentence) ? (
                                                    <div className="delta-inspector-grid">
                                                        {selectedDeltaSentence ? (
                                                            <section className="delta-inspector-card" aria-label="Selected sentence inspector">
                                                                <div className="delta-inspector-header">
                                                                    <strong>{selectedDeltaSentence.displaySentence}</strong>
                                                                    <span>Sentence {selectedDeltaSentence.index}</span>
                                                                </div>
                                                                <div className="delta-inspector-metrics">
                                                                    <span>Flow {selectedDeltaSentence.incomingFlowNorm.toFixed(3)}</span>
                                                                    <span>Drift {selectedDeltaSentence.driftNorm.toFixed(3)}</span>
                                                                    <span>Peak layer {selectedDeltaSentence.peakLayer + 1}</span>
                                                                </div>
                                                                {deltaInspectorSources.length ? (
                                                                    <div className="delta-source-badges">
                                                                        {deltaInspectorSources.map((source) => (
                                                                            <button
                                                                                key={`${selectedDeltaSentence.index}-${source.sourceIndex}`}
                                                                                type="button"
                                                                                className="delta-source-badge"
                                                                                onClick={() => {
                                                                                    setSelectedDeltaSentenceIndex(source.sourceIndex);
                                                                                    setSelectedDeltaEdgeId(null);
                                                                                }}
                                                                            >
                                                                                <span>{source.displaySentence}</span>
                                                                                <small>{source.distanceBucket} · {source.scoreNorm.toFixed(2)}</small>
                                                                            </button>
                                                                        ))}
                                                                    </div>
                                                                ) : null}
                                                            </section>
                                                        ) : null}

                                                        {deltaVisualization?.warnings.map((warning) => (
                                                            <div key={warning} className="attention-warning-card">
                                                                <strong>Runtime note</strong>
                                                                <span>{warning}</span>
                                                            </div>
                                                        ))}
                                                    </div>
                                                ) : null}
                                            </section>

                                            {deltaVisualizationError ? (
                                                <div className="status-banner">
                                                    <strong>Delta visualization is unavailable.</strong>
                                                    <span>{deltaVisualizationError}</span>
                                                </div>
                                            ) : null}

                                            <Render
                                                content={content}
                                                onChange={setContent}
                                                placeholder="Paste markdown here..."
                                                deltaVisualization={deltaVisualization ? { mode: deltaVisualizationMode, sentences: deltaVisualization.sentences } : null}
                                                sentenceArcConnections={deltaConnections}
                                                selectedSentenceIndices={selectedDeltaIndices}
                                                activeArcId={selectedDeltaEdgeId}
                                                isSentenceSelectionEnabled
                                                onDeltaSentenceClick={handleRenderedDeltaSentenceClick}
                                                onArcClick={(arcId) => setSelectedDeltaEdgeId((currentId) => currentId === arcId ? null : arcId)}
                                            />
                                        </>
                                    )}
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
            </main>
        </div>
    );
};

export default App;
