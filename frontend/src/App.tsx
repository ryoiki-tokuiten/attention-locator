import React, { startTransition, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react';
import { Compass, FileText, GitBranch, Globe, LoaderCircle, PanelLeftClose, PanelLeftOpen, SlidersHorizontal, SwatchBook, Target } from 'lucide-react';

import { GlobalImagePreviewModal, RenderMathMarkdown, type AttentionArcConnection } from '../RenderMathMarkdown';
import AttentionExplorer, { attentionColorForWeight, type AttentionTab } from './AttentionExplorer';
import DeltaVectorFlowCanvas from './DeltaVectorFlowCanvas';
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

const RENDER_SURFACE_MODES = [
    { id: 'semantic', label: 'Semantic surface' },
    { id: 'delta', label: 'Delta embedding based visualization' },
] as const;

const DELTA_RENDER_MODES: Array<{ id: DeltaVisualizationMode; label: string; icon: React.ReactNode; description: string }> = [
    {
        id: 'vector-flow',
        label: 'Vector flow',
        icon: <Compass size={15} />,
        description: 'Projects each sentence and its attention delta into 2D, scales them with bounded lengths, and chains the composed vectors across the canvas.',
    },
    {
        id: 'local-global',
        label: 'Local vs global',
        icon: <Globe size={15} />,
        description: 'Shows whether a sentence is shaped more by nearby structure or longer-range context.',
    },
    {
        id: 'flow-map',
        label: 'Flow map',
        icon: <GitBranch size={15} />,
        description: 'Draws sentence-to-sentence write paths so you can follow where context is flowing from.',
    },
    {
        id: 'composition',
        label: 'Composition',
        icon: <SwatchBook size={15} />,
        description: 'Colors each sentence by the strongest source sentences that most contributed to it.',
    },
    {
        id: 'drift',
        label: 'Drift',
        icon: <Target size={15} />,
        description: 'Highlights sentences whose representations are being rewritten most strongly across depth.',
    },
] as const;

type RenderSurfaceMode = typeof RENDER_SURFACE_MODES[number]['id'];

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

function asDeltaMode(value: unknown): DeltaVisualizationMode {
    return value === 'vector-flow' || value === 'flow-map' || value === 'composition' || value === 'drift' || value === 'local-global'
        ? value
        : 'vector-flow';
}

function sanitizeDeltaVisualizationPayload(payload: unknown): DeltaVisualizationPayload {
    const candidate = payload && typeof payload === 'object' ? payload as Record<string, unknown> : {};
    const sentences = Array.isArray(candidate.sentences) ? candidate.sentences : [];
    const globalTopEdges = Array.isArray(candidate.globalTopEdges) ? candidate.globalTopEdges : [];
    const driftByLayer = Array.isArray(candidate.driftByLayer) ? candidate.driftByLayer : [];

    return {
        analysisText: asString(candidate.analysisText),
        tokenCount: asFiniteNumber(candidate.tokenCount, 0),
        sentenceCount: asFiniteNumber(candidate.sentenceCount, 0),
        layerCount: asFiniteNumber(candidate.layerCount, 0),
        headCount: asFiniteNumber(candidate.headCount, 0),
        hiddenSize: asFiniteNumber(candidate.hiddenSize, 0),
        defaultMode: asDeltaMode(candidate.defaultMode),
        sentences: sentences.map((item, index) => {
            const sentenceCandidate = item && typeof item === 'object' ? item as Record<string, unknown> : {};
            const topSources = Array.isArray(sentenceCandidate.topSources) ? sentenceCandidate.topSources : [];
            return {
                index: asFiniteNumber(sentenceCandidate.index, index),
                sentence: asString(sentenceCandidate.sentence),
                displaySentence: asString(sentenceCandidate.displaySentence, asString(sentenceCandidate.sentence)),
                sourceStart: sentenceCandidate.sourceStart == null ? null : asFiniteNumber(sentenceCandidate.sourceStart, 0),
                sourceEnd: sentenceCandidate.sourceEnd == null ? null : asFiniteNumber(sentenceCandidate.sourceEnd, 0),
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
                vectorFlow: sentenceCandidate.vectorFlow && typeof sentenceCandidate.vectorFlow === 'object'
                    ? {
                        anchorX: asFiniteNumber((sentenceCandidate.vectorFlow as Record<string, unknown>).anchorX, 0),
                        anchorY: asFiniteNumber((sentenceCandidate.vectorFlow as Record<string, unknown>).anchorY, 0),
                        sentenceEndX: asFiniteNumber((sentenceCandidate.vectorFlow as Record<string, unknown>).sentenceEndX, 0),
                        sentenceEndY: asFiniteNumber((sentenceCandidate.vectorFlow as Record<string, unknown>).sentenceEndY, 0),
                        updatedEndX: asFiniteNumber((sentenceCandidate.vectorFlow as Record<string, unknown>).updatedEndX, 0),
                        updatedEndY: asFiniteNumber((sentenceCandidate.vectorFlow as Record<string, unknown>).updatedEndY, 0),
                        sentenceAngle: asFiniteNumber((sentenceCandidate.vectorFlow as Record<string, unknown>).sentenceAngle, 0),
                        deltaAngle: asFiniteNumber((sentenceCandidate.vectorFlow as Record<string, unknown>).deltaAngle, 0),
                        deltaRelativeAngle: asFiniteNumber((sentenceCandidate.vectorFlow as Record<string, unknown>).deltaRelativeAngle, 0),
                        composedAngle: asFiniteNumber((sentenceCandidate.vectorFlow as Record<string, unknown>).composedAngle, 0),
                        sentenceMagnitude: asFiniteNumber((sentenceCandidate.vectorFlow as Record<string, unknown>).sentenceMagnitude, 0),
                        deltaMagnitude: asFiniteNumber((sentenceCandidate.vectorFlow as Record<string, unknown>).deltaMagnitude, 0),
                        composedMagnitude: asFiniteNumber((sentenceCandidate.vectorFlow as Record<string, unknown>).composedMagnitude, 0),
                    }
                    : null,
                topSources: topSources.map((source, sourceIndex) => {
                    const sourceCandidate = source && typeof source === 'object' ? source as Record<string, unknown> : {};
                    const distanceBucket = sourceCandidate.distanceBucket === 'local' || sourceCandidate.distanceBucket === 'medium' || sourceCandidate.distanceBucket === 'global'
                        ? sourceCandidate.distanceBucket
                        : 'local';

                    return {
                        sourceIndex: asFiniteNumber(sourceCandidate.sourceIndex, sourceIndex),
                        score: asFiniteNumber(sourceCandidate.score, 0),
                        scoreNorm: clamp(asFiniteNumber(sourceCandidate.scoreNorm, 0)),
                        share: clamp(asFiniteNumber(sourceCandidate.share, 0)),
                        distance: asFiniteNumber(sourceCandidate.distance, 0),
                        distanceBucket,
                        sentence: asString(sourceCandidate.sentence),
                        displaySentence: asString(sourceCandidate.displaySentence, asString(sourceCandidate.sentence)),
                        sourceStart: sourceCandidate.sourceStart == null ? null : asFiniteNumber(sourceCandidate.sourceStart, 0),
                        sourceEnd: sourceCandidate.sourceEnd == null ? null : asFiniteNumber(sourceCandidate.sourceEnd, 0),
                        visible: Boolean(sourceCandidate.visible),
                    };
                }),
            };
        }),
        driftByLayer: driftByLayer.map((row) => (
            Array.isArray(row)
                ? row.map((value) => clamp(asFiniteNumber(value, 0)))
                : []
        )),
        globalTopEdges: globalTopEdges.map((edge, edgeIndex) => {
            const edgeCandidate = edge && typeof edge === 'object' ? edge as Record<string, unknown> : {};
            const distanceBucket = edgeCandidate.distanceBucket === 'local' || edgeCandidate.distanceBucket === 'medium' || edgeCandidate.distanceBucket === 'global'
                ? edgeCandidate.distanceBucket
                : 'local';

            return {
                id: asString(edgeCandidate.id, `edge-${edgeIndex}`),
                sourceIndex: asFiniteNumber(edgeCandidate.sourceIndex, 0),
                targetIndex: asFiniteNumber(edgeCandidate.targetIndex, 0),
                score: asFiniteNumber(edgeCandidate.score, 0),
                scoreNorm: clamp(asFiniteNumber(edgeCandidate.scoreNorm, 0)),
                distance: asFiniteNumber(edgeCandidate.distance, 0),
                distanceBucket,
                color: asString(edgeCandidate.color, attentionColorForWeight(asFiniteNumber(edgeCandidate.scoreNorm, 0))),
            };
        }),
        vectorFlow: candidate.vectorFlow && typeof candidate.vectorFlow === 'object'
            ? {
                projection: asString((candidate.vectorFlow as Record<string, unknown>).projection, 'pca'),
                angleAmplification: asFiniteNumber((candidate.vectorFlow as Record<string, unknown>).angleAmplification, 36),
                baseWeight: asFiniteNumber((candidate.vectorFlow as Record<string, unknown>).baseWeight, 1.2),
                deltaWeight: asFiniteNumber((candidate.vectorFlow as Record<string, unknown>).deltaWeight, 0.55),
                unitCircleRadius: asFiniteNumber((candidate.vectorFlow as Record<string, unknown>).unitCircleRadius, 1),
                bounds: {
                    minX: asFiniteNumber((((candidate.vectorFlow as Record<string, unknown>).bounds as Record<string, unknown>)?.minX), -1.5),
                    maxX: asFiniteNumber((((candidate.vectorFlow as Record<string, unknown>).bounds as Record<string, unknown>)?.maxX), 1.5),
                    minY: asFiniteNumber((((candidate.vectorFlow as Record<string, unknown>).bounds as Record<string, unknown>)?.minY), -1.5),
                    maxY: asFiniteNumber((((candidate.vectorFlow as Record<string, unknown>).bounds as Record<string, unknown>)?.maxY), 1.5),
                },
            }
            : {
                projection: 'pca',
                angleAmplification: 36,
                baseWeight: 1.2,
                deltaWeight: 0.55,
                unitCircleRadius: 1,
                bounds: { minX: -1.5, maxX: 1.5, minY: -1.5, maxY: 1.5 },
            },
        warnings: asStringArray(candidate.warnings),
    };
}

function previewPillLabel(
    activeTab: AttentionTab,
    renderSurfaceMode: RenderSurfaceMode,
    renderArcMode: RenderArcMode,
    semanticWeights: SemanticWeight[],
    documentAttention: DocumentAttentionPayload | null,
    pairAttention: SentencePairAttentionPayload | null,
    deltaVisualization: DeltaVisualizationPayload | null,
    analysisError: string | null,
    attentionError: string | null,
    deltaVisualizationError: string | null,
    isLoadingDocumentAttention: boolean,
    isLoadingPairAttention: boolean,
    isLoadingDeltaVisualization: boolean,
) {
    if (activeTab === 'rendered') {
        if (renderSurfaceMode === 'delta') {
            if (deltaVisualizationError) {
                return 'Delta visualization unavailable';
            }
            if (isLoadingDeltaVisualization) {
                return 'Building sentence flow surface';
            }
            return deltaVisualization ? `${deltaVisualization.sentenceCount} sentence surfaces mapped` : 'Delta surface ready on demand';
        }
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
    renderSurfaceMode: RenderSurfaceMode,
    isInputPending: boolean,
    isAnalyzing: boolean,
    isLoadingDocumentAttention: boolean,
    isLoadingPairAttention: boolean,
    isLoadingDeltaVisualization: boolean,
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

    if (activeTab === 'rendered' && renderSurfaceMode === 'delta' && isLoadingDeltaVisualization) {
        return {
            kicker: 'Delta capture',
            title: 'Building the sentence flow surface',
            description: 'We are capturing per-layer delta writes and deriving sentence-level flow summaries for the new rendering mode.',
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
    const [deltaVisualizationMode, setDeltaVisualizationMode] = useState<DeltaVisualizationMode>('vector-flow');
    const [selectedDeltaSentenceIndex, setSelectedDeltaSentenceIndex] = useState<number | null>(null);
    const [selectedDeltaEdgeId, setSelectedDeltaEdgeId] = useState<string | null>(null);
    const deferredContent = useDeferredValue(content);
    const latestRequestRef = useRef(0);
    const latestDocumentRequestRef = useRef(0);
    const latestPairRequestRef = useRef(0);
    const latestDeltaVisualizationRequestRef = useRef(0);

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
            setDeltaVisualization(null);
            setDeltaVisualizationError(null);
            setSelectedDeltaSentenceIndex(null);
            setSelectedDeltaEdgeId(null);
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

    useEffect(() => {
        if (activeTab !== 'rendered' || renderSurfaceMode !== 'delta' || !deferredContent.trim()) {
            setIsLoadingDeltaVisualization(false);
            return;
        }

        const controller = new AbortController();
        const requestId = latestDeltaVisualizationRequestRef.current + 1;
        latestDeltaVisualizationRequestRef.current = requestId;

        const timeoutId = window.setTimeout(async () => {
            setIsLoadingDeltaVisualization(true);
            setDeltaVisualizationError(null);

            try {
                const response = await fetch(`${API_BASE_URL}/delta/visualization`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: deferredContent }),
                    signal: controller.signal,
                });

                if (!response.ok) {
                    const errorBody = await response.json().catch(() => null) as { detail?: string } | null;
                    throw new Error(errorBody?.detail || `Delta visualization failed with status ${response.status}`);
                }

                const payload = sanitizeDeltaVisualizationPayload(await response.json());
                if (latestDeltaVisualizationRequestRef.current === requestId) {
                    startTransition(() => {
                        setDeltaVisualization(payload);
                        setSelectedDeltaSentenceIndex(null);
                        setSelectedDeltaEdgeId(null);
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
        }, 200);

        return () => {
            controller.abort();
            window.clearTimeout(timeoutId);
        };
    }, [activeTab, deferredContent, renderSurfaceMode]);

    const stats = useMemo(() => {
        const words = content.trim() ? content.trim().split(/\s+/).length : 0;

        return { words };
    }, [content]);

    const isInputPending = content !== deferredContent;
    const previewActivity = previewActivityState(
        activeTab,
        renderSurfaceMode,
        isInputPending,
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

    const currentPreviewPillLabel = previewPillLabel(
        activeTab,
        renderSurfaceMode,
        renderArcMode,
        semanticWeights,
        documentAttention,
        pairAttention,
        deltaVisualization,
        analysisError,
        attentionError,
        deltaVisualizationError,
        isLoadingDocumentAttention,
        isLoadingPairAttention,
        isLoadingDeltaVisualization,
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

        if (deltaVisualizationMode === 'vector-flow') {
            return `Each sentence is projected into 2D against a local unit-circle reference, then its attention delta is added and the composed update becomes the next anchor. Display angles are amplified ${deltaVisualization.vectorFlow.angleAmplification.toFixed(0)}x, while vector lengths are bounded from the sentence and delta magnitudes so the flow opens up without blowing out. Drag to pan and scroll to zoom.`;
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
                                    <section className="surface-mode-toolbar" aria-label="Rendered surface mode">
                                        <div className="surface-mode-group" role="group" aria-label="Rendered surface mode options">
                                            {RENDER_SURFACE_MODES.map((mode) => (
                                                <button
                                                    key={mode.id}
                                                    type="button"
                                                    className={`surface-mode-button ${renderSurfaceMode === mode.id ? 'surface-mode-button-active' : ''}`}
                                                    onClick={() => setRenderSurfaceMode(mode.id)}
                                                >
                                                    {mode.label}
                                                </button>
                                            ))}
                                        </div>
                                        <span className="surface-mode-caption">
                                            {renderSurfaceMode === 'semantic'
                                                ? 'Sentence-level surface styling from SAVIS attention.'
                                                : 'Sentence-level surface styling from cached delta writes.'}
                                        </span>
                                    </section>

                                    {renderSurfaceMode === 'semantic' ? (
                                        <>
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
                                        </>
                                    ) : (
                                        <>
                                            <section className="delta-toolbar" aria-label="Delta embedding visualization controls">
                                                <div className="delta-toolbar-header">
                                                    <div>
                                                        <p className="panel-label">Delta visualization</p>
                                                        <h3>Sentence flow through residual writes</h3>
                                                    </div>
                                                    <SwatchBook size={18} />
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
                                                                    {deltaVisualizationMode === 'vector-flow' && selectedDeltaSentence.vectorFlow ? (
                                                                        <>
                                                                            <span>Sentence {selectedDeltaSentence.vectorFlow.sentenceAngle.toFixed(1)}°</span>
                                                                            <span>Delta {selectedDeltaSentence.vectorFlow.deltaRelativeAngle.toFixed(1)}°</span>
                                                                            <span>Composed {selectedDeltaSentence.vectorFlow.composedMagnitude.toFixed(2)}</span>
                                                                            {deltaVisualization ? (
                                                                                <span>Visual {deltaVisualization.vectorFlow.angleAmplification.toFixed(0)}x</span>
                                                                            ) : null}
                                                                        </>
                                                                    ) : null}
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

                                            {deltaVisualizationMode === 'vector-flow' ? (
                                                <DeltaVectorFlowCanvas
                                                    visualization={deltaVisualization}
                                                    selectedSentenceIndex={selectedDeltaSentenceIndex}
                                                    onSelectSentence={handleRenderedDeltaSentenceClick}
                                                />
                                            ) : (
                                                <RenderMathMarkdown
                                                    content={content}
                                                    deltaVisualization={deltaVisualization ? { mode: deltaVisualizationMode, sentences: deltaVisualization.sentences } : null}
                                                    sentenceArcConnections={deltaConnections}
                                                    selectedSentenceIndices={selectedDeltaIndices}
                                                    activeArcId={selectedDeltaEdgeId}
                                                    isSentenceSelectionEnabled
                                                    onDeltaSentenceClick={handleRenderedDeltaSentenceClick}
                                                    onArcClick={(arcId) => setSelectedDeltaEdgeId((currentId) => currentId === arcId ? null : arcId)}
                                                />
                                            )}
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
                </section>
            </main>

            <GlobalImagePreviewModal />
        </div>
    );
};

export default App;
