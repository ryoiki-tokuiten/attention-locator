export type SemanticBand = 'low' | 'medium' | 'high';

export interface SemanticWeight {
    index: number;
    sentence: string;
    normalizedSentence: string;
    weight: number;
    band: SemanticBand;
    sourceStart: number | null;
    sourceEnd: number | null;
}

export interface HealthPayload {
    ready: boolean;
    error: string | null;
}

export interface DocumentAttentionPayload {
    analysisText: string;
    tokenCount: number;
    sentenceCount: number;
    sentenceWeights: SemanticWeight[];
    sentenceAttention: number[][];
    warnings: string[];
}

export interface SentencePairAttentionPayload {
    sentenceXIndex: number;
    sentenceYIndex: number;
    sentenceX: string;
    sentenceY: string;
    columnTokens: string[];
    rowTokens: string[];
    attention: number[][];
    warnings: string[];
}

export type DeltaVisualizationMode = 'local-global' | 'flow-map' | 'composition' | 'drift';
export type DeltaDistanceBucket = 'local' | 'medium' | 'global';

export interface DeltaSourceContribution {
    sourceIndex: number;
    score: number;
    scoreNorm: number;
    share: number;
    distance: number;
    distanceBucket: DeltaDistanceBucket;
    sentence: string;
    displaySentence: string;
    sourceStart: number | null;
    sourceEnd: number | null;
    visible: boolean;
}

export interface DeltaSentenceVisualization {
    index: number;
    sentence: string;
    displaySentence: string;
    sourceStart: number | null;
    sourceEnd: number | null;
    visible: boolean;
    incomingFlow: number;
    incomingFlowNorm: number;
    localShare: number;
    mediumShare: number;
    globalShare: number;
    drift: number;
    driftNorm: number;
    peakLayer: number;
    peakLayerRatio: number;
    topSources: DeltaSourceContribution[];
}

export interface DeltaGlobalEdge {
    id: string;
    sourceIndex: number;
    targetIndex: number;
    score: number;
    scoreNorm: number;
    distance: number;
    distanceBucket: DeltaDistanceBucket;
    color: string;
}

export interface DeltaVisualizationPayload {
    analysisText: string;
    tokenCount: number;
    sentenceCount: number;
    layerCount: number;
    headCount: number;
    hiddenSize: number;
    defaultMode: DeltaVisualizationMode;
    sentences: DeltaSentenceVisualization[];
    driftByLayer: number[][];
    globalTopEdges: DeltaGlobalEdge[];
    vectorFlow?: {
        projection: string;
        angleAmplification: number;
        baseWeight: number;
        deltaWeight: number;
        unitCircleRadius: number;
        bounds: {
            minX: number;
            maxX: number;
            minY: number;
            maxY: number;
        };
    };
    warnings: string[];
}
