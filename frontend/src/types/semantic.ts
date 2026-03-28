export type SemanticBand = 'low' | 'medium' | 'high';

export interface SemanticWeight {
    index: number;
    sentence: string;
    normalizedSentence: string;
    weight: number;
    band: SemanticBand;
}

export interface HealthPayload {
    ready: boolean;
    modelName: string;
    device: string;
    maxInputTokens?: number | null;
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
