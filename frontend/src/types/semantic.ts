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
    error: string | null;
}
