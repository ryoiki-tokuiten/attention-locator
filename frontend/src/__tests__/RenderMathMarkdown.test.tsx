import { describe, expect, it, vi } from 'vitest';

vi.mock('../Shiki', () => ({
    highlightCodeSync: (code: string) => code,
    isLanguageSupported: () => true,
    resolveLanguage: (language: string) => language || 'plaintext',
    isHighlighterReady: () => true,
    onHighlighterReady: () => () => undefined,
}));

import { renderMathContent } from '../../RenderMathMarkdownLogic';
import type { DeltaVisualizationPayload, SemanticWeight } from '../types/semantic';

describe('renderMathContent', () => {
    it('renders markdown without semantic attributes when no weights are present', () => {
        const html = renderMathContent('Plain sentence. Another sentence.');

        expect(html).not.toContain('data-semantic-band');
        expect(html).toContain('Plain sentence.');
    });

    it('injects semantic spans for matched prose sentences', () => {
        const weights: SemanticWeight[] = [
            {
                index: 0,
                sentence: 'Semantic focus belongs here.',
                normalizedSentence: 'semantic focus belongs here.',
                weight: 0.9,
                band: 'high',
                sourceStart: 0,
                sourceEnd: 'Semantic focus belongs here.'.length,
            },
        ];

        const html = renderMathContent('Semantic focus belongs here.', weights);

        expect(html).toContain('data-semantic-band="high"');
        expect(html).toContain('data-semantic-index="0"');
    });

    it('keeps code fences renderable when semantic weights are present', () => {
        const weights: SemanticWeight[] = [
            {
                index: 0,
                sentence: 'const a = 1;',
                normalizedSentence: 'const a = 1;',
                weight: 0.88,
                band: 'high',
                sourceStart: 0,
                sourceEnd: '```ts\nconst a = 1;\n```'.length,
            },
        ];

        const html = renderMathContent('```ts\nconst a = 1;\n```', weights);

        expect(html).toContain('code-block-container');
    });

    it('ignores invalid semantic weights instead of crashing', () => {
        const html = renderMathContent('Semantic focus belongs here.', [{
            index: 0,
            sentence: 'Semantic focus belongs here.',
            normalizedSentence: 'semantic focus belongs here.',
            weight: null,
            band: 'high',
            sourceStart: null,
            sourceEnd: null,
        } as unknown as SemanticWeight]);

        expect(html).toContain('Semantic focus belongs here.');
    });

    it('injects delta spans for sentence-level visualization mode', () => {
        const deltaVisualization: DeltaVisualizationPayload = {
            analysisText: 'Delta surface sentence.',
            tokenCount: 3,
            sentenceCount: 1,
            layerCount: 2,
            headCount: 2,
            hiddenSize: 8,
            defaultMode: 'local-global',
            sentences: [
                {
                    index: 0,
                    sentence: 'Delta surface sentence.',
                    displaySentence: 'Delta surface sentence.',
                    sourceStart: 0,
                    sourceEnd: 'Delta surface sentence.'.length,
                    visible: true,
                    incomingFlow: 0.3,
                    incomingFlowNorm: 0.7,
                    localShare: 0.6,
                    mediumShare: 0.3,
                    globalShare: 0.1,
                    drift: 0.2,
                    driftNorm: 0.4,
                    peakLayer: 1,
                    peakLayerRatio: 1,
                    vectorFlow: {
                        anchorX: 0,
                        anchorY: 0,
                        sentenceEndX: 1.8,
                        sentenceEndY: 0,
                        updatedEndX: 2.6,
                        updatedEndY: 0.45,
                        sentenceAngle: 0,
                        deltaAngle: 36,
                        deltaRelativeAngle: 36,
                        composedAngle: 10,
                        sentenceMagnitude: 1.8,
                        deltaMagnitude: 0.92,
                        composedMagnitude: 2.638181,
                    },
                    topSources: [],
                },
            ],
            driftByLayer: [[0.1, 0.2]],
            globalTopEdges: [],
            vectorFlow: {
                projection: 'pca',
                angleAmplification: 36,
                baseWeight: 1.2,
                deltaWeight: 0.55,
                unitCircleRadius: 1,
                bounds: { minX: -1.5, maxX: 1.5, minY: -1.5, maxY: 1.5 },
            },
            warnings: [],
        };

        const html = renderMathContent(
            'Delta surface sentence.',
            [],
            { mode: 'local-global', sentences: deltaVisualization.sentences },
        );

        expect(html).toContain('data-delta-index="0"');
        expect(html).toContain('delta-token');
    });
});
