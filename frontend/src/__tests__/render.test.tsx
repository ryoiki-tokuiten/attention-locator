import { describe, expect, it } from 'vitest';

import { renderPlainContent } from '../../render';
import type { DeltaVisualizationPayload, SemanticWeight } from '../types/semantic';

describe('renderPlainContent', () => {
    it('renders plain text without semantic attributes when no weights are present', () => {
        const html = renderPlainContent('Plain sentence. Another sentence.');

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

        const html = renderPlainContent('Semantic focus belongs here.', weights);

        expect(html).toContain('data-semantic-band="high"');
        expect(html).toContain('data-semantic-index="0"');
    });

    it('keeps raw text intact when semantic weights are present', () => {
        const raw = 'First sentence.\nSecond sentence.';
        const weights: SemanticWeight[] = [
            {
                index: 0,
                sentence: 'First sentence.',
                normalizedSentence: 'first sentence.',
                weight: 0.88,
                band: 'high',
                sourceStart: 0,
                sourceEnd: 'First sentence.'.length,
            },
            {
                index: 1,
                sentence: 'Second sentence.',
                normalizedSentence: 'second sentence.',
                weight: 0.42,
                band: 'medium',
                sourceStart: 'First sentence.\n'.length,
                sourceEnd: raw.length,
            },
        ];

        const html = renderPlainContent(raw, weights);

        expect(html).toContain('data-semantic-index="0"');
        expect(html).toContain('data-semantic-index="1"');
        expect(html).toContain('>Second sentence.</span>');
    });

    it('ignores invalid semantic weights instead of crashing', () => {
        const html = renderPlainContent('Semantic focus belongs here.', [{
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
                    topSources: [],
                },
            ],
            driftByLayer: [[0.1, 0.2]],
            globalTopEdges: [],
            warnings: [],
        };

        const html = renderPlainContent(
            'Delta surface sentence.',
            [],
            { mode: 'local-global', sentences: deltaVisualization.sentences },
        );

        expect(html).toContain('data-delta-index="0"');
        expect(html).toContain('delta-token');
    });
});
