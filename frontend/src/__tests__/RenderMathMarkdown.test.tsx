import { describe, expect, it, vi } from 'vitest';

vi.mock('../Shiki', () => ({
    highlightCodeSync: (code: string) => code,
    isLanguageSupported: () => true,
    resolveLanguage: (language: string) => language || 'plaintext',
    isHighlighterReady: () => true,
    onHighlighterReady: () => () => undefined,
}));

import { renderMathContent } from '../../RenderMathMarkdownLogic';
import type { SemanticWeight } from '../types/semantic';

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
            },
        ];

        const html = renderMathContent('Semantic focus belongs here.', weights);

        expect(html).toContain('data-semantic-band="high"');
        expect(html).toContain('data-semantic-index="0"');
    });

    it('does not inject semantic spans inside code fences', () => {
        const weights: SemanticWeight[] = [
            {
                index: 0,
                sentence: 'const a = 1;',
                normalizedSentence: 'const a = 1;',
                weight: 0.88,
                band: 'high',
            },
        ];

        const html = renderMathContent('```ts\nconst a = 1;\n```', weights);

        expect(html).not.toContain('data-semantic-band');
        expect(html).toContain('code-block-container');
    });
});
