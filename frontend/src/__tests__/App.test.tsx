import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../Shiki', () => ({
    highlightCodeSync: (code: string) => code,
    isLanguageSupported: () => true,
    resolveLanguage: (language: string) => language || 'plaintext',
    isHighlighterReady: () => true,
    onHighlighterReady: () => () => undefined,
}));

import App from '../App';

function deferredResponse<T>() {
    let resolve!: (value: T) => void;
    let reject!: (error?: unknown) => void;

    const promise = new Promise<T>((res, rej) => {
        resolve = res;
        reject = rej;
    });

    return { promise, resolve, reject };
}

describe('App', () => {
    beforeEach(() => {
        vi.restoreAllMocks();
    });

    afterEach(() => {
        vi.unstubAllGlobals();
    });

    it('falls back gracefully when the backend analyze call fails', async () => {
        vi.stubGlobal('fetch', vi.fn()
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({ ready: true, modelName: 'stub', device: 'cpu', error: null }),
            })
            .mockResolvedValueOnce({
                ok: false,
                json: async () => ({ detail: 'backend down' }),
            }));

        render(<App />);

        await act(async () => {
            await new Promise((resolve) => setTimeout(resolve, 900));
        });

        await waitFor(() => {
            expect(screen.getByText(/Semantic analysis is unavailable/i)).toBeInTheDocument();
        }, { timeout: 4000 });

        expect(screen.getByText(/backend down/i)).toBeInTheDocument();
    }, 8000);

    it('keeps the UI stable while newer analysis requests replace older ones', async () => {
        const first = deferredResponse<{ ok: boolean; json: () => Promise<unknown> }>();
        const second = deferredResponse<{ ok: boolean; json: () => Promise<unknown> }>();

        const fetchMock = vi.fn()
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({ ready: true, modelName: 'stub', device: 'cpu', error: null }),
            })
            .mockImplementationOnce(() => first.promise)
            .mockImplementationOnce(() => second.promise)
            .mockResolvedValue({
                ok: true,
                json: async () => ([]),
            });

        vi.stubGlobal('fetch', fetchMock);

        render(<App />);

        const textarea = screen.getByPlaceholderText(/Paste markdown here/i);

        await act(async () => {
            fireEvent.change(textarea, { target: { value: '# First revision\n\nOne sentence.' } });
            await new Promise((resolve) => setTimeout(resolve, 900));
        });

        await act(async () => {
            fireEvent.change(textarea, { target: { value: '# Second revision\n\nNew value.' } });
            await new Promise((resolve) => setTimeout(resolve, 900));
        });

        second.resolve({
            ok: true,
            json: async () => ([{
                index: 0,
                sentence: 'new value',
                normalizedSentence: 'new value',
                weight: 0.9,
                band: 'high',
            }]),
        });

        first.resolve({
            ok: true,
            json: async () => ([{
                index: 0,
                sentence: 'old value',
                normalizedSentence: 'old value',
                weight: 0.2,
                band: 'low',
            }]),
        });

        await waitFor(() => {
            expect(screen.getByText(/Model ready/i)).toBeInTheDocument();
        }, { timeout: 4000 });

        expect(screen.queryByText(/Semantic analysis is unavailable/i)).not.toBeInTheDocument();
        expect(screen.getAllByText(/Second revision/i).length).toBeGreaterThan(0);
    }, 8000);
});
