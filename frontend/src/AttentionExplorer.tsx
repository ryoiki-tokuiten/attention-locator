import React from 'react';

import type { DocumentAttentionPayload, SentencePairAttentionPayload } from './types/semantic';
import { asFiniteNumber, attentionColorForWeight, clamp } from './utils';

export type AttentionTab = 'rendered' | 'sentence' | 'token';

interface SentencePairSelection {
    sentenceXIndex: number;
    sentenceYIndex: number;
}

interface AttentionExplorerProps {
    activeTab: AttentionTab;
    documentAttention: DocumentAttentionPayload | null;
    pairAttention: SentencePairAttentionPayload | null;
    isLoadingDocument: boolean;
    isLoadingPair: boolean;
    error: string | null;
    onSelectPair: (selection: SentencePairSelection) => void;
}

function tokenLabel(token: string) {
    return token
        .replace(/Ġ/g, '␠')
        .replace(/▁/g, '␠')
        .replace(/\n/g, '\\n');
}

const EmptyState: React.FC<{ message: string }> = ({ message }) => (
    <div className="attention-empty-state">
        <strong>Attention data is waiting.</strong>
        <span>{message}</span>
    </div>
);

const SentenceAttentionView: React.FC<{
    documentAttention: DocumentAttentionPayload | null;
    isLoading: boolean;
    error: string | null;
    onSelectPair: (selection: SentencePairSelection) => void;
}> = ({ documentAttention, isLoading, error, onSelectPair }) => {
    if (error) {
        return <EmptyState message={error} />;
    }

    if (isLoading && !documentAttention) {
        return <EmptyState message="Computing full-document attention with the SAVIS pipeline..." />;
    }

    if (!documentAttention || !documentAttention.sentenceCount) {
        return <EmptyState message="Open the sentence view after entering text to inspect inter-sentence attention." />;
    }

    return (
        <div className="attention-explorer-content">
            <div className="attention-summary-card">
                <strong>{documentAttention.sentenceCount} sentences · {documentAttention.tokenCount} tokens</strong>
                <span>Click any cell to inspect the token-level attention between that sentence pair.</span>
            </div>

            {documentAttention.warnings.map((warning) => (
                <div key={warning} className="attention-warning-card">
                    <strong>Runtime note</strong>
                    <span>{warning}</span>
                </div>
            ))}

            <div className="attention-matrix-wrapper">
                <table className="attention-matrix">
                    <thead>
                        <tr>
                            <th>y/x</th>
                            {documentAttention.sentenceWeights.map((sentence) => (
                                <th key={`column-${sentence.index}`}>{sentence.index}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {documentAttention.sentenceAttention.map((row, rowIndex) => (
                            <tr key={`row-${rowIndex}`}>
                                <th>{rowIndex}</th>
                                {row.map((value, columnIndex) => {
                                    const normalizedValue = clamp(asFiniteNumber(value));

                                    return (
                                        <td key={`cell-${rowIndex}-${columnIndex}`}>
                                        <button
                                            type="button"
                                            className="attention-cell-button"
                                            style={{
                                                backgroundColor: attentionColorForWeight(normalizedValue),
                                                opacity: 0.24 + (normalizedValue * 0.76),
                                            }}
                                            title={`Sentence ${columnIndex} -> sentence ${rowIndex}: ${normalizedValue.toFixed(3)}`}
                                            onClick={() => onSelectPair({ sentenceXIndex: columnIndex, sentenceYIndex: rowIndex })}
                                        >
                                            <span>{normalizedValue.toFixed(2)}</span>
                                        </button>
                                    </td>
                                    );
                                })}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            <div className="attention-sentence-list">
                {documentAttention.sentenceWeights.map((sentence) => (
                    <div key={`sentence-${sentence.index}`} className="attention-sentence-item">
                        <strong>{sentence.index}</strong>
                        <span>{sentence.sentence}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

const TokenAttentionView: React.FC<{
    pairAttention: SentencePairAttentionPayload | null;
    isLoading: boolean;
    error: string | null;
}> = ({ pairAttention, isLoading, error }) => {
    if (error) {
        return <EmptyState message={error} />;
    }

    if (isLoading && !pairAttention) {
        return <EmptyState message="Loading token-level attention for the selected sentence pair..." />;
    }

    if (!pairAttention) {
        return <EmptyState message="Choose a cell from the sentence matrix first to open the token-level view." />;
    }

    return (
        <div className="attention-explorer-content">
            <div className="attention-summary-card">
                <strong>Sentence {pairAttention.sentenceXIndex}{' -> '}sentence {pairAttention.sentenceYIndex}</strong>
                <span>{pairAttention.sentenceX}</span>
                <span>{pairAttention.sentenceY}</span>
            </div>

            {pairAttention.warnings.map((warning) => (
                <div key={warning} className="attention-warning-card">
                    <strong>Runtime note</strong>
                    <span>{warning}</span>
                </div>
            ))}

            <div className="token-matrix-wrapper">
                <table className="token-matrix">
                    <thead>
                        <tr>
                            <th>y/x</th>
                            {pairAttention.columnTokens.map((token, tokenIndex) => (
                                <th key={`col-token-${tokenIndex}`} title={token}>{tokenLabel(token)}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {pairAttention.attention.map((row, rowIndex) => (
                            <tr key={`token-row-${rowIndex}`}>
                                <th title={pairAttention.rowTokens[rowIndex]}>{tokenLabel(pairAttention.rowTokens[rowIndex])}</th>
                                {row.map((value, columnIndex) => {
                                    const normalizedValue = clamp(asFiniteNumber(value));

                                    return (
                                        <td
                                        key={`token-cell-${rowIndex}-${columnIndex}`}
                                        style={{
                                            backgroundColor: attentionColorForWeight(normalizedValue),
                                            opacity: 0.22 + (normalizedValue * 0.78),
                                        }}
                                        title={`${pairAttention.columnTokens[columnIndex]} -> ${pairAttention.rowTokens[rowIndex]}: ${normalizedValue.toFixed(4)}`}
                                    >
                                        {normalizedValue.toFixed(2)}
                                    </td>
                                    );
                                })}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export const AttentionExplorer: React.FC<AttentionExplorerProps> = ({
    activeTab,
    documentAttention,
    pairAttention,
    isLoadingDocument,
    isLoadingPair,
    error,
    onSelectPair,
}) => {
    if (activeTab === 'sentence') {
        return (
            <SentenceAttentionView
                documentAttention={documentAttention}
                isLoading={isLoadingDocument}
                error={error}
                onSelectPair={onSelectPair}
            />
        );
    }

    if (activeTab === 'token') {
        return (
            <TokenAttentionView
                pairAttention={pairAttention}
                isLoading={isLoadingPair}
                error={error}
            />
        );
    }

    return null;
};

export default AttentionExplorer;
