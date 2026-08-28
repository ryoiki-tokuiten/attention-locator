import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

import { onHighlighterReady, isHighlighterReady } from './src/Shiki';
import type { DeltaSentenceVisualization, DeltaVisualizationMode, SemanticWeight } from './src/types/semantic';
import { Icon, setIconSlot } from './src/UI/Icons';
import { renderMathContent } from './RenderMathMarkdownLogic';

export function useHighlighting() {
    const [isReady, setIsReady] = useState(isHighlighterReady());

    useEffect(() => {
        if (isReady) {
            return;
        }

        return onHighlighterReady(() => setIsReady(true));
    }, [isReady]);

    return isReady;
}

export interface ImagePreviewData {
    src: string;
    alt: string;
    format: string;
}

export const ImagePreviewModal: React.FC<{
    data: ImagePreviewData | null;
    onClose: () => void;
}> = ({ data, onClose }) => {
    const [isClosing, setIsClosing] = useState(false);

    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                handleClose();
            }
        };

        if (data) {
            document.body.style.overflow = 'hidden';
            document.addEventListener('keydown', handleKeyDown);
        }

        return () => {
            document.body.style.overflow = '';
            document.removeEventListener('keydown', handleKeyDown);
        };
    }, [data]);

    const handleClose = React.useCallback(() => {
        setIsClosing(true);
        setTimeout(() => {
            setIsClosing(false);
            onClose();
        }, 180);
    }, [onClose]);

    if (!data) {
        return null;
    }

    return createPortal(
        <div
            className="file-preview-modal-overlay"
            onClick={handleClose}
            style={isClosing ? { animation: 'fadeIn 0.2s ease reverse' } : undefined}
        >
            <div
                className="file-preview-modal"
                onClick={(event) => event.stopPropagation()}
                style={isClosing ? { animation: 'scaleIn 0.2s ease reverse' } : undefined}
            >
                <div className="preview-modal-header">
                    <div className="preview-file-info">
                        <Icon name="image" style={{ color: '#10b981' }} />
                        <span className="preview-file-name">Generated Figure</span>
                        <span className="preview-file-badge" style={{ backgroundColor: 'rgba(16, 185, 129, 0.13)', color: '#10b981' }}>
                            {data.format}
                        </span>
                    </div>
                    <button className="preview-close-btn" title="Close (Esc)" onClick={handleClose}>
                        <Icon name="close" />
                    </button>
                </div>
                <div className="preview-modal-content">
                    <div className="preview-image-container">
                        <img src={data.src} alt={data.alt} className="preview-image" />
                    </div>
                </div>
            </div>
        </div>,
        document.body
    );
};

export const GlobalImagePreviewModal: React.FC = () => {
    const [previewData, setPreviewData] = useState<ImagePreviewData | null>(null);

    useEffect(() => {
        const handlePreview = (event: Event) => {
            const customEvent = event as CustomEvent<ImagePreviewData>;
            if (customEvent.detail?.src) {
                setPreviewData(customEvent.detail);
            }
        };

        window.addEventListener('exec-image-preview', handlePreview);
        return () => window.removeEventListener('exec-image-preview', handlePreview);
    }, []);

    return <ImagePreviewModal data={previewData} onClose={() => setPreviewData(null)} />;
};

export interface RenderMathMarkdownProps {
    content: string;
    className?: string;
    semanticWeights?: SemanticWeight[];
    deltaVisualization?: {
        mode: DeltaVisualizationMode;
        sentences: DeltaSentenceVisualization[];
    } | null;
    sentenceArcConnections?: AttentionArcConnection[];
    selectedSentenceIndices?: number[];
    activeArcId?: string | null;
    isSentenceSelectionEnabled?: boolean;
    onSentenceClick?: (sentenceIndex: number) => void;
    onDeltaSentenceClick?: (sentenceIndex: number) => void;
    onArcClick?: (arcId: string) => void;
}

export interface AttentionArcConnection {
    id: string;
    sourceIndex: number;
    targetIndex: number;
    score: number;
    color: string;
    label: string;
}

function clampNumber(value: number, min: number, max: number) {
    return Math.min(max, Math.max(min, value));
}

function slotOffset(index: number, count: number, span: number) {
    if (count <= 1 || span <= 0) {
        return 0;
    }

    const safeCount = Math.max(1, count);
    const progress = safeCount === 1 ? 0.5 : index / (safeCount - 1);
    return ((progress - 0.5) * span);
}

function cubicPoint(
    startX: number,
    startY: number,
    control1X: number,
    control1Y: number,
    control2X: number,
    control2Y: number,
    endX: number,
    endY: number,
    t: number,
) {
    const inverse = 1 - t;
    const inverseSquared = inverse * inverse;
    const tSquared = t * t;

    const x = (inverseSquared * inverse * startX) +
        (3 * inverseSquared * t * control1X) +
        (3 * inverse * tSquared * control2X) +
        (tSquared * t * endX);

    const y = (inverseSquared * inverse * startY) +
        (3 * inverseSquared * t * control1Y) +
        (3 * inverse * tSquared * control2Y) +
        (tSquared * t * endY);

    return { x, y };
}

export const RenderMathMarkdown: React.FC<RenderMathMarkdownProps> = ({
    content,
    className = '',
    semanticWeights = [],
    deltaVisualization = null,
    sentenceArcConnections = [],
    selectedSentenceIndices = [],
    activeArcId = null,
    isSentenceSelectionEnabled = false,
    onSentenceClick,
    onDeltaSentenceClick,
    onArcClick,
}) => {
    const _isHighlightingReady = useHighlighting();
    const containerRef = useRef<HTMLDivElement>(null);
    const busyRef = useRef<Record<string, boolean>>({});
    const [sentenceLayouts, setSentenceLayouts] = useState<Record<number, { x: number; y: number; width: number; height: number }>>({});
    const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });

    const handleContainerClick = async (event: React.MouseEvent<HTMLDivElement>) => {
        const target = event.target as HTMLElement;

        const copyButton = target.closest('.copy-code-btn') as HTMLElement | null;
        if (copyButton) {
            const codeId = copyButton.getAttribute('data-code-id') ||
                copyButton.closest('.code-block-container')?.querySelector('.code-block-content code')?.id;

            if (codeId) {
                const busyKey = `copy:${codeId}`;
                if (busyRef.current[busyKey]) {
                    return;
                }

                busyRef.current[busyKey] = true;

                try {
                    const codeElement = document.getElementById(codeId) ||
                        copyButton.closest('.code-block-container')?.querySelector('.code-block-content code') as HTMLElement | null;

                    if (codeElement) {
                        await navigator.clipboard.writeText(codeElement.textContent || '');

                        const iconSlot = copyButton.querySelector('.icon-slot');
                        if (iconSlot) {
                            const originalIcon = (iconSlot as HTMLElement).dataset.iconName || 'content_copy';
                            setIconSlot(iconSlot, 'check');
                            setTimeout(() => {
                                setIconSlot(iconSlot, originalIcon);
                                delete busyRef.current[busyKey];
                            }, 1500);
                            return;
                        }
                    }
                } catch (error) {
                    console.error('Copy failed:', error);
                }

                setTimeout(() => {
                    delete busyRef.current[busyKey];
                }, 500);
            }

            return;
        }

        const imageItem = target.closest('.exec-image-item') as HTMLElement | null;
        if (imageItem && !imageItem.classList.contains('exec-image-error-item')) {
            const src = imageItem.dataset.src || (imageItem.querySelector('img') as HTMLImageElement | null)?.src;
            const format = imageItem.dataset.format || 'PNG';
            const alt = imageItem.dataset.alt || 'Generated Figure';

            if (src) {
                window.dispatchEvent(new CustomEvent('exec-image-preview', {
                    detail: { src, alt, format },
                }));
            }

            return;
        }

        const semanticSentence = target.closest('.semantic-sentence[data-semantic-index]') as HTMLElement | null;
        if (semanticSentence && onSentenceClick) {
            const sentenceIndex = Number(semanticSentence.dataset.semanticIndex);
            if (Number.isFinite(sentenceIndex)) {
                onSentenceClick(sentenceIndex);
                return;
            }
        }

        const deltaToken = target.closest('.delta-token[data-delta-index]') as HTMLElement | null;
        if (deltaToken && onDeltaSentenceClick) {
            const sentenceIndex = Number(deltaToken.dataset.deltaIndex);
            if (Number.isFinite(sentenceIndex)) {
                onDeltaSentenceClick(sentenceIndex);
                return;
            }
        }
    };

    const htmlContent = useMemo(() => {
        return renderMathContent(content, semanticWeights, deltaVisualization);
    }, [content, deltaVisualization, semanticWeights, _isHighlightingReady]);

    const activeArc = useMemo(() => (
        sentenceArcConnections.find((connection) => connection.id === activeArcId) ?? null
    ), [activeArcId, sentenceArcConnections]);

    const updateSentenceLayouts = React.useCallback(() => {
        const container = containerRef.current;
        if (!container) {
            setSentenceLayouts({});
            setCanvasSize({ width: 0, height: 0 });
            return;
        }

        const containerRect = container.getBoundingClientRect();
        const groupedRects = new Map<number, Array<{ left: number; top: number; right: number; bottom: number }>>();
        const connectedSentenceIndices = new Set<number>();
        sentenceArcConnections.forEach((connection) => {
            connectedSentenceIndices.add(connection.sourceIndex);
            connectedSentenceIndices.add(connection.targetIndex);
        });

        const selectedSentenceIndexSet = new Set(selectedSentenceIndices);
        const activeEndpoints = new Set<number>();
        if (activeArc) {
            activeEndpoints.add(activeArc.sourceIndex);
            activeEndpoints.add(activeArc.targetIndex);
        }

        const selector = deltaVisualization
            ? '.delta-token[data-delta-index]'
            : '.semantic-sentence[data-semantic-index]';
        const indexKey = deltaVisualization ? 'deltaIndex' : 'semanticIndex';

        container.querySelectorAll<HTMLElement>(selector).forEach((node) => {
            const sentenceIndex = Number(node.dataset[indexKey as 'deltaIndex' | 'semanticIndex']);
            if (!Number.isFinite(sentenceIndex)) {
                return;
            }

            const rect = node.getBoundingClientRect();
            if (rect.width <= 0 || rect.height <= 0) {
                return;
            }

            groupedRects.set(sentenceIndex, [
                ...(groupedRects.get(sentenceIndex) ?? []),
                {
                    left: rect.left - containerRect.left,
                    top: rect.top - containerRect.top,
                    right: rect.right - containerRect.left,
                    bottom: rect.bottom - containerRect.top,
                },
            ]);

            node.dataset.arcSelectable = isSentenceSelectionEnabled ? 'true' : 'false';
            node.dataset.arcSelected = selectedSentenceIndexSet.has(sentenceIndex) ? 'true' : 'false';
            node.dataset.arcConnected = connectedSentenceIndices.has(sentenceIndex) ? 'true' : 'false';

            if (activeEndpoints.has(sentenceIndex)) {
                node.dataset.arcEndpoint = activeArc?.sourceIndex === sentenceIndex ? 'source' : 'target';
            } else {
                delete node.dataset.arcEndpoint;
            }
        });

        const nextLayouts: Record<number, { x: number; y: number; width: number; height: number }> = {};
        groupedRects.forEach((rects, sentenceIndex) => {
            const left = Math.min(...rects.map((rect) => rect.left));
            const top = Math.min(...rects.map((rect) => rect.top));
            const right = Math.max(...rects.map((rect) => rect.right));
            const bottom = Math.max(...rects.map((rect) => rect.bottom));

            nextLayouts[sentenceIndex] = {
                x: left,
                y: top,
                width: right - left,
                height: bottom - top,
            };
        });

        setSentenceLayouts(nextLayouts);
        setCanvasSize({
            width: Math.max(container.scrollWidth, Math.ceil(containerRect.width)),
            height: Math.max(container.scrollHeight, Math.ceil(containerRect.height)),
        });
    }, [activeArc, deltaVisualization, isSentenceSelectionEnabled, selectedSentenceIndices, sentenceArcConnections]);

    useEffect(() => {
        const container = containerRef.current;
        if (!container) {
            return;
        }

        let frameId = window.requestAnimationFrame(updateSentenceLayouts);
        const handleResize = () => {
            window.cancelAnimationFrame(frameId);
            frameId = window.requestAnimationFrame(updateSentenceLayouts);
        };
        const resizeObserver = typeof ResizeObserver === 'undefined'
            ? null
            : new ResizeObserver(handleResize);

        resizeObserver?.observe(container);
        window.addEventListener('resize', updateSentenceLayouts);

        return () => {
            window.cancelAnimationFrame(frameId);
            resizeObserver?.disconnect();
            window.removeEventListener('resize', updateSentenceLayouts);
        };
    }, [htmlContent, updateSentenceLayouts]);

    const arcPaths = useMemo(() => {
        const sortedConnections = [...sentenceArcConnections]
            .sort((left, right) => (
                left.sourceIndex - right.sourceIndex ||
                right.score - left.score ||
                left.targetIndex - right.targetIndex
            ));
        const sourceGroups = new Map<number, AttentionArcConnection[]>();
        const targetGroups = new Map<number, AttentionArcConnection[]>();

        sortedConnections.forEach((connection) => {
            sourceGroups.set(connection.sourceIndex, [
                ...(sourceGroups.get(connection.sourceIndex) ?? []),
                connection,
            ]);
            targetGroups.set(connection.targetIndex, [
                ...(targetGroups.get(connection.targetIndex) ?? []),
                connection,
            ]);
        });

        return sortedConnections
            .map((connection, connectionIndex) => {
                const source = sentenceLayouts[connection.sourceIndex];
                const target = sentenceLayouts[connection.targetIndex];

                if (!source || !target) {
                    return null;
                }

                const sourceGroup = sourceGroups.get(connection.sourceIndex) ?? [connection];
                const targetGroup = targetGroups.get(connection.targetIndex) ?? [connection];
                const sourceGroupIndex = sourceGroup.findIndex((item) => item.id === connection.id);
                const targetGroupIndex = targetGroup.findIndex((item) => item.id === connection.id);

                const sourceAnchorSpan = Math.min(source.width * 0.58, 92);
                const targetAnchorSpan = Math.min(target.width * 0.58, 92);
                const sourceX = clampNumber(
                    source.x + (source.width * 0.5) + slotOffset(sourceGroupIndex, sourceGroup.length, sourceAnchorSpan),
                    source.x + 12,
                    source.x + source.width - 12,
                );
                const targetX = clampNumber(
                    target.x + (target.width * 0.5) + slotOffset(targetGroupIndex, targetGroup.length, targetAnchorSpan),
                    target.x + 12,
                    target.x + target.width - 12,
                );
                const sourceY = source.y + (source.height * 0.52);
                const targetY = target.y + (target.height * 0.48);
                const verticalGap = Math.max(28, Math.abs(sourceY - targetY));
                const lane = Math.floor(connectionIndex / 2);
                const side = connectionIndex % 2 === 0 ? -1 : 1;
                const bendStrength = Math.min(
                    canvasSize.width * 0.28,
                    96 + (lane * 42) + (verticalGap * 0.22),
                );
                const control1X = clampNumber(sourceX + (side * bendStrength), 20, Math.max(24, canvasSize.width - 20));
                const control2X = clampNumber(targetX + (side * bendStrength * 0.88), 20, Math.max(24, canvasSize.width - 20));
                const control1Y = sourceY - Math.max(22, verticalGap * 0.28);
                const control2Y = targetY + Math.max(22, verticalGap * 0.28);
                const midpoint = cubicPoint(
                    sourceX,
                    sourceY,
                    control1X,
                    control1Y,
                    control2X,
                    control2Y,
                    targetX,
                    targetY,
                    0.5,
                );

                return {
                    ...connection,
                    d: `M ${sourceX} ${sourceY} C ${control1X} ${control1Y}, ${control2X} ${control2Y}, ${targetX} ${targetY}`,
                    sourceX,
                    sourceY,
                    targetX,
                    targetY,
                    midpointX: midpoint.x,
                    midpointY: midpoint.y,
                };
            })
            .filter((connection): connection is AttentionArcConnection & {
                d: string;
                sourceX: number;
                sourceY: number;
                targetX: number;
                targetY: number;
                midpointX: number;
                midpointY: number;
            } => Boolean(connection))
            .sort((left, right) => left.score - right.score);
    }, [canvasSize.width, sentenceArcConnections, sentenceLayouts]);

    return (
        <div className="render-math-markdown-shell">
            <div
                ref={containerRef}
                className={`render-math-markdown ${className}`.trim()}
                onClick={handleContainerClick}
                dangerouslySetInnerHTML={{ __html: htmlContent }}
            />

            {canvasSize.width > 0 && canvasSize.height > 0 && arcPaths.length ? (
                <svg
                    className="render-math-markdown-arcs"
                    viewBox={`0 0 ${canvasSize.width} ${canvasSize.height}`}
                    width={canvasSize.width}
                    height={canvasSize.height}
                    aria-label="Sentence attention arc overlay"
                >
                    <defs>
                        {arcPaths.map((connection) => (
                            <marker
                                key={`marker-${connection.id}`}
                                id={`sentence-arc-arrow-${connection.id}`}
                                viewBox="0 0 10 10"
                                refX="8.4"
                                refY="5"
                                markerWidth="8"
                                markerHeight="8"
                                orient="auto-start-reverse"
                            >
                                <path d="M 0 0 L 10 5 L 0 10 z" fill={connection.color} opacity="0.96" />
                            </marker>
                        ))}
                    </defs>
                    {arcPaths.map((connection) => {
                        const isActive = connection.id === activeArcId;
                        const labelWidth = Math.max(118, (connection.label.length * 7) + 18);

                        return (
                            <g
                                key={connection.id}
                                className={`sentence-arc ${isActive ? 'sentence-arc-active' : ''}`}
                                role="button"
                                tabIndex={0}
                                aria-label={connection.label}
                                onClick={() => onArcClick?.(connection.id)}
                                onKeyDown={(event) => {
                                    if (event.key === 'Enter' || event.key === ' ') {
                                        event.preventDefault();
                                        onArcClick?.(connection.id);
                                    }
                                }}
                            >
                                <path className="sentence-arc-hit" d={connection.d} />
                                <path
                                    className="sentence-arc-aura"
                                    d={connection.d}
                                    style={{
                                        stroke: connection.color,
                                        opacity: isActive ? 0.26 : 0.1 + (connection.score * 0.12),
                                        strokeWidth: isActive ? 13 : 8 + (connection.score * 4),
                                    }}
                                />
                                <path
                                    className="sentence-arc-line"
                                    d={connection.d}
                                    style={{
                                        stroke: connection.color,
                                        opacity: isActive ? 0.98 : 0.28 + (connection.score * 0.44),
                                        strokeWidth: isActive ? 4.4 : 1.8 + (connection.score * 2.2),
                                    }}
                                    markerEnd={`url(#sentence-arc-arrow-${connection.id})`}
                                />
                                <circle
                                    className="sentence-arc-endpoint"
                                    cx={connection.sourceX}
                                    cy={connection.sourceY}
                                    r={3.4}
                                    style={{ fill: connection.color }}
                                />
                                <circle
                                    className="sentence-arc-endpoint"
                                    cx={connection.targetX}
                                    cy={connection.targetY}
                                    r={3.4}
                                    style={{ fill: connection.color }}
                                />
                                {isActive ? (
                                    <>
                                        <circle
                                            cx={connection.targetX}
                                            cy={connection.targetY}
                                            r={6.6}
                                            className="sentence-arc-endpoint-ring"
                                            style={{ stroke: connection.color }}
                                        />
                                        <circle
                                            cx={connection.midpointX}
                                            cy={connection.midpointY}
                                            r={4.5}
                                            className="sentence-arc-focus-dot"
                                            style={{ fill: connection.color }}
                                        />
                                        <g className="sentence-arc-label" transform={`translate(${connection.midpointX}, ${connection.midpointY - 18})`}>
                                            <rect x={-labelWidth / 2} y={-14} width={labelWidth} height={28} rx={14} />
                                            <text textAnchor="middle" dominantBaseline="middle">{connection.label}</text>
                                        </g>
                                    </>
                                ) : null}
                            </g>
                        );
                    })}
                </svg>
            ) : null}
        </div>
    );
};

export default RenderMathMarkdown;
