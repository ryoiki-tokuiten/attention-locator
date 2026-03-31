import React, { useEffect, useMemo, useRef, useState } from 'react';

import type { DeltaVisualizationPayload } from './types/semantic';

interface ViewBox {
    x: number;
    y: number;
    width: number;
    height: number;
}

interface ViewportSize {
    width: number;
    height: number;
}

interface DeltaVectorFlowCanvasProps {
    visualization: DeltaVisualizationPayload | null;
    selectedSentenceIndex: number | null;
    onSelectSentence?: (sentenceIndex: number) => void;
}

const MIN_VIEWBOX_SIZE = 6;
const ZOOM_IN_FACTOR = 0.88;
const ZOOM_OUT_FACTOR = 1.12;
const DRAG_THRESHOLD_PX = 6;

function createViewBox(
    bounds: DeltaVisualizationPayload['vectorFlow']['bounds'],
    viewport: ViewportSize,
): ViewBox {
    const baseWidth = Math.max(MIN_VIEWBOX_SIZE, bounds.maxX - bounds.minX);
    const baseHeight = Math.max(MIN_VIEWBOX_SIZE, bounds.maxY - bounds.minY);
    const paddingX = baseWidth * 0.06;
    const paddingY = baseHeight * 0.06;
    let width = baseWidth + (paddingX * 2);
    let height = baseHeight + (paddingY * 2);
    const viewportRatio = viewport.width > 0 && viewport.height > 0
        ? viewport.width / viewport.height
        : 1;
    const currentRatio = width / height;

    if (viewportRatio > currentRatio) {
        width = height * viewportRatio;
    } else {
        height = width / viewportRatio;
    }

    const centerX = (bounds.minX + bounds.maxX) / 2;
    const centerY = (bounds.minY + bounds.maxY) / 2;
    return {
        x: centerX - (width / 2),
        y: centerY - (height / 2),
        width,
        height,
    };
}

function clampViewBox(next: ViewBox): ViewBox {
    return {
        ...next,
        width: Math.max(MIN_VIEWBOX_SIZE, next.width),
        height: Math.max(MIN_VIEWBOX_SIZE, next.height),
    };
}

function chooseGridStep(span: number): number {
    const safeSpan = Math.max(span, MIN_VIEWBOX_SIZE);
    const target = safeSpan / 10;
    const magnitude = Math.pow(10, Math.floor(Math.log10(target)));
    const normalized = target / magnitude;

    if (normalized <= 1) {
        return magnitude;
    }
    if (normalized <= 2) {
        return 2 * magnitude;
    }
    if (normalized <= 5) {
        return 5 * magnitude;
    }
    return 10 * magnitude;
}

function zoomViewBox(
    current: ViewBox,
    rect: DOMRect,
    clientX: number,
    clientY: number,
    deltaY: number,
): ViewBox {
    const scaleFactor = deltaY < 0 ? ZOOM_IN_FACTOR : ZOOM_OUT_FACTOR;
    const pointerX = (clientX - rect.left) / rect.width;
    const pointerY = (clientY - rect.top) / rect.height;
    const anchorX = current.x + (pointerX * current.width);
    const anchorY = current.y + (pointerY * current.height);
    const width = current.width * scaleFactor;
    const height = current.height * scaleFactor;

    return clampViewBox({
        x: anchorX - (pointerX * width),
        y: anchorY - (pointerY * height),
        width,
        height,
    });
}

function extractSentenceIndex(target: EventTarget | null): number | null {
    if (!(target instanceof Element)) {
        return null;
    }
    const sentenceNode = target.closest('[data-sentence-index]');
    if (!(sentenceNode instanceof Element)) {
        return null;
    }
    const sentenceIndex = Number(sentenceNode.getAttribute('data-sentence-index'));
    return Number.isFinite(sentenceIndex) ? sentenceIndex : null;
}

const DeltaVectorFlowCanvas: React.FC<DeltaVectorFlowCanvasProps> = ({
    visualization,
    selectedSentenceIndex,
    onSelectSentence,
}) => {
    const svgRef = useRef<SVGSVGElement>(null);
    const wrapperRef = useRef<HTMLDivElement>(null);
    const dragStateRef = useRef<{
        x: number;
        y: number;
        pointerId: number;
        dragged: boolean;
        sentenceIndex: number | null;
    } | null>(null);
    const initializedViewKeyRef = useRef<string | null>(null);
    const [viewport, setViewport] = useState<ViewportSize>({ width: 1, height: 1 });

    const flowSentences = useMemo(() => (
        visualization?.sentences.filter((sentence) => sentence.vectorFlow) ?? []
    ), [visualization]);

    const initialViewBox = useMemo<ViewBox>(() => (
        createViewBox(
            visualization?.vectorFlow.bounds ?? { minX: -1.5, maxX: 1.5, minY: -1.5, maxY: 1.5 },
            viewport,
        )
    ), [viewport, visualization]);

    const viewResetKey = useMemo(() => {
        const bounds = visualization?.vectorFlow.bounds ?? { minX: -1.5, maxX: 1.5, minY: -1.5, maxY: 1.5 };
        return [
            visualization?.analysisText ?? 'empty',
            bounds.minX,
            bounds.maxX,
            bounds.minY,
            bounds.maxY,
        ].join(':');
    }, [visualization]);

    const [viewBox, setViewBox] = useState<ViewBox>(initialViewBox);

    useEffect(() => {
        if (viewport.width <= 1 || viewport.height <= 1) {
            return;
        }
        if (initializedViewKeyRef.current === viewResetKey) {
            return;
        }
        initializedViewKeyRef.current = viewResetKey;
        setViewBox(initialViewBox);
    }, [initialViewBox, viewResetKey, viewport.height, viewport.width]);

    useEffect(() => {
        const wrapper = wrapperRef.current;
        if (!wrapper) {
            return;
        }

        const measure = () => {
            const rect = wrapper.getBoundingClientRect();
            setViewport({
                width: Math.max(rect.width, 1),
                height: Math.max(rect.height, 1),
            });
        };

        measure();

        if (typeof ResizeObserver === 'undefined') {
            window.addEventListener('resize', measure);
            return () => window.removeEventListener('resize', measure);
        }

        const observer = new ResizeObserver(measure);
        observer.observe(wrapper);
        return () => observer.disconnect();
    }, []);

    const gridLines = useMemo(() => {
        const lines: Array<{ x1: number; y1: number; x2: number; y2: number; key: string; className: string }> = [];
        const majorStep = chooseGridStep(Math.max(viewBox.width, viewBox.height));
        const minorStep = majorStep / 4;
        const minX = Math.floor(viewBox.x / minorStep) * minorStep;
        const maxX = Math.ceil((viewBox.x + viewBox.width) / minorStep) * minorStep;
        const minY = Math.floor(viewBox.y / minorStep) * minorStep;
        const maxY = Math.ceil((viewBox.y + viewBox.height) / minorStep) * minorStep;
        const epsilon = minorStep / 100;

        for (let x = minX; x <= maxX + epsilon; x += minorStep) {
            const snapped = Number(x.toFixed(6));
            const isAxis = Math.abs(snapped) <= epsilon;
            const isMajor = Math.abs((snapped / majorStep) - Math.round(snapped / majorStep)) <= 1e-4;
            lines.push({
                x1: snapped,
                y1: minY,
                x2: snapped,
                y2: maxY,
                key: `vx-${snapped}`,
                className: isAxis ? 'delta-flow-axis' : isMajor ? 'delta-flow-grid-line-major' : 'delta-flow-grid-line',
            });
        }

        for (let y = minY; y <= maxY + epsilon; y += minorStep) {
            const snapped = Number(y.toFixed(6));
            const isAxis = Math.abs(snapped) <= epsilon;
            const isMajor = Math.abs((snapped / majorStep) - Math.round(snapped / majorStep)) <= 1e-4;
            lines.push({
                x1: minX,
                y1: snapped,
                x2: maxX,
                y2: snapped,
                key: `hy-${snapped}`,
                className: isAxis ? 'delta-flow-axis' : isMajor ? 'delta-flow-grid-line-major' : 'delta-flow-grid-line',
            });
        }

        return lines;
    }, [viewBox]);

    useEffect(() => {
        const wrapper = wrapperRef.current;
        if (!wrapper) {
            return;
        }

        const handleWheel = (event: WheelEvent) => {
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            const rect = wrapper.getBoundingClientRect();

            setViewBox((current) => {
                return zoomViewBox(current, rect, event.clientX, event.clientY, event.deltaY);
            });
        };

        wrapper.addEventListener('wheel', handleWheel, { passive: false, capture: true });
        return () => wrapper.removeEventListener('wheel', handleWheel, true);
    }, []);

    const handlePointerDown = (event: React.PointerEvent<SVGSVGElement>) => {
        dragStateRef.current = {
            x: event.clientX,
            y: event.clientY,
            pointerId: event.pointerId,
            dragged: false,
            sentenceIndex: extractSentenceIndex(event.target),
        };
        event.currentTarget.setPointerCapture(event.pointerId);
    };

    const handlePointerMove = (event: React.PointerEvent<SVGSVGElement>) => {
        const dragState = dragStateRef.current;
        const svg = svgRef.current;
        if (!dragState || dragState.pointerId !== event.pointerId || !svg) {
            return;
        }

        const rect = svg.getBoundingClientRect();
        const deltaClientX = event.clientX - dragState.x;
        const deltaClientY = event.clientY - dragState.y;

        if (!dragState.dragged && Math.hypot(deltaClientX, deltaClientY) < DRAG_THRESHOLD_PX) {
            return;
        }

        if (!dragState.dragged) {
            dragStateRef.current = {
                ...dragState,
                x: event.clientX,
                y: event.clientY,
                dragged: true,
            };
            return;
        }

        dragStateRef.current = {
            ...dragState,
            x: event.clientX,
            y: event.clientY,
            dragged: true,
        };
        setViewBox((current) => {
            const deltaX = (deltaClientX / rect.width) * current.width;
            const deltaY = (deltaClientY / rect.height) * current.height;
            return {
                ...current,
                x: current.x - deltaX,
                y: current.y - deltaY,
            };
        });
    };

    const handlePointerUp = (event: React.PointerEvent<SVGSVGElement>) => {
        const dragState = dragStateRef.current;
        if (dragState?.pointerId === event.pointerId) {
            if (!dragState.dragged && dragState.sentenceIndex != null) {
                onSelectSentence?.(dragState.sentenceIndex);
            }
            dragStateRef.current = null;
            event.currentTarget.releasePointerCapture(event.pointerId);
        }
    };

    if (!visualization || !flowSentences.length) {
        return (
            <div className="delta-flow-empty">
                <strong>Vector flow will appear here.</strong>
                <span>The backend needs at least one sentence with a valid projected delta path.</span>
            </div>
        );
    }

    return (
        <section className="delta-flow-shell" aria-label="Sentence vector flow">
            <div className="delta-flow-header">
                <div>
                    <p className="panel-label">Projected sentence chain</p>
                    <h3>Embedding and delta flow on a 2D canvas</h3>
                </div>
                <button
                    type="button"
                    className="arc-clear-button"
                    onClick={() => setViewBox(initialViewBox)}
                >
                    Reset view
                </button>
            </div>

            <div className="delta-flow-legend">
                <span><i className="delta-flow-legend-base" /> Sentence vector</span>
                <span><i className="delta-flow-legend-delta" /> Delta vector</span>
                <span><i className="delta-flow-legend-composed" /> Composed update</span>
                <small>Drag to pan. Scroll to zoom.</small>
            </div>

            <div ref={wrapperRef} className="delta-flow-canvas-wrap">
                <svg
                    ref={svgRef}
                    className="delta-flow-canvas"
                    viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`}
                    preserveAspectRatio="xMidYMid slice"
                    onPointerDown={handlePointerDown}
                    onPointerMove={handlePointerMove}
                    onPointerUp={handlePointerUp}
                    onPointerLeave={handlePointerUp}
                >
                    <defs>
                        <marker id="delta-flow-arrow-base" markerWidth="0.34" markerHeight="0.34" refX="0.28" refY="0.17" orient="auto" markerUnits="userSpaceOnUse">
                            <path d="M0,0 L0.34,0.17 L0,0.34 z" fill="rgba(96, 165, 250, 0.95)" />
                        </marker>
                        <marker id="delta-flow-arrow-delta" markerWidth="0.30" markerHeight="0.30" refX="0.25" refY="0.15" orient="auto" markerUnits="userSpaceOnUse">
                            <path d="M0,0 L0.30,0.15 L0,0.30 z" fill="rgba(248, 113, 113, 0.95)" />
                        </marker>
                        <marker id="delta-flow-arrow-composed" markerWidth="0.28" markerHeight="0.28" refX="0.22" refY="0.14" orient="auto" markerUnits="userSpaceOnUse">
                            <path d="M0,0 L0.28,0.14 L0,0.28 z" fill="rgba(52, 211, 153, 0.95)" />
                        </marker>
                    </defs>

                    <g className="delta-flow-grid">
                        {gridLines.map((line) => (
                            <line
                                key={line.key}
                                className={line.className}
                                x1={line.x1}
                                y1={line.y1}
                                x2={line.x2}
                                y2={line.y2}
                            />
                        ))}
                    </g>

                    <g className="delta-flow-layer">
                        {flowSentences.map((sentence) => {
                            const flow = sentence.vectorFlow;
                            if (!flow) {
                                return null;
                            }

                            const isSelected = selectedSentenceIndex === sentence.index;
                            return (
                                <g
                                    key={`flow-${sentence.index}`}
                                    data-sentence-index={sentence.index}
                                    className={`delta-flow-sentence ${isSelected ? 'delta-flow-sentence-selected' : ''}`}
                                >
                                    <title>
                                        {`${sentence.displaySentence}\nSentence angle ${flow.sentenceAngle.toFixed(1)}°\nDelta angle ${flow.deltaAngle.toFixed(1)}°\nRelative delta ${flow.deltaRelativeAngle.toFixed(1)}°`}
                                    </title>
                                    <circle
                                        className="delta-flow-unit-circle"
                                        cx={flow.anchorX}
                                        cy={flow.anchorY}
                                        r={visualization.vectorFlow.unitCircleRadius}
                                    />
                                    <line
                                        className="delta-flow-composed-line"
                                        x1={flow.anchorX}
                                        y1={flow.anchorY}
                                        x2={flow.updatedEndX}
                                        y2={flow.updatedEndY}
                                        markerEnd="url(#delta-flow-arrow-composed)"
                                    />
                                    <line
                                        className="delta-flow-base-line"
                                        x1={flow.anchorX}
                                        y1={flow.anchorY}
                                        x2={flow.sentenceEndX}
                                        y2={flow.sentenceEndY}
                                        markerEnd="url(#delta-flow-arrow-base)"
                                    />
                                    <line
                                        className="delta-flow-delta-line"
                                        x1={flow.sentenceEndX}
                                        y1={flow.sentenceEndY}
                                        x2={flow.updatedEndX}
                                        y2={flow.updatedEndY}
                                        markerEnd="url(#delta-flow-arrow-delta)"
                                    />
                                    <circle className="delta-flow-anchor-dot" cx={flow.anchorX} cy={flow.anchorY} r={0.08} />
                                    <circle className="delta-flow-end-dot" cx={flow.updatedEndX} cy={flow.updatedEndY} r={0.1} />
                                    <circle className="delta-flow-hit" cx={flow.updatedEndX} cy={flow.updatedEndY} r={0.28} />
                                    <text className="delta-flow-label" x={flow.updatedEndX + 0.14} y={flow.updatedEndY - 0.14}>
                                        S{sentence.index}
                                    </text>
                                </g>
                            );
                        })}
                    </g>
                </svg>
            </div>
        </section>
    );
};

export default DeltaVectorFlowCanvas;
