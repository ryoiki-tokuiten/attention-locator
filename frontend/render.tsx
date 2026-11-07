import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';

import type { DeltaSentenceVisualization, DeltaVisualizationMode, SemanticWeight } from './src/types/semantic';
import { asFiniteNumber, attentionRgbForWeight, clamp } from './src/utils';

export interface AttentionArcConnection {
    id: string;
    sourceIndex: number;
    targetIndex: number;
    score: number;
    color: string;
    label: string;
}

export interface RenderProps {
    content: string;
    onChange?: (nextContent: string) => void;
    placeholder?: string;
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

interface ColorStop {
    at: number;
    color: [number, number, number];
}

function escapeHtml(value: string): string {
    return value
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function interpolateColorStops(weight: number, stops: ColorStop[]): [number, number, number] {
    if (weight <= stops[0].at) {
        return stops[0].color;
    }

    for (let index = 1; index < stops.length; index += 1) {
        const previous = stops[index - 1];
        const current = stops[index];

        if (weight <= current.at) {
            const range = current.at - previous.at || 1;
            const progress = (weight - previous.at) / range;

            return previous.color.map((channel, channelIndex) => (
                Math.round(channel + ((current.color[channelIndex] - channel) * progress))
            )) as [number, number, number];
        }
    }

    return stops[stops.length - 1].color;
}

function rgbVar(color: readonly [number, number, number]): string {
    return color.join(', ');
}

function styleForSemanticWeight(weight: number): string {
    const normalizedWeight = clamp(asFiniteNumber(weight, 0.5));
    const primaryColor = attentionRgbForWeight(normalizedWeight);
    const secondaryColor = interpolateColorStops(normalizedWeight, [
        { at: 0, color: [22, 163, 74] },
        { at: 0.5, color: [37, 99, 235] },
        { at: 1, color: [220, 38, 38] },
    ]);
    const textColor = interpolateColorStops(normalizedWeight, [
        { at: 0, color: [112, 238, 156] },
        { at: 0.5, color: [133, 198, 255] },
        { at: 1, color: [255, 148, 148] },
    ]);
    const fillOpacity = (0.19 + (normalizedWeight * 0.28)).toFixed(3);
    const glowOpacity = (0.10 + (normalizedWeight * 0.16)).toFixed(3);
    const edgeOpacity = (0.18 + (normalizedWeight * 0.22)).toFixed(3);

    return [
        `--semantic-primary-rgb: ${rgbVar(primaryColor)}`,
        `--semantic-secondary-rgb: ${rgbVar(secondaryColor)}`,
        `--semantic-text-rgb: ${rgbVar(textColor)}`,
        `--semantic-fill-opacity: ${fillOpacity}`,
        `--semantic-glow-opacity: ${glowOpacity}`,
        `--semantic-edge-opacity: ${edgeOpacity}`,
    ].join('; ');
}

function semanticScoreLabel(semantic: SemanticWeight): string {
    const weight = clamp(asFiniteNumber(semantic.weight, 0.5));
    return `Attention score ${weight.toFixed(3)} · ${semantic.band}`;
}

function sourceDrivenColor(positionRatio: number): [number, number, number] {
    return interpolateColorStops(clamp(asFiniteNumber(positionRatio, 0.5)), [
        { at: 0, color: [40, 209, 122] },
        { at: 0.5, color: [69, 138, 255] },
        { at: 1, color: [255, 110, 110] },
    ]);
}

function deltaTooltip(sentence: DeltaSentenceVisualization, mode: DeltaVisualizationMode): string {
    if (mode === 'drift') {
        return `Drift ${sentence.driftNorm.toFixed(3)} · peak layer ${sentence.peakLayer + 1}`;
    }
    if (mode === 'local-global') {
        return `Flow ${sentence.incomingFlowNorm.toFixed(3)} · local ${sentence.localShare.toFixed(2)} · medium ${sentence.mediumShare.toFixed(2)} · global ${sentence.globalShare.toFixed(2)}`;
    }
    return `Flow ${sentence.incomingFlowNorm.toFixed(3)} · ${sentence.topSources.length} source sentences`;
}

function styleForDeltaSentence(
    sentence: DeltaSentenceVisualization,
    mode: DeltaVisualizationMode,
    sentenceCount: number,
): string {
    const intensity = mode === 'drift' ? sentence.driftNorm : sentence.incomingFlowNorm;
    const [baseRed, baseGreen, baseBlue] = attentionRgbForWeight(asFiniteNumber(intensity, 0.5));
    const localColor: [number, number, number] = [40, 209, 122];
    const mediumColor: [number, number, number] = [69, 138, 255];
    const globalColor: [number, number, number] = [255, 110, 110];
    const primaryComposition = sentence.topSources[0];
    const secondaryComposition = sentence.topSources[1] ?? sentence.topSources[0];
    const compositionColorA = sourceDrivenColor((primaryComposition?.sourceIndex ?? sentence.index) / Math.max(1, sentenceCount - 1));
    const compositionColorB = sourceDrivenColor((secondaryComposition?.sourceIndex ?? sentence.index) / Math.max(1, sentenceCount - 1));
    const driftColor = interpolateColorStops(sentence.peakLayerRatio, [
        { at: 0, color: [52, 211, 153] },
        { at: 0.5, color: [96, 165, 250] },
        { at: 1, color: [248, 113, 113] },
    ]);

    const glowOpacity = (0.12 + (intensity * 0.24)).toFixed(3);
    const textOpacity = (0.74 + (intensity * 0.20)).toFixed(3);
    const baseDeclarations = [
        `--delta-weight: ${intensity.toFixed(3)}`,
        `--delta-primary-rgb: ${rgbVar([baseRed, baseGreen, baseBlue])}`,
        `--delta-glow-opacity: ${glowOpacity}`,
        `--delta-text-opacity: ${textOpacity}`,
    ];

    if (mode === 'local-global') {
        return [
            ...baseDeclarations,
            `background: linear-gradient(135deg, rgba(${rgbVar(localColor)}, ${0.10 + (sentence.localShare * 0.46)}) 0%, rgba(${rgbVar(mediumColor)}, ${0.10 + (sentence.mediumShare * 0.48)}) 56%, rgba(${rgbVar(globalColor)}, ${0.10 + (sentence.globalShare * 0.52)}) 100%)`,
        ].join('; ');
    }

    if (mode === 'composition') {
        return [
            ...baseDeclarations,
            `background: linear-gradient(135deg, rgba(${rgbVar(compositionColorA)}, ${0.16 + ((primaryComposition?.share ?? 0.5) * 0.52)}) 0%, rgba(${rgbVar(compositionColorB)}, ${0.14 + ((secondaryComposition?.share ?? 0.3) * 0.50)}) 100%)`,
        ].join('; ');
    }

    if (mode === 'drift') {
        return [
            ...baseDeclarations,
            `background: linear-gradient(135deg, rgba(${rgbVar(driftColor)}, ${0.14 + (intensity * 0.46)}) 0%, rgba(${rgbVar([baseRed, baseGreen, baseBlue])}, ${0.10 + (intensity * 0.38)}) 100%)`,
        ].join('; ');
    }

    return [
        ...baseDeclarations,
        `background: linear-gradient(135deg, rgba(${rgbVar([baseRed, baseGreen, baseBlue])}, ${0.14 + (intensity * 0.50)}) 0%, rgba(${rgbVar(globalColor)}, ${0.08 + ((primaryComposition?.scoreNorm ?? intensity) * 0.36)}) 100%)`,
    ].join('; ');
}

type RenderSegment =
    | { type: 'plain'; text: string }
    | { type: 'semantic'; text: string; index: number; band: string; label: string; style: string }
    | { type: 'delta'; text: string; index: number; tooltip: string; style: string };

function buildRenderSegments(
    content: string,
    semanticWeights: SemanticWeight[] = [],
    deltaVisualization: {
        mode: DeltaVisualizationMode;
        sentences: DeltaSentenceVisualization[];
    } | null = null,
): RenderSegment[] {
    if (!content) {
        return [];
    }

    if (deltaVisualization) {
        const ranges = deltaVisualization.sentences
            .filter((sentence) => (
                sentence.visible &&
                sentence.sourceStart != null &&
                sentence.sourceEnd != null &&
                sentence.sourceEnd > sentence.sourceStart &&
                sentence.sourceStart < content.length &&
                sentence.sourceEnd > 0
            ))
            .map((sentence) => ({
                start: Math.max(0, sentence.sourceStart as number),
                end: Math.min(content.length, sentence.sourceEnd as number),
                sentence,
            }))
            .sort((left, right) => left.start - right.start);

        if (!ranges.length) {
            return [{ type: 'plain', text: content }];
        }

        const segments: RenderSegment[] = [];
        let cursor = 0;

        for (const range of ranges) {
            const start = Math.max(cursor, range.start);
            const end = Math.max(start, range.end);

            if (start > cursor) {
                segments.push({ type: 'plain', text: content.slice(cursor, start) });
            }

            if (end > start) {
                segments.push({
                    type: 'delta',
                    text: content.slice(start, end),
                    index: range.sentence.index,
                    tooltip: deltaTooltip(range.sentence, deltaVisualization.mode),
                    style: styleForDeltaSentence(range.sentence, deltaVisualization.mode, deltaVisualization.sentences.length),
                });
            }

            cursor = end;
        }

        if (cursor < content.length) {
            segments.push({ type: 'plain', text: content.slice(cursor) });
        }

        return segments;
    }

    const validRanges = semanticWeights
        .filter((semantic) => (
            semantic.sourceStart != null &&
            semantic.sourceEnd != null &&
            semantic.sourceEnd > semantic.sourceStart &&
            semantic.sourceStart < content.length &&
            semantic.sourceEnd > 0
        ))
        .map((semantic) => ({
            start: Math.max(0, semantic.sourceStart as number),
            end: Math.min(content.length, semantic.sourceEnd as number),
            semantic: {
                ...semantic,
                weight: clamp(asFiniteNumber(semantic.weight, 0.5)),
            },
        }))
        .sort((left, right) => left.start - right.start);

    if (!validRanges.length) {
        return [{ type: 'plain', text: content }];
    }

    const segments: RenderSegment[] = [];
    let cursor = 0;

    for (const range of validRanges) {
        const start = Math.max(cursor, range.start);
        const end = Math.max(start, range.end);

        if (start > cursor) {
            segments.push({ type: 'plain', text: content.slice(cursor, start) });
        }

        if (end > start) {
            segments.push({
                type: 'semantic',
                text: content.slice(start, end),
                index: range.semantic.index,
                band: range.semantic.band,
                label: semanticScoreLabel(range.semantic),
                style: styleForSemanticWeight(range.semantic.weight),
            });
        }

        cursor = end;
    }

    if (cursor < content.length) {
        segments.push({ type: 'plain', text: content.slice(cursor) });
    }

    return segments;
}

export function renderPlainContent(
    content: string,
    semanticWeights: SemanticWeight[] = [],
    deltaVisualization: {
        mode: DeltaVisualizationMode;
        sentences: DeltaSentenceVisualization[];
    } | null = null,
): string {
    const segments = buildRenderSegments(content, semanticWeights, deltaVisualization);
    return segments
        .map((segment) => {
            if (segment.type === 'plain') {
                return escapeHtml(segment.text);
            }
            if (segment.type === 'delta') {
                return `<span class="delta-token" data-delta-index="${segment.index}" title="${escapeHtml(segment.tooltip)}" style="${escapeHtml(segment.style)}">${escapeHtml(segment.text)}</span>`;
            }
            return `<span class="semantic-sentence" data-semantic-band="${escapeHtml(segment.band)}" data-semantic-index="${segment.index}" data-semantic-score-label="${escapeHtml(segment.label)}" title="${escapeHtml(segment.label)}" style="${escapeHtml(segment.style)}">${escapeHtml(segment.text)}</span>`;
        })
        .join('');
}

function tryUpdateSegmentsInPlace(container: HTMLElement, segments: RenderSegment[]): boolean {
    if (container.childNodes.length !== segments.length) {
        return false;
    }

    for (let index = 0; index < segments.length; index += 1) {
        const segment = segments[index];
        const node = container.childNodes[index];

        if ((node.textContent ?? '') !== segment.text) {
            return false;
        }

        if (segment.type === 'plain') {
            if (node.nodeType !== Node.TEXT_NODE) {
                return false;
            }
            continue;
        }

        if (node.nodeType !== Node.ELEMENT_NODE) {
            return false;
        }

        const element = node as HTMLElement;
        const expectedClass = segment.type === 'delta' ? 'delta-token' : 'semantic-sentence';
        if (!element.classList.contains(expectedClass)) {
            return false;
        }
    }

    for (let index = 0; index < segments.length; index += 1) {
        const segment = segments[index];
        if (segment.type === 'plain') {
            continue;
        }

        const element = container.childNodes[index] as HTMLElement;
        if (segment.type === 'delta') {
            element.dataset.deltaIndex = String(segment.index);
            element.title = segment.tooltip;
            element.setAttribute('style', segment.style);
        } else {
            element.dataset.semanticBand = segment.band;
            element.dataset.semanticIndex = String(segment.index);
            element.dataset.semanticScoreLabel = segment.label;
            element.title = segment.label;
            element.setAttribute('style', segment.style);
        }
    }

    return true;
}

function readSelectionOffsets(container: HTMLElement): { start: number; end: number } | null {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) {
        return null;
    }

    const range = selection.getRangeAt(0);
    if (!container.contains(range.startContainer) || !container.contains(range.endContainer)) {
        return null;
    }

    const resolveOffset = (targetNode: Node, targetOffset: number): number => {
        if (targetNode === container) {
            let total = 0;
            for (let i = 0; i < targetOffset && i < container.childNodes.length; i += 1) {
                total += container.childNodes[i].textContent?.length ?? 0;
            }
            return total;
        }

        const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
        let offset = 0;
        let current = walker.nextNode();
        while (current) {
            if (current === targetNode) {
                return offset + targetOffset;
            }
            offset += current.textContent?.length ?? 0;
            current = walker.nextNode();
        }
        return offset;
    };

    return {
        start: resolveOffset(range.startContainer, range.startOffset),
        end: resolveOffset(range.endContainer, range.endOffset),
    };
}

function restoreSelectionOffsets(container: HTMLElement, offsets: { start: number; end: number }) {
    const selection = window.getSelection();
    if (!selection) {
        return;
    }

    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    let current = walker.nextNode();
    let accumulated = 0;
    let startNode: Node = container;
    let startOffset = 0;
    let endNode: Node = container;
    let endOffset = 0;
    let foundStart = false;
    let foundEnd = false;

    while (current) {
        const length = current.textContent?.length ?? 0;
        if (!foundStart && offsets.start <= accumulated + length) {
            startNode = current;
            startOffset = Math.max(0, offsets.start - accumulated);
            foundStart = true;
        }
        if (!foundEnd && offsets.end <= accumulated + length) {
            endNode = current;
            endOffset = Math.max(0, offsets.end - accumulated);
            foundEnd = true;
            break;
        }
        accumulated += length;
        current = walker.nextNode();
    }

    const range = document.createRange();
    range.setStart(startNode, startOffset);
    range.setEnd(endNode, endOffset);
    selection.removeAllRanges();
    selection.addRange(range);
}

function extractPlainText(container: HTMLElement): string {
    const pieces: string[] = [];
    const visit = (node: Node) => {
        if (node.nodeType === Node.TEXT_NODE) {
            pieces.push(node.textContent ?? '');
            return;
        }
        if (node.nodeName === 'BR') {
            pieces.push('\n');
            return;
        }
        const isBlock = node !== container && (node.nodeName === 'DIV' || node.nodeName === 'P');
        if (isBlock && pieces.length > 0 && !pieces[pieces.length - 1].endsWith('\n')) {
            pieces.push('\n');
        }
        node.childNodes.forEach(visit);
    };
    visit(container);
    return pieces.join('').replace(/\r\n?/g, '\n');
}

function insertPlainTextAtSelection(text: string) {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) {
        return;
    }
    const range = selection.getRangeAt(0);
    range.deleteContents();
    const textNode = document.createTextNode(text);
    range.insertNode(textNode);
    range.setStartAfter(textNode);
    range.setEndAfter(textNode);
    selection.removeAllRanges();
    selection.addRange(range);
}

export const Render: React.FC<RenderProps> = ({
    content,
    onChange,
    placeholder = 'Type or paste plain text here...',
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
    const containerRef = useRef<HTMLDivElement>(null);
    const lastUserTextRef = useRef<string | null>(null);
    const lastAnnotationSignatureRef = useRef<string>('');
    const [sentenceLayouts, setSentenceLayouts] = useState<Record<number, { x: number; y: number; width: number; height: number }>>({});
    const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });

    const htmlContent = useMemo(
        () => renderPlainContent(content, semanticWeights, deltaVisualization),
        [content, deltaVisualization, semanticWeights],
    );

    const annotationSignature = useMemo(() => {
        if (deltaVisualization) {
            return `delta:${deltaVisualization.mode}:${deltaVisualization.sentences
                .map((s) => `${s.index}:${s.sourceStart}-${s.sourceEnd}:${s.incomingFlowNorm}:${s.driftNorm}`)
                .join('|')}`;
        }
        return `semantic:${semanticWeights
            .map((w) => `${w.index}:${w.sourceStart}-${w.sourceEnd}:${w.weight}:${w.band}`)
            .join('|')}`;
    }, [deltaVisualization, semanticWeights]);

    const activeArc = useMemo(() => (
        sentenceArcConnections.find((connection) => connection.id === activeArcId) ?? null
    ), [activeArcId, sentenceArcConnections]);

    const updateSentenceLayouts = React.useCallback(() => {
        const container = containerRef.current;
        if (!container) {
            setSentenceLayouts((prev) => (Object.keys(prev).length === 0 ? prev : {}));
            setCanvasSize((prev) => (prev.width === 0 && prev.height === 0 ? prev : { width: 0, height: 0 }));
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

        setSentenceLayouts((prev) => {
            const prevKeys = Object.keys(prev);
            const nextKeys = Object.keys(nextLayouts);
            if (
                prevKeys.length === nextKeys.length &&
                nextKeys.every((key) => {
                    const p = prev[Number(key)];
                    const n = nextLayouts[Number(key)];
                    return p && p.x === n.x && p.y === n.y && p.width === n.width && p.height === n.height;
                })
            ) {
                return prev;
            }
            return nextLayouts;
        });

        const nextWidth = Math.max(container.scrollWidth, Math.ceil(containerRect.width));
        const nextHeight = Math.max(container.scrollHeight, Math.ceil(containerRect.height));
        setCanvasSize((prev) => (
            prev.width === nextWidth && prev.height === nextHeight
                ? prev
                : { width: nextWidth, height: nextHeight }
        ));
    }, [activeArc, deltaVisualization, isSentenceSelectionEnabled, selectedSentenceIndices, sentenceArcConnections]);

    useLayoutEffect(() => {
        const container = containerRef.current;
        if (!container) {
            return;
        }

        const isUserTypingSameText =
            lastUserTextRef.current === content &&
            lastAnnotationSignatureRef.current === annotationSignature;

        if (!isUserTypingSameText) {
            const segments = buildRenderSegments(content, semanticWeights, deltaVisualization);
            if (!tryUpdateSegmentsInPlace(container, segments)) {
                const isFocused = document.activeElement === container;
                const savedSelection = isFocused ? readSelectionOffsets(container) : null;

                container.innerHTML = htmlContent;

                if (isFocused && savedSelection) {
                    restoreSelectionOffsets(container, savedSelection);
                }
            }
            lastUserTextRef.current = content;
            lastAnnotationSignatureRef.current = annotationSignature;
        }

        updateSentenceLayouts();
    }, [annotationSignature, content, deltaVisualization, htmlContent, semanticWeights, updateSentenceLayouts]);

    useEffect(() => {
        const container = containerRef.current;
        if (!container) {
            return;
        }

        let frameId = 0;
        const handleResize = () => {
            window.cancelAnimationFrame(frameId);
            frameId = window.requestAnimationFrame(updateSentenceLayouts);
        };
        const resizeObserver = typeof ResizeObserver === 'undefined'
            ? null
            : new ResizeObserver(handleResize);

        resizeObserver?.observe(container);
        window.addEventListener('resize', handleResize);

        return () => {
            window.cancelAnimationFrame(frameId);
            resizeObserver?.disconnect();
            window.removeEventListener('resize', handleResize);
        };
    }, [updateSentenceLayouts]);

    const handleContainerClick = (event: React.MouseEvent<HTMLDivElement>) => {
        const selection = window.getSelection();
        if (selection && !selection.isCollapsed) {
            return;
        }

        const target = event.target as HTMLElement;

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
            }
        }
    };

    const handleInput = (event: React.FormEvent<HTMLDivElement>) => {
        if (!onChange) {
            return;
        }
        const nextText = extractPlainText(event.currentTarget);
        lastUserTextRef.current = nextText;
        onChange(nextText);
    };

    const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
        if (event.key === 'Enter' && onChange) {
            event.preventDefault();
            insertPlainTextAtSelection('\n');
            const nextText = extractPlainText(event.currentTarget);
            lastUserTextRef.current = nextText;
            onChange(nextText);
        }
    };

    const handlePaste = (event: React.ClipboardEvent<HTMLDivElement>) => {
        if (!onChange) {
            return;
        }
        event.preventDefault();
        const plainText = event.clipboardData.getData('text/plain').replace(/\r\n?/g, '\n');
        insertPlainTextAtSelection(plainText);
        const nextText = extractPlainText(event.currentTarget);
        lastUserTextRef.current = nextText;
        onChange(nextText);
    };

    const arcPaths = useMemo(() => sentenceArcConnections
        .flatMap((connection) => {
            const source = sentenceLayouts[connection.sourceIndex];
            const target = sentenceLayouts[connection.targetIndex];
            if (!source || !target) {
                return [];
            }

            const sourceX = source.x + source.width / 2;
            const sourceY = source.y + source.height / 2;
            const targetX = target.x + target.width / 2;
            const targetY = target.y + target.height / 2;
            const dx = targetX - sourceX;
            const dy = targetY - sourceY;
            const length = Math.hypot(dx, dy) || 1;

            const bow = Math.min(length * 0.35, 160);
            const controlX = Math.min(
                Math.max((sourceX + targetX) / 2 + (dy / length) * bow, 20),
                Math.max(24, canvasSize.width - 20),
            );
            const controlY = (sourceY + targetY) / 2 - (dx / length) * bow;

            return [{
                ...connection,
                d: `M ${sourceX} ${sourceY} Q ${controlX} ${controlY} ${targetX} ${targetY}`,
                sourceX,
                sourceY,
                targetX,
                targetY,
                midpointX: (sourceX + 2 * controlX + targetX) / 4,
                midpointY: (sourceY + 2 * controlY + targetY) / 4,
            }];
        })
        .sort((left, right) => left.score - right.score), [canvasSize.width, sentenceArcConnections, sentenceLayouts]);

    return (
        <div className="render-shell">
            <div
                ref={containerRef}
                role="textbox"
                aria-multiline="true"
                aria-label="Editable document text"
                contentEditable={Boolean(onChange)}
                suppressContentEditableWarning
                spellCheck={false}
                data-placeholder={placeholder}
                className={`render-surface ${className}`.trim()}
                onClick={handleContainerClick}
                onInput={handleInput}
                onKeyDown={handleKeyDown}
                onPaste={handlePaste}
            />

            {canvasSize.width > 0 && canvasSize.height > 0 && arcPaths.length ? (
                <svg
                    className="render-arcs"
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
                        const dim = activeArcId && !isActive ? 0.3 : 1;
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
                                        opacity: isActive ? 0.26 : (0.1 + (connection.score * 0.12)) * dim,
                                        strokeWidth: isActive ? 13 : 8 + (connection.score * 4),
                                    }}
                                />
                                <path
                                    className="sentence-arc-line"
                                    d={connection.d}
                                    style={{
                                        stroke: connection.color,
                                        opacity: isActive ? 0.98 : (0.28 + (connection.score * 0.44)) * dim,
                                        strokeWidth: isActive ? 4.4 : 1.8 + (connection.score * 2.2),
                                    }}
                                    markerEnd={`url(#sentence-arc-arrow-${connection.id})`}
                                />
                                <circle
                                    className="sentence-arc-endpoint"
                                    cx={connection.sourceX}
                                    cy={connection.sourceY}
                                    r={3.4}
                                    style={{ fill: connection.color, opacity: dim }}
                                />
                                <circle
                                    className="sentence-arc-endpoint"
                                    cx={connection.targetX}
                                    cy={connection.targetY}
                                    r={3.4}
                                    style={{ fill: connection.color, opacity: dim }}
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

export default Render;
