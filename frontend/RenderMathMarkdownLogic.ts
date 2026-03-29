import DOMPurify from 'dompurify';
import he from 'he';
import { nanoid } from 'nanoid';
import type { Code } from 'mdast';
import type { Element as HastElement, Root as HastRoot, Text as HastText } from 'hast';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import rehypeStringify from 'rehype-stringify';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkParse from 'remark-parse';
import remarkRehype from 'remark-rehype';
import { unified } from 'unified';

import { highlightCodeSync, isLanguageSupported, resolveLanguage } from './src/Shiki';
import type { DeltaTokenVisualization, DeltaVisualizationMode, SemanticWeight } from './src/types/semantic';
import { renderIconMarkup } from './src/UI/Icons';

const MAX_CODE_HIGHLIGHT_SIZE = 50000;
const SKIPPED_SEMANTIC_TAGS = new Set([
    'annotation',
    'semantics',
    'svg',
    'script',
    'style',
    'textarea',
    'input',
]);
const SEMANTIC_BLOCK_TAGS = new Set([
    'p',
    'li',
    'h1',
    'h2',
    'h3',
    'h4',
    'h5',
    'h6',
    'td',
    'th',
    'blockquote',
    'pre',
]);

interface TextLeaf {
    node: HastText;
    parent: HastRoot | HastElement;
    index: number;
    sourceStart: number;
    sourceEnd: number;
}

interface SemanticRange {
    sourceStart: number;
    sourceEnd: number;
    semantic: SemanticWeight;
}

interface DeltaRange {
    sourceStart: number;
    sourceEnd: number;
    token: DeltaTokenVisualization;
}

interface DeltaVisualizationInput {
    mode: DeltaVisualizationMode;
    tokens: DeltaTokenVisualization[];
}

interface CodeBlockOptions {
    codeId: string;
    language: string;
    displayLang: string;
    highlightedCode: string;
    extraClass?: string;
}

interface OutputBlockOptions {
    title: string;
    content: string;
    extraClass?: string;
}

interface ImageItemOptions {
    dataUrl: string;
    mimeType: string;
    formatLabel: string;
    estimatedSizeKB: number;
}

const COPY_ICON_SVG = renderIconMarkup('content_copy', '', { style: 'font-size: 16px;' });

export function escapeHtml(value: string): string {
    return he.encode(value, { useNamedReferences: true });
}

export function decodeEntities(value: string): string {
    return he.decode(value);
}

export function generateCodeId(): string {
    return `code-${nanoid(9)}`;
}

function asFiniteSemanticWeight(weight: unknown, fallback = 0.5): number {
    const numeric = typeof weight === 'number' ? weight : Number(weight);
    if (!Number.isFinite(numeric)) {
        return fallback;
    }
    return Math.min(1, Math.max(0, numeric));
}

function toClassList(value: unknown): string[] {
    if (Array.isArray(value)) {
        return value.map(String);
    }
    if (typeof value === 'string') {
        return value.split(/\s+/).filter(Boolean);
    }
    return [];
}

function isSkippedSemanticElement(element: HastElement): boolean {
    if (SKIPPED_SEMANTIC_TAGS.has(element.tagName)) {
        return true;
    }

    const classes = toClassList(element.properties?.className);
    return classes.some((className) =>
        className === 'exec-image-item' ||
        className === 'exec-image-grid' ||
        className === 'exec-output-block'
    );
}

function shouldAnnotateElement(element: HastElement): boolean {
    if (!SEMANTIC_BLOCK_TAGS.has(element.tagName)) {
        return false;
    }

    if (element.tagName === 'li') {
        return !element.children.some((child) => child.type === 'element' && child.tagName === 'p');
    }

    return true;
}

function appendStyleDeclaration(existingStyle: unknown, declaration: string): string {
    const normalizedExisting = typeof existingStyle === 'string' ? existingStyle.trim() : '';
    if (!normalizedExisting) {
        return declaration;
    }

    return normalizedExisting.endsWith(';')
        ? `${normalizedExisting} ${declaration}`
        : `${normalizedExisting}; ${declaration}`;
}

interface ColorStop {
    at: number;
    color: [number, number, number];
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

function rgbVar(color: [number, number, number]): string {
    return color.join(', ');
}

function styleForSemanticWeight(weight: number): string {
    const normalizedWeight = asFiniteSemanticWeight(weight);
    const primaryColor = interpolateColorStops(normalizedWeight, [
        { at: 0, color: [64, 196, 110] },
        { at: 0.5, color: [80, 154, 255] },
        { at: 1, color: [255, 92, 92] },
    ]);
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
    const underlineOpacity = (0.24 + (normalizedWeight * 0.22)).toFixed(3);

    return [
        `--semantic-weight: ${normalizedWeight.toFixed(3)}`,
        `--semantic-primary-rgb: ${rgbVar(primaryColor)}`,
        `--semantic-secondary-rgb: ${rgbVar(secondaryColor)}`,
        `--semantic-text-rgb: ${rgbVar(textColor)}`,
        `--semantic-fill-opacity: ${fillOpacity}`,
        `--semantic-glow-opacity: ${glowOpacity}`,
        `--semantic-edge-opacity: ${edgeOpacity}`,
        `--semantic-underline-opacity: ${underlineOpacity}`,
    ].join('; ');
}

function semanticScoreLabel(semantic: SemanticWeight): string {
    const weight = asFiniteSemanticWeight(semantic.weight);
    return `Attention score ${weight.toFixed(3)} · ${semantic.band}`;
}

function createSemanticSpan(text: string, semantic: SemanticWeight): HastElement {
    const weight = asFiniteSemanticWeight(semantic.weight);

    return {
        type: 'element',
        tagName: 'span',
        properties: {
            className: ['semantic-sentence', `semantic-${semantic.band}`],
            'data-semantic-band': semantic.band,
            'data-semantic-weight': weight.toFixed(3),
            'data-semantic-index': String(semantic.index),
            'data-semantic-score': weight.toFixed(3),
            'data-semantic-score-label': semanticScoreLabel(semantic),
            title: semanticScoreLabel(semantic),
            style: styleForSemanticWeight(weight),
        },
        children: [{ type: 'text', value: text }],
    };
}

function deltaColor(weight: number): [number, number, number] {
    return interpolateColorStops(asFiniteSemanticWeight(weight), [
        { at: 0, color: [64, 196, 110] },
        { at: 0.5, color: [80, 154, 255] },
        { at: 1, color: [255, 92, 92] },
    ]);
}

function sourceDrivenColor(positionRatio: number): [number, number, number] {
    return interpolateColorStops(asFiniteSemanticWeight(positionRatio), [
        { at: 0, color: [40, 209, 122] },
        { at: 0.5, color: [69, 138, 255] },
        { at: 1, color: [255, 110, 110] },
    ]);
}

function deltaTooltip(token: DeltaTokenVisualization, mode: DeltaVisualizationMode): string {
    if (mode === 'drift') {
        return `Drift ${token.driftNorm.toFixed(3)} · peak layer ${token.peakLayer + 1}`;
    }
    if (mode === 'local-global') {
        return `Flow ${token.incomingFlowNorm.toFixed(3)} · local ${token.localShare.toFixed(2)} · medium ${token.mediumShare.toFixed(2)} · global ${token.globalShare.toFixed(2)}`;
    }
    return `Flow ${token.incomingFlowNorm.toFixed(3)} · ${token.topSources.length} source contributors`;
}

function styleForDeltaToken(token: DeltaTokenVisualization, mode: DeltaVisualizationMode, tokenCount: number): string {
    const intensity = mode === 'drift' ? token.driftNorm : token.incomingFlowNorm;
    const [baseRed, baseGreen, baseBlue] = deltaColor(intensity);
    const localColor: [number, number, number] = [40, 209, 122];
    const mediumColor: [number, number, number] = [69, 138, 255];
    const globalColor: [number, number, number] = [255, 110, 110];
    const primaryComposition = token.topSources[0];
    const secondaryComposition = token.topSources[1] ?? token.topSources[0];
    const compositionColorA = sourceDrivenColor((primaryComposition?.sourceIndex ?? token.index) / Math.max(1, tokenCount - 1));
    const compositionColorB = sourceDrivenColor((secondaryComposition?.sourceIndex ?? token.index) / Math.max(1, tokenCount - 1));
    const driftColor = interpolateColorStops(token.peakLayerRatio, [
        { at: 0, color: [52, 211, 153] },
        { at: 0.5, color: [96, 165, 250] },
        { at: 1, color: [248, 113, 113] },
    ]);

    const fillOpacity = (0.18 + (intensity * 0.30)).toFixed(3);
    const glowOpacity = (0.12 + (intensity * 0.24)).toFixed(3);
    const textOpacity = (0.74 + (intensity * 0.20)).toFixed(3);
    const baseDeclarations = [
        `--delta-weight: ${intensity.toFixed(3)}`,
        `--delta-primary-rgb: ${rgbVar([baseRed, baseGreen, baseBlue])}`,
        `--delta-local-rgb: ${rgbVar(localColor)}`,
        `--delta-medium-rgb: ${rgbVar(mediumColor)}`,
        `--delta-global-rgb: ${rgbVar(globalColor)}`,
        `--delta-drift-rgb: ${rgbVar(driftColor)}`,
        `--delta-comp-a-rgb: ${rgbVar(compositionColorA)}`,
        `--delta-comp-b-rgb: ${rgbVar(compositionColorB)}`,
        `--delta-local-share: ${token.localShare.toFixed(3)}`,
        `--delta-medium-share: ${token.mediumShare.toFixed(3)}`,
        `--delta-global-share: ${token.globalShare.toFixed(3)}`,
        `--delta-fill-opacity: ${fillOpacity}`,
        `--delta-glow-opacity: ${glowOpacity}`,
        `--delta-text-opacity: ${textOpacity}`,
    ];

    if (mode === 'local-global') {
        return [
            ...baseDeclarations,
            `background: linear-gradient(135deg, rgba(${rgbVar(localColor)}, ${0.10 + (token.localShare * 0.46)}) 0%, rgba(${rgbVar(mediumColor)}, ${0.10 + (token.mediumShare * 0.48)}) 56%, rgba(${rgbVar(globalColor)}, ${0.10 + (token.globalShare * 0.52)}) 100%)`,
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

function createDeltaSpan(
    text: string,
    token: DeltaTokenVisualization,
    mode: DeltaVisualizationMode,
    tokenCount: number,
): HastElement {
    const intensity = mode === 'drift' ? token.driftNorm : token.incomingFlowNorm;
    return {
        type: 'element',
        tagName: 'span',
        properties: {
            className: ['delta-token', `delta-token-${mode}`],
            'data-delta-index': String(token.index),
            'data-delta-mode': mode,
            'data-delta-intensity': intensity.toFixed(3),
            'data-delta-label': token.displayToken,
            title: deltaTooltip(token, mode),
            style: styleForDeltaToken(token, mode, tokenCount),
        },
        children: [{ type: 'text', value: text }],
    };
}

function getNodeOffset(node: HastRoot | HastElement | HastText, edge: 'start' | 'end'): number | null {
    const position = node.position as {
        start?: { offset?: number };
        end?: { offset?: number };
    } | undefined;

    const offset = position?.[edge]?.offset;
    return typeof offset === 'number' ? offset : null;
}

function collectTextLeaves(
    node: HastRoot | HastElement | HastText,
    leaves: TextLeaf[],
    parent?: HastRoot | HastElement,
    index?: number,
    skipSemantic = false,
) {
    if (node.type === 'text') {
        if (!skipSemantic && parent && typeof index === 'number') {
            const sourceStart = getNodeOffset(node, 'start');
            const sourceEnd = getNodeOffset(node, 'end');
            if (sourceStart != null && sourceEnd != null && sourceEnd > sourceStart) {
                leaves.push({ node, parent, index, sourceStart, sourceEnd });
            }
        }
        return;
    }

    if (node.type !== 'element' && node.type !== 'root') {
        return;
    }

    const shouldSkip = skipSemantic || (node.type === 'element' && isSkippedSemanticElement(node));

    if (!('children' in node) || !node.children) {
        return;
    }

    node.children.forEach((child, childIndex) => {
        collectTextLeaves(child as HastRoot | HastElement | HastText, leaves, node, childIndex, shouldSkip);
    });
}

function applyRanges<T>(
    leaves: TextLeaf[],
    ranges: Array<{ sourceStart: number; sourceEnd: number; payload: T }>,
    createSpan: (text: string, payload: T) => HastElement,
) {
    if (!ranges.length) {
        return;
    }

    const orderedLeaves = [...leaves].sort((left, right) => right.sourceStart - left.sourceStart);
    const orderedRanges = [...ranges].sort((left, right) => left.sourceStart - right.sourceStart);

    for (const leaf of orderedLeaves) {
        const applicableRanges = orderedRanges.filter(
            (range) => range.sourceEnd > leaf.sourceStart && range.sourceStart < leaf.sourceEnd,
        );
        if (!applicableRanges.length) {
            continue;
        }

        const fragments: Array<HastText | HastElement> = [];
        let cursor = 0;

        for (const range of applicableRanges) {
            const overlapStart = Math.max(leaf.sourceStart, range.sourceStart);
            const overlapEnd = Math.min(leaf.sourceEnd, range.sourceEnd);
            const localStart = Math.max(0, overlapStart - leaf.sourceStart);
            const localEnd = Math.min(leaf.node.value.length, overlapEnd - leaf.sourceStart);

            if (localStart > cursor) {
                fragments.push({ type: 'text', value: leaf.node.value.slice(cursor, localStart) });
            }

            if (localEnd > localStart) {
                fragments.push(createSpan(leaf.node.value.slice(localStart, localEnd), range.payload));
            }

            cursor = Math.max(cursor, localEnd);
        }

        if (cursor < leaf.node.value.length) {
            fragments.push({ type: 'text', value: leaf.node.value.slice(cursor) });
        }

        leaf.parent.children.splice(leaf.index, 1, ...fragments);
    }
}

function annotateElementWithSemanticWeights(element: HastElement, semanticWeights: SemanticWeight[]) {
    const leaves: TextLeaf[] = [];
    collectTextLeaves(element, leaves);

    if (!leaves.length) {
        return;
    }

    const blockStart = Math.min(...leaves.map((leaf) => leaf.sourceStart));
    const blockEnd = Math.max(...leaves.map((leaf) => leaf.sourceEnd));
    const ranges: SemanticRange[] = semanticWeights
        .filter((semantic) => (
            semantic.sourceStart != null &&
            semantic.sourceEnd != null &&
            semantic.sourceEnd > semantic.sourceStart &&
            semantic.sourceEnd > blockStart &&
            semantic.sourceStart < blockEnd
        ))
        .map((semantic) => ({
            sourceStart: semantic.sourceStart as number,
            sourceEnd: semantic.sourceEnd as number,
            semantic: {
                ...semantic,
                weight: asFiniteSemanticWeight(semantic.weight),
            },
        }));

    if (ranges.length) {
        const strongestSemantic = ranges.reduce((strongest, range) => (
            range.semantic.weight > strongest.weight ? range.semantic : strongest
        ), ranges[0].semantic);
        const strongestWeight = asFiniteSemanticWeight(strongestSemantic.weight);
        const existingClasses = toClassList(element.properties?.className);

        element.properties = element.properties || {};
        element.properties.className = [
            ...existingClasses,
            'semantic-block',
            `semantic-block-${strongestSemantic.band}`,
        ];
        element.properties['data-semantic-block-band'] = strongestSemantic.band;
        element.properties['data-semantic-block-weight'] = strongestWeight.toFixed(3);
        element.properties['data-semantic-block-score-label'] = semanticScoreLabel(strongestSemantic);
        element.properties.title = semanticScoreLabel(strongestSemantic);
        element.properties.style = appendStyleDeclaration(
            element.properties.style,
            styleForSemanticWeight(strongestWeight),
        );
    }

    applyRanges(
        leaves,
        ranges.map((range) => ({
            sourceStart: range.sourceStart,
            sourceEnd: range.sourceEnd,
            payload: range.semantic,
        })),
        createSemanticSpan,
    );
}

function annotateElementWithDeltaTokens(
    element: HastElement,
    deltaVisualization: DeltaVisualizationInput,
) {
    const leaves: TextLeaf[] = [];
    collectTextLeaves(element, leaves);

    if (!leaves.length) {
        return;
    }

    const blockStart = Math.min(...leaves.map((leaf) => leaf.sourceStart));
    const blockEnd = Math.max(...leaves.map((leaf) => leaf.sourceEnd));
    const ranges: DeltaRange[] = deltaVisualization.tokens
        .filter((token) => (
            token.visible &&
            token.sourceStart != null &&
            token.sourceEnd != null &&
            token.sourceEnd > token.sourceStart &&
            token.sourceEnd > blockStart &&
            token.sourceStart < blockEnd
        ))
        .map((token) => ({
            sourceStart: token.sourceStart as number,
            sourceEnd: token.sourceEnd as number,
            token,
        }));

    applyRanges(
        leaves,
        ranges.map((range) => ({
            sourceStart: range.sourceStart,
            sourceEnd: range.sourceEnd,
            payload: range.token,
        })),
        (text, token) => createDeltaSpan(text, token, deltaVisualization.mode, deltaVisualization.tokens.length),
    );
}

function renderCodeBlockHtml({
    codeId,
    language,
    displayLang,
    highlightedCode,
    extraClass = '',
}: CodeBlockOptions): string {
    const isShikiBlock = highlightedCode.includes('<pre class="shiki');

    const contentHtml = isShikiBlock
        ? highlightedCode
        : `<pre><code id="${escapeHtml(codeId)}" class="language-${escapeHtml(language)}">${highlightedCode}</code></pre>`;

    return `<div class="code-block-container ${extraClass}">
<div class="code-block-header">
<span class="code-block-title">${escapeHtml(displayLang)}</span>
<button class="code-copy-icon copy-code-btn" data-code-id="${escapeHtml(codeId)}" title="Copy code" onclick="window.copyCodeBlock && window.copyCodeBlock('${escapeHtml(codeId)}')">${COPY_ICON_SVG}</button>
</div>
<div class="code-block-content">
${contentHtml}
</div>
</div>`;
}

function renderOutputBlockHtml({ title, content, extraClass = '' }: OutputBlockOptions): string {
    return `<div class="code-block-container exec-output-block ${extraClass}">
<div class="code-block-header">
<span class="code-block-title">${escapeHtml(title)}</span>
</div>
<div class="code-block-content exec-output-content">
<pre><code class="exec-output-text">${content}</code></pre>
</div>
</div>`;
}

function renderImageItemHtml({ dataUrl, mimeType, formatLabel, estimatedSizeKB }: ImageItemOptions): string {
    return `<div class="exec-image-item" data-src="${escapeHtml(dataUrl)}" data-mime="${escapeHtml(mimeType)}" data-format="${escapeHtml(formatLabel)}" data-size="${estimatedSizeKB}" onclick="window.previewImage && window.previewImage(this)">
<img src="${escapeHtml(dataUrl)}" alt="Generated visualization" class="exec-rendered-image" loading="lazy" onerror="this.onerror=null;this.parentElement.innerHTML='<div class=\\'exec-image-error\\'>Failed to render image</div>';"/>
</div>`;
}

function renderImageGridHtml(imagesHtml: string): string {
    return `<div class="code-block-container exec-image-block">
<div class="code-block-header">
<span class="code-block-title">FIGURE</span>
</div>
<div class="exec-image-grid">${imagesHtml}</div>
</div>`;
}

function renderImageErrorHtml(errorMessage: string): string {
    return `<div class="exec-image-item exec-image-error-item">
<div class="exec-image-error">${escapeHtml(errorMessage)}</div>
</div>`;
}

function renderCompleteDocument(content: string, type: 'latex' | 'html'): string {
    const codeId = generateCodeId();
    const codeToHighlight = type === 'html' ? decodeEntities(content) : content;
    let highlightedCode = '';

    try {
        if (content.length <= MAX_CODE_HIGHLIGHT_SIZE && isLanguageSupported(type)) {
            highlightedCode = highlightCodeSync(codeToHighlight, type);
        } else {
            highlightedCode = escapeHtml(codeToHighlight);
        }
    } catch {
        highlightedCode = escapeHtml(codeToHighlight);
    }

    const displayLang = type === 'latex' ? 'LATEX DOCUMENT' : 'HTML';
    const codeBlock = renderCodeBlockHtml({ codeId, language: type, displayLang, highlightedCode });

    return `<div class="rich-content-display"><div class="latex-content-wrapper"><div style="font-size: 1.4rem; line-height: 1.6;">${codeBlock}</div></div></div>`;
}

function renderCodeBlock(code: string, language: string): string {
    const codeId = generateCodeId();
    const validLanguage = language && isLanguageSupported(language) ? resolveLanguage(language) : 'plaintext';
    const cleanCode = decodeEntities(code);
    let highlightedCode = '';

    const alreadyHighlighted = /<span\s+class="(hljs-|shiki)/.test(cleanCode);

    if (alreadyHighlighted) {
        highlightedCode = cleanCode;
    } else if (cleanCode.length > MAX_CODE_HIGHLIGHT_SIZE) {
        highlightedCode = escapeHtml(cleanCode);
        console.warn(`Code block too large (${cleanCode.length} chars), skipping syntax highlighting`);
    } else {
        try {
            highlightedCode = highlightCodeSync(cleanCode, validLanguage);
        } catch {
            highlightedCode = escapeHtml(cleanCode);
        }
    }

    return renderCodeBlockHtml({
        codeId,
        language: validLanguage || 'text',
        displayLang: (language || 'Code').toUpperCase(),
        highlightedCode,
    });
}

let executionCellCounter = 0;
let storedBlocks: string[] = [];

function renderCodeExecutionBlock(code: string, language: string): string {
    const codeId = generateCodeId();
    const validLanguage = language && isLanguageSupported(language) ? resolveLanguage(language) : 'python';
    const cleanCode = decodeEntities(code);
    let highlightedCode = '';

    const alreadyHighlighted = /<span\s+class="(hljs-|shiki)/.test(cleanCode);

    if (alreadyHighlighted) {
        highlightedCode = cleanCode;
    } else if (cleanCode.length > MAX_CODE_HIGHLIGHT_SIZE) {
        highlightedCode = escapeHtml(cleanCode);
    } else {
        try {
            highlightedCode = highlightCodeSync(cleanCode, validLanguage);
        } catch {
            highlightedCode = escapeHtml(cleanCode);
        }
    }

    return renderCodeBlockHtml({
        codeId,
        language: validLanguage,
        displayLang: validLanguage.toUpperCase(),
        highlightedCode,
        extraClass: 'exec-code-block',
    });
}

function renderExecutionOutputBlock(output: string): string {
    const escapedOutput = escapeHtml(output);
    const loweredOutput = output.toLowerCase();
    const hasError = loweredOutput.includes('error') ||
        loweredOutput.includes('traceback') ||
        loweredOutput.includes('exception');

    return renderOutputBlockHtml({
        title: hasError ? 'ERROR' : 'OUTPUT',
        content: escapedOutput,
        extraClass: hasError ? 'exec-output-error' : '',
    });
}

function renderExecutionImageBlock(base64Data: string, mimeType: string): string {
    if (!base64Data || !base64Data.trim()) {
        return renderImageErrorHtml('Empty image data received');
    }

    const normalizedMimeType = mimeType?.startsWith('image/') ? mimeType : 'image/png';
    const formatLabel = normalizedMimeType.replace('image/', '').toUpperCase();

    let dataUrl: string;
    if (base64Data.startsWith('data:')) {
        dataUrl = base64Data;
    } else {
        const cleanBase64 = base64Data.replace(/\s/g, '');
        if (!/^[A-Za-z0-9+/=]+$/.test(cleanBase64)) {
            return renderImageErrorHtml('Invalid base64 encoding');
        }
        dataUrl = `data:${normalizedMimeType};base64,${cleanBase64}`;
    }

    const estimatedSizeKB = Math.round((base64Data.length * 3) / 4 / 1024);
    return renderImageItemHtml({ dataUrl, mimeType: normalizedMimeType, formatLabel, estimatedSizeKB });
}

function preprocessCodeExecutionBlocks(content: string): { processed: string; blocks: string[] } {
    let processed = content;
    executionCellCounter = 0;
    storedBlocks = [];

    const storePlaceholder = (html: string): string => {
        const index = storedBlocks.length;
        storedBlocks.push(html);
        return `<!--STORED_BLOCK_${index}-->`;
    };

    const codeExecPattern = /<!-- CODE_EXECUTION_START -->\s*\n?<!-- LANGUAGE: (\w+) -->\s*\n?```\w*\n([\s\S]*?)\n```\s*\n?<!-- CODE_EXECUTION_END -->/g;
    processed = processed.replace(codeExecPattern, (_, language, code) => {
        executionCellCounter += 1;
        return storePlaceholder(renderCodeExecutionBlock(code.trim(), language.toLowerCase()));
    });

    const outputPattern = /<!-- EXECUTION_OUTPUT_START -->\s*\n?```\n?([\s\S]*?)\n?```\s*\n?<!-- EXECUTION_OUTPUT_END -->/g;
    processed = processed.replace(outputPattern, (_, output) => {
        return storePlaceholder(renderExecutionOutputBlock(output.trim()));
    });

    const imagePattern = /<!-- EXECUTION_IMAGE_START -->\s*\n?<!-- MIME_TYPE: ([^\s]+) -->\s*\n?([\s\S]*?)\n?<!-- EXECUTION_IMAGE_END -->/g;
    const imageHtmlParts: string[] = [];
    processed = processed.replace(imagePattern, (_, mimeType, base64Data) => {
        const html = renderExecutionImageBlock(base64Data.trim(), mimeType);
        imageHtmlParts.push(html);
        return `<!--TEMP_IMG_${imageHtmlParts.length - 1}-->`;
    });

    let finalProcessed = '';
    let currentImageGroup: number[] = [];
    const parts = processed.split(/(<!--TEMP_IMG_\d+-->)/);

    for (const part of parts) {
        const match = part.match(/<!--TEMP_IMG_(\d+)-->/);

        if (match) {
            currentImageGroup.push(parseInt(match[1], 10));
            continue;
        }

        if (part.trim() !== '') {
            if (currentImageGroup.length > 0) {
                const imagesHtml = currentImageGroup.map((index) => imageHtmlParts[index]).join('');
                finalProcessed += storePlaceholder(renderImageGridHtml(imagesHtml));
                currentImageGroup = [];
            }

            finalProcessed += part;
        } else if (currentImageGroup.length === 0) {
            finalProcessed += part;
        }
    }

    if (currentImageGroup.length > 0) {
        const imagesHtml = currentImageGroup.map((index) => imageHtmlParts[index]).join('');
        finalProcessed += storePlaceholder(renderImageGridHtml(imagesHtml));
    }

    return { processed: finalProcessed, blocks: storedBlocks };
}

function rehypeAddClasses(
    semanticWeights: SemanticWeight[],
    deltaVisualization: DeltaVisualizationInput | null,
) {
    return () => {
        return (tree: HastRoot) => {
            const visit = (node: HastRoot | HastElement | HastText) => {
                if (node.type === 'text') {
                    return;
                }

                if (node.type !== 'element' && node.type !== 'root') {
                    return;
                }

                if (node.type === 'element') {
                    const element = node;

                    if (['h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(element.tagName)) {
                        const level = parseInt(element.tagName[1], 10);
                        const existingClasses = toClassList(element.properties?.className);
                        const levelClass = level <= 3 ? `token-heading${level}` : '';
                        element.properties = element.properties || {};
                        element.properties.className = ['token-heading', levelClass, ...existingClasses].filter(Boolean);
                    }

                    if (element.tagName === 'strong') {
                        element.properties = element.properties || {};
                        element.properties.className = [...toClassList(element.properties.className), 'token-critical'];
                    }

                    if (element.tagName === 'img') {
                        element.properties = element.properties || {};
                        const src = String(element.properties.src || '');
                        const alt = String(element.properties.alt || 'Image');

                        const wrapper: HastElement = {
                            type: 'element',
                            tagName: 'div',
                            properties: {
                                className: ['exec-image-item'],
                                'data-src': src,
                                'data-format': 'IMAGE',
                                'data-alt': alt,
                                onclick: 'window.previewImage && window.previewImage(this)',
                            },
                            children: [{
                                ...element,
                                properties: {
                                    ...element.properties,
                                    className: ['exec-rendered-image', 'markdown-image'],
                                    loading: 'lazy',
                                },
                            }],
                        };

                        Object.assign(element, wrapper);
                    }
                }

                if ('children' in node && node.children) {
                    node.children.forEach((child) => visit(child as HastRoot | HastElement | HastText));
                }

                if (node.type === 'element' && shouldAnnotateElement(node) && !isSkippedSemanticElement(node)) {
                    if (deltaVisualization) {
                        annotateElementWithDeltaTokens(node, deltaVisualization);
                    } else {
                        annotateElementWithSemanticWeights(node, semanticWeights);
                    }
                }
            };

            tree.children.forEach((child) => visit(child as HastRoot | HastElement | HastText));
        };
    };
}

function createProcessor(
    semanticWeights: SemanticWeight[],
    deltaVisualization: DeltaVisualizationInput | null,
) {
    return unified()
        .use(remarkParse)
        .use(remarkMath)
        .use(remarkGfm)
        .use(remarkRehype, {
            allowDangerousHtml: true,
            handlers: {
                code(_state: unknown, node: Code) {
                    return {
                        type: 'raw',
                        value: renderCodeBlock(node.value, node.lang || ''),
                    };
                },
            },
        })
        .use(rehypeRaw)
        .use(rehypeKatex, {
            throwOnError: false,
            trust: true,
            strict: false,
        })
        .use(rehypeAddClasses(semanticWeights, deltaVisualization))
        .use(rehypeStringify, {
            allowDangerousHtml: true,
        });
}

const LATEX_COMMAND_PATTERN = /\\[a-zA-Z]+/;
const MATH_NOTATION_PATTERNS = [
    /[_^]\{[^}]+\}/,
    /\\[{}]/,
];

function containsLatexMath(content: string): boolean {
    if (LATEX_COMMAND_PATTERN.test(content)) {
        return true;
    }
    return MATH_NOTATION_PATTERNS.some((pattern) => pattern.test(content));
}

function convertBacktickedLatexToMath(content: string): string {
    return content.replace(/(?<!`)`([^`]+)`(?!`)/g, (match, codeContent) => {
        if (containsLatexMath(codeContent)) {
            return `$$${codeContent}$$`;
        }
        return match;
    });
}

export function renderMathContent(
    content: string,
    semanticWeights: SemanticWeight[] = [],
    deltaVisualization: DeltaVisualizationInput | null = null,
): string {
    if (!content) {
        return '';
    }

    const mathConvertedContent = convertBacktickedLatexToMath(content);
    const { processed: preprocessedContent, blocks } = preprocessCodeExecutionBlocks(mathConvertedContent);
    const trimmed = preprocessedContent.trim();

    if (trimmed.startsWith('\\documentclass') || (trimmed.includes('\\begin{document}') && trimmed.includes('\\end{document}'))) {
        return renderCompleteDocument(preprocessedContent, 'latex');
    }
    if (trimmed.startsWith('<!DOCTYPE html>') || (trimmed.startsWith('<html') && trimmed.includes('</html>'))) {
        return renderCompleteDocument(preprocessedContent, 'html');
    }

    const processor = createProcessor(semanticWeights, deltaVisualization);
    let resultHtml: string;

    try {
        const result = processor.processSync(preprocessedContent);
        resultHtml = String(result);
    } catch (error) {
        console.error('Unified processing error:', error);
        resultHtml = `<pre>${escapeHtml(preprocessedContent)}</pre>`;
    }

    blocks.forEach((html, index) => {
        resultHtml = resultHtml.replace(`<!--STORED_BLOCK_${index}-->`, html);
    });

    DOMPurify.addHook('afterSanitizeAttributes', (node) => {
        if (node instanceof HTMLElement && node.style.color) {
            node.style.removeProperty('color');
        }
    });

    resultHtml = DOMPurify.sanitize(resultHtml, {
        ADD_TAGS: [
            'div', 'span', 'pre', 'code',
            'math', 'annotation', 'semantics', 'mrow', 'mn', 'mo', 'mi', 'msup', 'msub', 'mfrac',
            'mtext', 'mspace', 'msqrt', 'mroot', 'mover', 'munder', 'munderover', 'mtable', 'mtr', 'mtd',
            'table', 'tr', 'td', 'th', 'tbody', 'thead', 'caption', 'colgroup', 'col',
            'svg', 'polygon', 'polyline', 'line', 'rect', 'path', 'circle', 'g', 'defs', 'use', 'symbol',
            'img', 'del', 's', 'input',
        ],
        ADD_ATTR: [
            'class', 'style', 'title', 'id',
            'data-code-id', 'data-cell-number',
            'data-src', 'data-mime', 'data-format', 'data-size', 'data-alt',
            'data-semantic-band', 'data-semantic-weight', 'data-semantic-index',
            'data-semantic-score', 'data-semantic-score-label',
            'data-semantic-block-band', 'data-semantic-block-weight', 'data-semantic-block-score-label',
            'data-delta-index', 'data-delta-mode', 'data-delta-intensity', 'data-delta-label',
            'viewBox', 'd', 'fill', 'stroke', 'stroke-width',
            'x', 'y', 'width', 'height', 'rx', 'ry',
            'points', 'x1', 'y1', 'x2', 'y2', 'cx', 'cy', 'r',
            'xmlns', 'xmlns:xlink', 'xlink:href', 'transform',
            'src', 'alt', 'loading',
            'align', 'valign', 'colspan', 'rowspan',
            'type', 'checked', 'disabled',
        ],
        ADD_URI_SCHEMES: ['data'],
        RETURN_TRUSTED_TYPE: false,
    } as never) as unknown as string;

    DOMPurify.removeHook('afterSanitizeAttributes');

    return `<div class="rich-content-display"><div class="latex-content-wrapper" style="font-size: 1.4rem; line-height: 1.6;">${resultHtml}</div></div>`;
}
