import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

import { onHighlighterReady, isHighlighterReady } from './src/Shiki';
import type { SemanticWeight } from './src/types/semantic';
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
}

export const RenderMathMarkdown: React.FC<RenderMathMarkdownProps> = ({
    content,
    className = '',
    semanticWeights = [],
}) => {
    const _isHighlightingReady = useHighlighting();
    const containerRef = useRef<HTMLDivElement>(null);
    const busyRef = useRef<Record<string, boolean>>({});

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
            const windowWithPreview = window as Window & { previewImage?: (node: HTMLElement) => void };
            if (typeof windowWithPreview.previewImage === 'function') {
                windowWithPreview.previewImage(imageItem);
            } else {
                const src = imageItem.dataset.src || (imageItem.querySelector('img') as HTMLImageElement | null)?.src;
                const format = imageItem.dataset.format || 'PNG';
                const alt = imageItem.dataset.alt || 'Generated Figure';

                if (src) {
                    window.dispatchEvent(new CustomEvent('exec-image-preview', {
                        detail: { src, alt, format },
                    }));
                }
            }

            return;
        }

        if (target.id?.startsWith('toggle-') || target.closest('[id^="toggle-"]')) {
            const toggleElement = target.id?.startsWith('toggle-')
                ? target
                : target.closest('[id^="toggle-"]') as HTMLElement | null;

            if (!toggleElement) {
                return;
            }

            const codeId = toggleElement.id.replace('toggle-', '');
            const codeContent = document.getElementById(codeId);
            const container = codeContent?.closest('.code-block-container');

            if (codeContent && container) {
                const isExpanded = codeContent.classList.contains('expanded');
                codeContent.classList.toggle('expanded', !isExpanded);
                codeContent.classList.toggle('collapsed', isExpanded);
                toggleElement.classList.toggle('expanded', !isExpanded);
                container.classList.toggle('expanded', !isExpanded);
                container.classList.toggle('collapsed', isExpanded);
            }
        }
    };

    const htmlContent = useMemo(() => {
        return renderMathContent(content, semanticWeights);
    }, [content, semanticWeights, _isHighlightingReady]);

    return (
        <div
            ref={containerRef}
            className={`render-math-markdown ${className}`.trim()}
            onClick={handleContainerClick}
            dangerouslySetInnerHTML={{ __html: htmlContent }}
        />
    );
};

export default RenderMathMarkdown;
