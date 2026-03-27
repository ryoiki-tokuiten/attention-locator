import React from 'react';

const ICONS: Record<string, string> = {
    content_copy: '⧉',
    check: '✓',
    close: '✕',
    image: '◫',
};

function iconGlyph(name: string) {
    return ICONS[name] || '•';
}

function escapeAttribute(value: string) {
    return value
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

export function renderIconMarkup(name: string, className = '', attributes: Record<string, string> = {}) {
    const attrString = Object.entries(attributes)
        .map(([key, value]) => `${key}="${escapeAttribute(value)}"`)
        .join(' ');

    const classes = ['icon-slot', className].filter(Boolean).join(' ');
    return `<span class="${escapeAttribute(classes)}" data-icon-name="${escapeAttribute(name)}" aria-hidden="true" ${attrString}>${escapeAttribute(iconGlyph(name))}</span>`;
}

export function setIconSlot(node: Element, name: string) {
    if (!(node instanceof HTMLElement)) {
        return;
    }

    node.dataset.iconName = name;
    node.textContent = iconGlyph(name);
}

export const Icon: React.FC<{
    name: string;
    className?: string;
    style?: React.CSSProperties;
}> = ({ name, className = '', style }) => {
    return (
        <span
            className={['icon-slot', className].filter(Boolean).join(' ')}
            data-icon-name={name}
            style={style}
            aria-hidden="true"
        >
            {iconGlyph(name)}
        </span>
    );
};
