export function clamp(value: number, min = 0, max = 1): number {
    return Math.min(max, Math.max(min, value));
}

export function asFiniteNumber(value: unknown, fallback = 0): number {
    const numeric = typeof value === 'number' ? value : Number(value);
    return Number.isFinite(numeric) ? numeric : fallback;
}

const ATTENTION_COLOR_STOPS = [
    { at: 0, color: [64, 196, 110] },
    { at: 0.5, color: [80, 154, 255] },
    { at: 1, color: [255, 92, 92] },
] as const;

export function attentionRgbForWeight(weight: number): readonly [number, number, number] {
    const normalized = clamp(asFiniteNumber(weight));

    if (normalized <= ATTENTION_COLOR_STOPS[0].at) {
        return ATTENTION_COLOR_STOPS[0].color;
    }

    for (let index = 1; index < ATTENTION_COLOR_STOPS.length; index += 1) {
        const previous = ATTENTION_COLOR_STOPS[index - 1];
        const current = ATTENTION_COLOR_STOPS[index];

        if (normalized <= current.at) {
            const range = current.at - previous.at || 1;
            const progress = (normalized - previous.at) / range;
            return previous.color.map((channel, channelIndex) => (
                Math.round(channel + ((current.color[channelIndex] - channel) * progress))
            )) as [number, number, number];
        }
    }

    return ATTENTION_COLOR_STOPS[ATTENTION_COLOR_STOPS.length - 1].color;
}

export function attentionColorForWeight(weight: number): string {
    return `rgb(${attentionRgbForWeight(weight).join(', ')})`;
}
