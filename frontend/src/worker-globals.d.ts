interface Worker extends EventTarget {
    onmessage: ((this: Worker, event: MessageEvent) => unknown) | null;
    onmessageerror: ((this: Worker, event: MessageEvent) => unknown) | null;
    postMessage(message: unknown, transfer?: Transferable[]): void;
    terminate(): void;
}

interface WorkerOptions {
    credentials?: RequestCredentials;
    name?: string;
    type?: 'classic' | 'module';
}

declare var Worker: {
    prototype: Worker;
    new (scriptURL: string | URL, options?: WorkerOptions): Worker;
};
