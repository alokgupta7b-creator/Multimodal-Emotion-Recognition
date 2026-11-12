/** Framework-agnostic retry and abort scope helpers for Emotion Studio API requests. */

export type RetryStateCallbacks<T> = {
  onStart: () => void;
  onSuccess: (value: T) => void;
  onError: (message: string) => void;
  onFinally: () => void;
  fallbackError: string;
};

export type RequestToken = {
  signal: AbortSignal;
  isLatest: () => boolean;
};

export type RequestScope = {
  next: () => RequestToken;
  abortActive: () => void;
};

export function createRequestScope(): RequestScope {
  let activeController: AbortController | null = null;
  let generation = 0;
  return {
    next: () => {
      activeController?.abort();
      activeController = new AbortController();
      generation += 1;
      const myGeneration = generation;
      return {
        signal: activeController.signal,
        isLatest: () => myGeneration === generation,
      };
    },
    abortActive: () => {
      activeController?.abort();
      activeController = null;
    },
  };
}

export function createRetryableExecution<T>(
  operation: (signal: AbortSignal) => Promise<T>,
  callbacks: RetryStateCallbacks<T>,
  scope: RequestScope = createRequestScope(),
): () => Promise<boolean> {
  return async () => {
    const token = scope.next();
    callbacks.onStart();
    try {
      const value = await operation(token.signal);
      if (token.isLatest()) {
        callbacks.onSuccess(value);
      }
      return true;
    } catch (cause) {
      if (!token.isLatest() || (cause instanceof Error && cause.name === "AbortError")) {
        return false;
      }
      callbacks.onError(cause instanceof Error ? cause.message : callbacks.fallbackError);
      return false;
    } finally {
      if (token.isLatest()) {
        callbacks.onFinally();
      }
    }
  };
}
