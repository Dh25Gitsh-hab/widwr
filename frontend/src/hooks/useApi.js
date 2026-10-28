import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Load data for a key. A new key aborts the previous request and fetches again;
 * the previous data stays visible while the next request runs, so charts do not flash.
 * A null key means "nothing to load" (for example, an invalid date).
 */
export function useApi(key, loader) {
  const loaderRef = useRef(loader);
  loaderRef.current = loader;
  const [state, setState] = useState({ data: null, error: null, loading: key !== null });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (key === null) {
      setState({ data: null, error: null, loading: false });
      return undefined;
    }
    const controller = new AbortController();
    setState((previous) => ({ data: previous.data, error: null, loading: true }));
    loaderRef
      .current(controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setState({ data, error: null, loading: false });
      })
      .catch((error) => {
        if (error?.name !== "AbortError" && !controller.signal.aborted) setState({ data: null, error, loading: false });
      });
    return () => controller.abort();
  }, [key, attempt]);

  const retry = useCallback(() => setAttempt((n) => n + 1), []);
  return { ...state, retry };
}
