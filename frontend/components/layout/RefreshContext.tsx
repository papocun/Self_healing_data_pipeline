"use client";
import { createContext, useCallback, useContext, useState } from "react";

interface Ctx {
  tick: number;
  refresh: () => void;
  lastUpdated: Date | null;
  markUpdated: () => void;
}
const RefreshCtx = createContext<Ctx>({ tick: 0, refresh: () => {}, lastUpdated: null, markUpdated: () => {} });

export function RefreshProvider({ children }: { children: React.ReactNode }) {
  const [tick, setTick] = useState(0);
  const [lastUpdated, setLast] = useState<Date | null>(null);
  const refresh = useCallback(() => setTick((t) => t + 1), []);
  const markUpdated = useCallback(() => setLast(new Date()), []);
  return <RefreshCtx.Provider value={{ tick, refresh, lastUpdated, markUpdated }}>{children}</RefreshCtx.Provider>;
}

export const useRefresh = () => useContext(RefreshCtx);
