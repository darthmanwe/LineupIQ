"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";

/**
 * A searchable player field, backed by `GET /api/players`.
 *
 * It replaced a native `<select>` carrying 240 options. That worked, and it
 * stopped working as soon as there were two lineups on screen: eleven
 * dropdowns of 240 names each is a form nobody can fill in, and the lineup you
 * actually want to build is a specific five whose names you already know.
 *
 * **The endpoint it queries was already live and used by nothing.**
 * `/api/players?q=` has been serving a typeahead since the route registry was
 * written; the UI simply never called it. Preferring it to the committed JSON
 * also means the picker searches the same roster the scorer scores, rather than
 * a build-time snapshot that can fall behind it.
 *
 * Three things that are deliberate rather than incidental:
 *
 * 1. **It degrades to the committed list.** A static export opened without the
 *    Worker behind it still filters, over the `players.json` the page already
 *    imports. A picker that goes blank when the network does is worse than a
 *    slightly stale one.
 * 2. **Attempts are shown next to every name.** This is the number that decides
 *    whether the API will answer at all, so putting it in the picker means the
 *    refusal is predictable before it happens rather than surprising after.
 * 3. **It is a combobox, not a text box with a menu underneath.** Arrow keys,
 *    Enter, Escape, and `aria-activedescendant`, because a control that can
 *    only be operated with a mouse is not finished.
 *
 * `restrictTo` exists because the first version did not have it and was wrong.
 * The shooter field is a picker over *the five on the floor*, and without it the
 * component happily searched the whole league and let you select a shooter who
 * was not in the lineup -- which the API then rejects with a 400. A control that
 * offers an option the server will refuse is a broken control, not a strict
 * server.
 */

export type PickablePlayer = { id: number; name: string; attempts: number };

type ApiPlayer = { player_id: string; name?: string | null; attempts?: number | null };

/** How long to wait after a keystroke before asking the API. */
const DEBOUNCE_MS = 140;

/** Rows shown at once. Enough to recognise a name, few enough to scan. */
const LIMIT = 8;

export function PlayerPicker({
  label,
  value,
  onChange,
  fallback,
  exclude = [],
  disabled = false,
  restrictTo = false,
}: {
  label: string;
  value: number;
  onChange: (id: number) => void;
  /** The committed roster, used when the API cannot be reached. */
  fallback: PickablePlayer[];
  /** Ids already used elsewhere in the same lineup. Shown, but not choosable. */
  exclude?: number[];
  disabled?: boolean;
  /**
   * Search only `fallback`, never the league.
   *
   * For the shooter field, where `fallback` *is* the set of legal answers: the
   * five on the floor. Without this the picker searched all 766 players and
   * would offer one the scoring endpoint rejects with a 400.
   */
  restrictTo?: boolean;
}) {
  const listId = useId();
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [results, setResults] = useState<PickablePlayer[]>([]);
  const [offline, setOffline] = useState(false);
  const boxRef = useRef<HTMLDivElement | null>(null);

  const byId = useMemo(() => new Map(fallback.map((p) => [p.id, p])), [fallback]);
  const selected = byId.get(value);
  const excluded = useMemo(() => new Set(exclude.filter((id) => id !== value)), [exclude, value]);

  const localMatches = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const pool =
      needle === "" ? fallback : fallback.filter((p) => p.name.toLowerCase().includes(needle));
    return pool.slice(0, LIMIT);
  }, [query, fallback]);

  useEffect(() => {
    if (!open || restrictTo) return;
    let cancelled = false;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      const url = `/api/players?q=${encodeURIComponent(query.trim())}&limit=${LIMIT}`;
      fetch(url, { signal: controller.signal })
        .then((response) => (response.ok ? response.json() : Promise.reject(new Error("no"))))
        .then((body: unknown) => {
          if (cancelled) return;
          const rows = (body as { data?: { players?: ApiPlayer[] } }).data?.players ?? [];
          setResults(
            rows.map((row) => ({
              id: Number(row.player_id),
              name: row.name ?? String(row.player_id),
              attempts: row.attempts ?? 0,
            }))
          );
          setOffline(false);
        })
        .catch(() => {
          // Including an aborted request, which is the common case while typing.
          if (!cancelled) setOffline(true);
        });
    }, DEBOUNCE_MS);
    return () => {
      cancelled = true;
      controller.abort();
      clearTimeout(timer);
    };
  }, [query, open, restrictTo]);

  const options = restrictTo || offline || results.length === 0 ? localMatches : results;

  useEffect(() => {
    if (active >= options.length) setActive(0);
  }, [options.length, active]);

  // Close on a click outside. A picker that stays open behind the next one is
  // how two of these end up fighting for the same keystrokes.
  useEffect(() => {
    if (!open) return;
    const onDown = (event: MouseEvent): void => {
      if (boxRef.current && !boxRef.current.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  function choose(player: PickablePlayer): void {
    if (excluded.has(player.id)) return;
    onChange(player.id);
    setQuery("");
    setOpen(false);
  }

  function onKeyDown(event: React.KeyboardEvent<HTMLInputElement>): void {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setOpen(true);
      setActive((i) => Math.min(i + 1, options.length - 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActive((i) => Math.max(i - 1, 0));
    } else if (event.key === "Enter") {
      const player = options[active];
      if (open && player) {
        event.preventDefault();
        choose(player);
      }
    } else if (event.key === "Escape") {
      setOpen(false);
      setQuery("");
    }
  }

  return (
    <div className="picker" ref={boxRef}>
      <style>{`
        .picker { position: relative; margin-bottom: 0.5rem; }
        .picker__label { display: block; font-size: 0.72rem; color: var(--muted); margin-bottom: 0.15rem; }
        .picker__field { display: flex; align-items: baseline; gap: 0.4rem; border: 1px solid var(--border); border-radius: 6px; background: var(--bg); padding: 0.3rem 0.45rem; }
        .picker__field:focus-within { border-color: var(--accent); }
        .picker__field[data-disabled="true"] { opacity: 0.55; }
        .picker input { flex: 1; min-width: 0; border: 0; background: transparent; color: var(--text); font-size: 0.85rem; padding: 0.05rem 0; }
        .picker input:focus { outline: none; }
        .picker__count { font-size: 0.7rem; color: var(--muted); font-family: var(--mono, monospace); white-space: nowrap; }
        .picker__list { position: absolute; z-index: 20; left: 0; right: 0; top: calc(100% + 2px); margin: 0; padding: 0.2rem; list-style: none; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; box-shadow: 0 6px 20px rgb(0 0 0 / 0.12); max-height: 15rem; overflow-y: auto; }
        .picker__option { display: flex; justify-content: space-between; gap: 0.6rem; padding: 0.3rem 0.45rem; border-radius: 4px; font-size: 0.85rem; cursor: pointer; }
        .picker__option[aria-selected="true"] { background: var(--accent-soft); color: var(--accent); }
        .picker__option[data-excluded="true"] { opacity: 0.45; cursor: not-allowed; }
        .picker__option span:last-child { font-family: var(--mono, monospace); font-size: 0.72rem; color: var(--muted); }
        .picker__empty { padding: 0.4rem 0.45rem; font-size: 0.82rem; color: var(--muted); }
      `}</style>

      <label className="picker__label" htmlFor={`${listId}-input`}>
        {label}
      </label>
      <div className="picker__field" data-disabled={disabled}>
        <input
          id={`${listId}-input`}
          type="text"
          role="combobox"
          autoComplete="off"
          disabled={disabled}
          aria-expanded={open}
          aria-controls={listId}
          aria-autocomplete="list"
          aria-activedescendant={
            open && options[active] ? `${listId}-${options[active].id}` : undefined
          }
          placeholder={selected?.name ?? "Search a player"}
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={onKeyDown}
        />
        {/* The number that decides whether the API will answer, shown before it
            has to refuse. */}
        <span className="picker__count">
          {selected ? `${selected.attempts.toLocaleString()} att` : "—"}
        </span>
      </div>

      {open && (
        <ul className="picker__list" id={listId} role="listbox" aria-label={label}>
          {options.length === 0 && <li className="picker__empty">No player matches that.</li>}
          {options.map((player, index) => {
            const isExcluded = excluded.has(player.id);
            return (
              // No key handler on the option itself: this is the
              // `aria-activedescendant` combobox pattern, where the input keeps
              // focus and owns every keystroke. A duplicate handler here would
              // be unreachable.
              <li
                key={player.id}
                id={`${listId}-${player.id}`}
                role="option"
                aria-selected={index === active}
                data-excluded={isExcluded}
                className="picker__option"
                onMouseEnter={() => setActive(index)}
                onMouseDown={(e) => {
                  e.preventDefault();
                  choose(player);
                }}
              >
                <span>
                  {player.name}
                  {isExcluded ? " · already on the floor" : ""}
                </span>
                <span>{player.attempts.toLocaleString()}</span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
