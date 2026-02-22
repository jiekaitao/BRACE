"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import DuoButton from "@/components/ui/DuoButton";
import {
  fetchVectorStats,
  fetchVectorEntries,
  searchSimilarVectors,
} from "@/lib/dashboard";
import type {
  VectorStats,
  VectorEntry,
  CollectionStats,
} from "@/lib/types";

const COLLECTIONS = [
  "person_embeddings",
  "motion_segments",
  "activity_templates",
] as const;
type CollectionName = (typeof COLLECTIONS)[number];

const COLLECTION_LABELS: Record<CollectionName, string> = {
  person_embeddings: "Person Embeddings",
  motion_segments: "Motion Segments",
  activity_templates: "Activity Templates",
};

const PAGE_SIZE = 20;

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

function formatTimestamp(ts: number): string {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString();
}

// --- Stats Card ---
function StatsCard({
  name,
  stats,
}: {
  name: string;
  stats: CollectionStats | undefined;
}) {
  return (
    <div className="bg-white rounded-[16px] border-2 border-[#E5E5E5] p-5">
      <p className="text-xs font-bold text-[#AFAFAF] uppercase tracking-wider mb-1">
        {name}
      </p>
      {stats?.error ? (
        <p className="text-sm text-[#FF4B4B]">{stats.error}</p>
      ) : (
        <>
          <p className="text-2xl font-extrabold text-[#3C3C3C]">
            {stats?.count?.toLocaleString() ?? "—"}
          </p>
          <div className="flex gap-4 mt-2 text-xs text-[#777777]">
            <span>Storage: {formatBytes(stats?.storage_bytes ?? 0)}</span>
            <span>Index: {formatBytes(stats?.index_memory_bytes ?? 0)}</span>
          </div>
          <div className="flex gap-4 mt-1 text-xs text-[#AFAFAF]">
            <span>Indexed: {stats?.indexed_vectors?.toLocaleString() ?? 0}</span>
            <span>Deleted: {stats?.deleted_vectors ?? 0}</span>
          </div>
        </>
      )}
    </div>
  );
}

// --- Entry Row ---
function EntryRow({
  entry,
  collection,
  isSelected,
  onClick,
}: {
  entry: VectorEntry;
  collection: CollectionName;
  isSelected: boolean;
  onClick: () => void;
}) {
  return (
    <tr
      className={`cursor-pointer transition-colors ${
        isSelected
          ? "bg-[#E8F5FE] border-l-4 border-l-[#4FC3F7]"
          : "hover:bg-[#FAFAFA]"
      }`}
      onClick={onClick}
    >
      {collection === "person_embeddings" && (
        <>
          <td className="px-3 py-2">
            {entry.person_crop_b64 ? (
              <img
                src={`data:image/jpeg;base64,${entry.person_crop_b64}`}
                alt="crop"
                className="w-8 h-16 object-cover rounded"
              />
            ) : (
              <div className="w-8 h-16 bg-[#F0F0F0] rounded flex items-center justify-center text-xs text-[#AFAFAF]">
                —
              </div>
            )}
          </td>
          <td className="px-3 py-2 text-sm text-[#3C3C3C] font-mono">
            {entry.person_id ?? "—"}
          </td>
          <td className="px-3 py-2 text-sm text-[#777777] font-mono">
            {entry.session_id?.slice(0, 8) ?? "—"}
          </td>
        </>
      )}
      {collection === "motion_segments" && (
        <>
          <td className="px-3 py-2 text-sm text-[#3C3C3C]">
            {entry.activity_label ?? "—"}
          </td>
          <td className="px-3 py-2 text-sm text-[#3C3C3C] font-mono">
            {entry.person_id ?? "—"}
          </td>
          <td className="px-3 py-2 text-sm text-[#777777] font-mono">
            {entry.session_id?.slice(0, 8) ?? "—"}
          </td>
          <td className="px-3 py-2 text-sm text-[#777777]">
            {typeof entry.metadata?.risk_score === "number"
              ? (entry.metadata.risk_score as number).toFixed(2)
              : "—"}
          </td>
        </>
      )}
      {collection === "activity_templates" && (
        <>
          <td className="px-3 py-2 text-sm text-[#3C3C3C]">
            {(entry.metadata?.activity_name as string) ?? "—"}
          </td>
          <td className="px-3 py-2 text-sm text-[#777777]">
            {(entry.metadata?.source as string) ?? "—"}
          </td>
        </>
      )}
      <td className="px-3 py-2 text-xs text-[#AFAFAF]">
        {formatTimestamp(entry.timestamp)}
      </td>
    </tr>
  );
}

// --- Detail Panel ---
function DetailPanel({
  entry,
  collection,
  onFindSimilar,
  similarResults,
  searchingSimlar,
}: {
  entry: VectorEntry;
  collection: CollectionName;
  onFindSimilar: () => void;
  similarResults: Array<{ uuid: string; score: number; metadata: Record<string, unknown> }> | null;
  searchingSimlar: boolean;
}) {
  return (
    <div className="bg-white rounded-[16px] border-2 border-[#E5E5E5] p-5 mt-4">
      <div className="flex items-start gap-4">
        {/* Crop thumbnail */}
        {entry.person_crop_b64 && (
          <img
            src={`data:image/jpeg;base64,${entry.person_crop_b64}`}
            alt="Person crop"
            className="w-16 h-32 object-cover rounded-lg border border-[#E5E5E5]"
          />
        )}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-bold text-[#3C3C3C] mb-2">
            UUID: <span className="font-mono font-normal">{entry.vector_uuid}</span>
          </p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-[#777777]">
            <span>Collection: {entry.collection}</span>
            <span>Timestamp: {formatTimestamp(entry.timestamp)}</span>
            {entry.person_id && <span>Person: {entry.person_id}</span>}
            {entry.session_id && <span>Session: {entry.session_id}</span>}
            {entry.activity_label && <span>Activity: {entry.activity_label}</span>}
          </div>
          {/* Metadata */}
          <details className="mt-2">
            <summary className="text-xs text-[#AFAFAF] cursor-pointer">
              Raw metadata
            </summary>
            <pre className="mt-1 text-xs text-[#777777] bg-[#F7F7F7] p-2 rounded overflow-auto max-h-40">
              {JSON.stringify(entry.metadata, null, 2)}
            </pre>
          </details>
        </div>
      </div>

      {/* Find Similar */}
      <div className="mt-4">
        <DuoButton
          variant="blue"
          onClick={onFindSimilar}
          disabled={searchingSimlar}
        >
          {searchingSimlar ? "Searching..." : "Find Similar"}
        </DuoButton>
        {similarResults && (
          <div className="mt-3">
            <p className="text-xs font-bold text-[#3C3C3C] mb-1">
              Similar Vectors ({similarResults.length})
            </p>
            {similarResults.length === 0 ? (
              <p className="text-xs text-[#AFAFAF]">No similar vectors found.</p>
            ) : (
              <div className="space-y-1">
                {similarResults.map((r, i) => (
                  <div
                    key={r.uuid || i}
                    className="flex items-center gap-3 text-xs text-[#777777] bg-[#F7F7F7] px-3 py-2 rounded"
                  >
                    <span className="font-mono">{r.uuid?.slice(0, 8)}</span>
                    <span>Score: {r.score.toFixed(3)}</span>
                    {r.metadata?.activity_label ? (
                      <span className="text-[#3C3C3C]">
                        {String(r.metadata.activity_label)}
                      </span>
                    ) : null}
                    {r.metadata?.person_id ? (
                      <span>Person: {String(r.metadata.person_id)}</span>
                    ) : null}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// --- Main Page ---
export default function VectorAIPage() {
  const [stats, setStats] = useState<VectorStats | null>(null);
  const [activeCollection, setActiveCollection] =
    useState<CollectionName>("person_embeddings");
  const [entries, setEntries] = useState<VectorEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);

  // Filters
  const [filterPersonId, setFilterPersonId] = useState("");
  const [filterSessionId, setFilterSessionId] = useState("");
  const [filterActivity, setFilterActivity] = useState("");

  // Detail
  const [selectedUuid, setSelectedUuid] = useState<string | null>(null);
  const [similarResults, setSimilarResults] = useState<
    Array<{ uuid: string; score: number; metadata: Record<string, unknown> }> | null
  >(null);
  const [searchingSimilar, setSearchingSimilar] = useState(false);

  // Fetch stats on mount
  useEffect(() => {
    fetchVectorStats().then(setStats).catch(() => {});
  }, []);

  // Fetch entries when collection/offset/filters change
  const loadEntries = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetchVectorEntries(activeCollection, {
        limit: PAGE_SIZE,
        offset,
        person_id: filterPersonId || undefined,
        session_id: filterSessionId || undefined,
        activity_label: filterActivity || undefined,
      });
      setEntries(res.entries ?? []);
      setTotal(res.total ?? 0);
    } catch {
      setEntries([]);
      setTotal(0);
    }
    setLoading(false);
  }, [activeCollection, offset, filterPersonId, filterSessionId, filterActivity]);

  useEffect(() => {
    loadEntries();
  }, [loadEntries]);

  // Reset offset + selection when switching collections or filters
  useEffect(() => {
    setOffset(0);
    setSelectedUuid(null);
    setSimilarResults(null);
  }, [activeCollection, filterPersonId, filterSessionId, filterActivity]);

  const selectedEntry = entries.find((e) => e.vector_uuid === selectedUuid);

  const handleFindSimilar = async () => {
    if (!selectedUuid) return;
    setSearchingSimilar(true);
    try {
      const res = await searchSimilarVectors(selectedUuid, activeCollection, 5);
      setSimilarResults(res.results ?? []);
    } catch {
      setSimilarResults([]);
    }
    setSearchingSimilar(false);
  };

  const pageStart = offset + 1;
  const pageEnd = Math.min(offset + PAGE_SIZE, total);

  return (
    <div className="min-h-screen flex flex-col items-center px-5 py-8">
      {/* Header */}
      <div className="w-full max-w-4xl mb-6">
        <div className="flex items-center gap-2 text-sm text-[#AFAFAF] mb-2">
          <Link href="/" className="hover:text-[#777777] transition-colors">
            Home
          </Link>
          <span>/</span>
          <span className="text-[#3C3C3C]">VectorAI Store</span>
        </div>
        <h1 className="text-2xl font-extrabold text-[#3C3C3C]">
          VectorAI Store
        </h1>
        <p className="text-sm text-[#777777] mt-1">
          Browse stored person embeddings, motion segments, and activity
          templates
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 w-full max-w-4xl mb-6">
        {COLLECTIONS.map((col) => (
          <StatsCard
            key={col}
            name={COLLECTION_LABELS[col]}
            stats={stats?.[col]}
          />
        ))}
      </div>

      {/* Collection Tabs */}
      <div className="flex gap-2 w-full max-w-4xl mb-4">
        {COLLECTIONS.map((col) => (
          <button
            key={col}
            onClick={() => setActiveCollection(col)}
            className={`px-4 py-2 text-sm font-bold rounded-[10px] transition-colors border-2 ${
              activeCollection === col
                ? "bg-[#1CB0F6] text-white border-[#1899D6]"
                : "bg-white text-[#777777] border-[#E5E5E5] hover:border-[#4FC3F7]"
            }`}
          >
            {COLLECTION_LABELS[col]}
          </button>
        ))}
      </div>

      {/* Filter Bar */}
      <div className="flex flex-wrap gap-3 w-full max-w-4xl mb-4">
        <input
          type="text"
          placeholder="Person ID"
          value={filterPersonId}
          onChange={(e) => setFilterPersonId(e.target.value)}
          className="px-3 py-2 text-sm border-2 border-[#E5E5E5] rounded-[10px] outline-none focus:border-[#4FC3F7] transition-colors w-36"
        />
        <input
          type="text"
          placeholder="Session ID"
          value={filterSessionId}
          onChange={(e) => setFilterSessionId(e.target.value)}
          className="px-3 py-2 text-sm border-2 border-[#E5E5E5] rounded-[10px] outline-none focus:border-[#4FC3F7] transition-colors w-36"
        />
        {activeCollection === "motion_segments" && (
          <input
            type="text"
            placeholder="Activity label"
            value={filterActivity}
            onChange={(e) => setFilterActivity(e.target.value)}
            className="px-3 py-2 text-sm border-2 border-[#E5E5E5] rounded-[10px] outline-none focus:border-[#4FC3F7] transition-colors w-40"
          />
        )}
        <button
          onClick={loadEntries}
          className="px-4 py-2 text-sm font-bold text-[#1CB0F6] border-2 border-[#1CB0F6] rounded-[10px] hover:bg-[#E8F5FE] transition-colors"
        >
          Refresh
        </button>
      </div>

      {/* Entries Table */}
      <div className="w-full max-w-4xl bg-white rounded-[16px] border-2 border-[#E5E5E5] overflow-hidden">
        {loading ? (
          <div className="p-8 text-center text-[#AFAFAF]">Loading...</div>
        ) : entries.length === 0 ? (
          <div className="p-8 text-center text-[#AFAFAF]">
            No entries found. Run an analysis session to populate VectorAI.
          </div>
        ) : (
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-[#E5E5E5] bg-[#FAFAFA]">
                {activeCollection === "person_embeddings" && (
                  <>
                    <th className="px-3 py-2 text-xs font-bold text-[#AFAFAF] uppercase">
                      Crop
                    </th>
                    <th className="px-3 py-2 text-xs font-bold text-[#AFAFAF] uppercase">
                      Person
                    </th>
                    <th className="px-3 py-2 text-xs font-bold text-[#AFAFAF] uppercase">
                      Session
                    </th>
                  </>
                )}
                {activeCollection === "motion_segments" && (
                  <>
                    <th className="px-3 py-2 text-xs font-bold text-[#AFAFAF] uppercase">
                      Activity
                    </th>
                    <th className="px-3 py-2 text-xs font-bold text-[#AFAFAF] uppercase">
                      Person
                    </th>
                    <th className="px-3 py-2 text-xs font-bold text-[#AFAFAF] uppercase">
                      Session
                    </th>
                    <th className="px-3 py-2 text-xs font-bold text-[#AFAFAF] uppercase">
                      Risk
                    </th>
                  </>
                )}
                {activeCollection === "activity_templates" && (
                  <>
                    <th className="px-3 py-2 text-xs font-bold text-[#AFAFAF] uppercase">
                      Activity
                    </th>
                    <th className="px-3 py-2 text-xs font-bold text-[#AFAFAF] uppercase">
                      Source
                    </th>
                  </>
                )}
                <th className="px-3 py-2 text-xs font-bold text-[#AFAFAF] uppercase">
                  Time
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#F0F0F0]">
              {entries.map((entry) => (
                <EntryRow
                  key={entry.vector_uuid}
                  entry={entry}
                  collection={activeCollection}
                  isSelected={entry.vector_uuid === selectedUuid}
                  onClick={() => {
                    setSelectedUuid(
                      entry.vector_uuid === selectedUuid
                        ? null
                        : entry.vector_uuid
                    );
                    setSimilarResults(null);
                  }}
                />
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Pagination */}
      {total > 0 && (
        <div className="flex items-center gap-4 mt-4 w-full max-w-4xl justify-between">
          <DuoButton
            variant="secondary"
            disabled={offset === 0}
            onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
          >
            Previous
          </DuoButton>
          <span className="text-sm text-[#777777]">
            {pageStart}–{pageEnd} of {total.toLocaleString()}
          </span>
          <DuoButton
            variant="secondary"
            disabled={offset + PAGE_SIZE >= total}
            onClick={() => setOffset(offset + PAGE_SIZE)}
          >
            Next
          </DuoButton>
        </div>
      )}

      {/* Detail Panel */}
      {selectedEntry && (
        <div className="w-full max-w-4xl">
          <DetailPanel
            entry={selectedEntry}
            collection={activeCollection}
            onFindSimilar={handleFindSimilar}
            similarResults={similarResults}
            searchingSimlar={searchingSimilar}
          />
        </div>
      )}
    </div>
  );
}
