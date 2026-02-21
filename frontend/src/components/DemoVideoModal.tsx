"use client";

import { useEffect, useState } from "react";
import DuoButton from "@/components/ui/DuoButton";
import { getApiBase } from "@/lib/api";

interface DemoVideo {
  filename: string;
  size_mb: number;
}

interface DemoVideoModalProps {
  onSelect: (filename: string) => void;
  onClose: () => void;
  filter?: "personal" | "team";
}

const MULTI_PERSON_KEYWORDS = ["match", "team", "spar"];

function isMultiPerson(filename: string): boolean {
  const lower = filename.toLowerCase();
  return MULTI_PERSON_KEYWORDS.some((kw) => lower.includes(kw));
}

function humanizeFilename(filename: string): string {
  return filename
    .replace(/\.mp4$/, "")
    .replace(/[_-]/g, " ")
    .replace(/(\d)/g, " $1")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function DemoVideoModal({ onSelect, onClose, filter }: DemoVideoModalProps) {
  const [videos, setVideos] = useState<DemoVideo[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    fetch(`${getApiBase()}/api/demo-videos`)
      .then((r) => r.json())
      .then((data) => {
        setVideos(data.videos || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const filteredVideos = !filter || showAll
    ? videos
    : videos.filter((v) =>
        filter === "team" ? isMultiPerson(v.filename) : !isMultiPerson(v.filename)
      );

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      onClick={onClose}
    >
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50" />

      {/* Modal */}
      <div
        className="pop-in relative bg-white rounded-[20px] border-2 border-[#E5E5E5] shadow-[0_8px_0_#E5E5E5] w-[90vw] max-w-4xl max-h-[85vh] flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b-2 border-[#E5E5E5]">
          <div>
            <h2 className="text-xl font-extrabold text-[#3C3C3C] m-0">
              Demo Videos
            </h2>
            <p className="text-sm text-[#AFAFAF] mt-0.5">
              {filteredVideos.length} video{filteredVideos.length !== 1 ? "s" : ""} available
            </p>
          </div>
          <div className="flex items-center gap-3">
            {filter && (
              <button
                onClick={() => setShowAll((prev) => !prev)}
                className="text-sm font-bold text-[#1CB0F6] cursor-pointer bg-transparent border-none hover:underline"
              >
                {showAll ? `Show ${filter === "team" ? "team" : "personal"} only` : "Show all"}
              </button>
            )}
            <button
              onClick={onClose}
              className="w-9 h-9 flex items-center justify-center rounded-full bg-[#F7F7F7] border-2 border-[#E5E5E5] text-[#777777] font-bold text-lg cursor-pointer hover:bg-[#E5E5E5] transition-colors"
            >
              &times;
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-5">
          {loading ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
              {Array.from({ length: 8 }).map((_, i) => (
                <div key={i} className="skeleton rounded-[12px]" style={{ aspectRatio: "16/9" }} />
              ))}
            </div>
          ) : filteredVideos.length === 0 ? (
            <p className="text-center text-[#777777] py-8">
              No demo videos found.
            </p>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
              {filteredVideos.map((video) => (
                <button
                  key={video.filename}
                  onClick={() => onSelect(video.filename)}
                  className="group flex flex-col rounded-[12px] border-2 border-[#E5E5E5] bg-white overflow-hidden cursor-pointer shadow-[0_3px_0_#E5E5E5] hover:border-[#1CB0F6] hover:shadow-[0_3px_0_#1899D6] active:shadow-none active:translate-y-[3px] transition-all duration-100 text-left"
                >
                  <div
                    className="w-full bg-[#F7F7F7] overflow-hidden"
                    style={{ aspectRatio: "16/9" }}
                  >
                    <img
                      src={`${getApiBase()}/api/demo-videos/${encodeURIComponent(video.filename)}/thumbnail`}
                      alt={humanizeFilename(video.filename)}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                      loading="lazy"
                    />
                  </div>
                  <div className="px-2.5 py-2">
                    <div className="text-xs font-bold text-[#3C3C3C] truncate">
                      {humanizeFilename(video.filename)}
                    </div>
                    <div className="text-[11px] text-[#AFAFAF]">
                      {video.size_mb} MB
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end px-6 py-4 border-t-2 border-[#E5E5E5]">
          <DuoButton variant="secondary" onClick={onClose}>
            Cancel
          </DuoButton>
        </div>
      </div>
    </div>
  );
}
