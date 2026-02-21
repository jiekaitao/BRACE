"use client";

import { useCallback, useRef, useState } from "react";
import DuoButton from "./ui/DuoButton";

interface VideoUploaderProps {
  onFileSelected: (file: File) => void;
}

export default function VideoUploader({ onFileSelected }: VideoUploaderProps) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (file.type.startsWith("video/")) {
        onFileSelected(file);
      }
    },
    [onFileSelected]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div
      className={`
        flex flex-col items-center justify-center gap-4
        w-full h-full min-h-[300px]
        rounded-[16px] border-2 border-dashed
        transition-colors duration-150
        ${dragging ? "border-[#1CB0F6] bg-[#DDF4FF]" : "border-[#E5E5E5] bg-white"}
      `}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      <svg
        width="48"
        height="48"
        viewBox="0 0 24 24"
        fill="none"
        stroke="#AFAFAF"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="17 8 12 3 7 8" />
        <line x1="12" y1="3" x2="12" y2="15" />
      </svg>

      <p className="text-base text-[#777777] text-center">
        Drag & drop a video file here
      </p>

      <DuoButton
        variant="secondary"
        onClick={() => inputRef.current?.click()}
      >
        Browse Files
      </DuoButton>

      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={handleChange}
      />
    </div>
  );
}
