"use client";

import { HTMLAttributes, useState } from "react";

interface PathCardProps extends HTMLAttributes<HTMLDivElement> {
  image?: string;
  title: string;
  subtitle: string;
  onClick?: () => void;
}

export default function PathCard({
  image,
  title,
  subtitle,
  onClick,
  className = "",
  ...props
}: PathCardProps) {
  const [imageLoaded, setImageLoaded] = useState(false);

  return (
    <div
      onClick={onClick}
      className={`
        group relative overflow-hidden rounded-[20px] border-2 border-[#E5E5E5]
        cursor-pointer select-none
        shadow-[0_6px_0_#E5E5E5]
        transition-all duration-400 ease-out
        hover:shadow-[0_8px_0_#1899D6] hover:border-[#1CB0F6]
        hover:scale-[1.02]
        active:shadow-none active:translate-y-[6px]
        ${className}
      `.trim()}
      {...props}
    >
      {/* Background image with fade-in on load + grayscale -> color on hover */}
      {image && (
        <img
          src={image}
          alt=""
          onLoad={() => setImageLoaded(true)}
          className={`
            absolute inset-0 w-full h-full object-cover scale-[1.02]
            transition-all duration-600 ease-out
            grayscale brightness-80 blur-[0.5px] group-hover:grayscale-0
            ${imageLoaded ? "opacity-100" : "opacity-0"}
          `}
        />
      )}

      {/* Dark gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/30 to-transparent" />

      {/* Text content */}
      <div className="relative flex flex-col justify-end p-6" style={{ minHeight: 260 }}>
        <h2 className="text-2xl font-extrabold text-white leading-tight mb-1">
          {title}
        </h2>
        <p className="text-sm text-white/80">
          {subtitle}
        </p>
      </div>
    </div>
  );
}
