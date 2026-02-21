"use client";

interface VideoButtonProps {
  title: string;
  thumbnail: string;
  onClick?: () => void;
}

export default function VideoButton({ title, thumbnail, onClick }: VideoButtonProps) {
  return (
    <button
      onClick={onClick}
      className="
        group relative flex items-center w-full h-20 overflow-hidden
        rounded-[12px] border-2 border-[#E5E5E5] bg-white
        shadow-[0_4px_0_#E5E5E5]
        hover:border-[#1CB0F6] hover:shadow-[0_4px_0_#1899D6]
        active:shadow-none active:translate-y-[4px]
        transition-all duration-100 cursor-pointer text-left
      "
    >
      {/* Title */}
      <div className="flex-1 px-4 min-w-0">
        <span className="font-bold text-[#3C3C3C] text-sm">
          {title}
        </span>
      </div>

      {/* Thumbnail with gradient mask */}
      <div className="w-1/4 h-full flex-shrink-0">
        <img
          src={thumbnail}
          alt={title}
          className="w-full h-full object-cover"
          style={{
            maskImage: "linear-gradient(to right, transparent, black 30%)",
            WebkitMaskImage: "linear-gradient(to right, transparent, black 30%)",
          }}
          loading="lazy"
        />
      </div>
    </button>
  );
}
