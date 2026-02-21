"use client";

import { HTMLAttributes } from "react";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  interactive?: boolean;
}

export default function Card({
  interactive = false,
  className = "",
  children,
  ...props
}: CardProps) {
  return (
    <div
      className={`
        bg-white rounded-[16px] border-2 border-[#E5E5E5] p-5
        ${
          interactive
            ? "cursor-pointer shadow-[0_4px_0_#E5E5E5] transition-all duration-100 ease-linear active:shadow-none active:translate-y-[4px]"
            : ""
        }
        ${className}
      `.trim()}
      {...props}
    >
      {children}
    </div>
  );
}
