"use client";

import { ButtonHTMLAttributes } from "react";

type Variant = "primary" | "secondary" | "danger" | "blue";

const variantStyles: Record<Variant, string> = {
  primary:
    "bg-[#58CC02] text-white shadow-[0_4px_0_#46A302] hover:brightness-105 active:shadow-none active:translate-y-[4px]",
  secondary:
    "bg-white text-[#4B4B4B] border-2 border-[#E5E5E5] shadow-[0_4px_0_#E5E5E5] hover:bg-[#F7F7F7] active:shadow-none active:translate-y-[4px] active:border-[#CDCDCD]",
  danger:
    "bg-[#EA2B2B] text-white shadow-[0_4px_0_#CC2424] hover:brightness-105 active:shadow-none active:translate-y-[4px]",
  blue:
    "bg-[#1CB0F6] text-white shadow-[0_4px_0_#1899D6] hover:brightness-105 active:shadow-none active:translate-y-[4px]",
};

interface DuoButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  fullWidth?: boolean;
}

export default function DuoButton({
  variant = "primary",
  fullWidth = false,
  className = "",
  disabled,
  children,
  ...props
}: DuoButtonProps) {
  return (
    <button
      className={`
        inline-flex items-center justify-center
        px-6 py-3.5
        text-base font-bold uppercase tracking-[0.05em]
        rounded-[12px] border-none cursor-pointer
        select-none
        transition-all duration-100 ease-linear
        min-h-[44px]
        ${fullWidth ? "w-full" : ""}
        ${
          disabled
            ? "bg-[#E5E5E5] shadow-[0_4px_0_#CDCDCD] text-[#AFAFAF] cursor-not-allowed"
            : variantStyles[variant]
        }
        ${className}
      `.trim()}
      disabled={disabled}
      {...props}
    >
      {children}
    </button>
  );
}
