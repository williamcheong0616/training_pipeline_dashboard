"use client";

export default function Tooltip({ text }: { text: string }) {
  return (
    <span className="lf-tt-wrap" aria-label={text}>
      <span className="lf-tt-icon">?</span>
      <span className="lf-tt-box">{text}</span>
    </span>
  );
}
