import type { Metadata } from "next";
import "./globals.css";
import TopNav from "@/components/TopNav";
import Providers from "@/components/Providers";

export const metadata: Metadata = {
  title: "Forge — LLM Fine-Tuning",
  description: "LLM Fine-Tuning Platform",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, padding: 0, display: "flex", flexDirection: "column", minHeight: "100vh" }}>
        <Providers>
          <TopNav />
          <main style={{ flex: 1, overflow: "hidden" }}>{children}</main>
        </Providers>
      </body>
    </html>
  );
}
