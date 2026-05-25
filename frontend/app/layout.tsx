import type { Metadata } from "next";
import "./globals.css";
import TopNav from "@/components/TopNav";
import Providers from "@/components/Providers";
import { ThemeProvider } from "@/components/ThemeProvider";

export const metadata: Metadata = {
  title: "RojBot — LLM Fine-Tuning",
  description: "LLM Fine-Tuning Platform",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body style={{ margin: 0, padding: 0, display: "flex", flexDirection: "column", minHeight: "100vh" }}>
        <ThemeProvider>
          <Providers>
            <TopNav />
            <main style={{ flex: 1 }}>{children}</main>
          </Providers>
        </ThemeProvider>
      </body>
    </html>
  );
}
