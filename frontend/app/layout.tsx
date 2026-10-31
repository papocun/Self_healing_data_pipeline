import type { Metadata } from "next";
import "./globals.css";
import { Shell } from "@/components/layout/Shell";

export const metadata: Metadata = {
  title: "SelfHealingPipeline Dashboard",
  description: "Sentiment analytics and pipeline observability",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" data-theme="light">
      <body>
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
