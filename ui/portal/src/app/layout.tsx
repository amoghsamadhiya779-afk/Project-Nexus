import type { Metadata } from "next";
import { Inter, Fira_Code } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-geist-sans",
});

const firaCode = Fira_Code({
  subsets: ["latin"],
  variable: "--font-geist-mono",
});

export const metadata: Metadata = {
  title: "NEXUS OS | Deep-Space Intelligence Portal",
  description: "A production-grade Personalized Marketplace Intelligence Platform and Real-time MLOps serving plane.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${firaCode.variable}`}>
      <body className="antialiased bg-background text-primaryText overflow-hidden select-none">
        {children}
      </body>
    </html>
  );
}
