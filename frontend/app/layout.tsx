import type { Metadata, Viewport } from "next";
import { Barlow, Barlow_Semi_Condensed } from "next/font/google";
import Link from "next/link";
import "./globals.css";

// Barlow (body) and its semi-condensed cut (headings, numbers) echo stadium
// timing boards without resorting to a novelty "sporty" face.
const barlow = Barlow({ variable: "--font-barlow", subsets: ["latin"], weight: ["400", "500", "600"] });
const barlowCondensed = Barlow_Semi_Condensed({
  variable: "--font-barlow-condensed",
  subsets: ["latin"],
  weight: ["500", "600", "700"],
});

export const metadata: Metadata = {
  title: "UniAthleteIQ",
  description: "Training-load injury risk for running teams",
  manifest: "/manifest.webmanifest",
};

export const viewport: Viewport = {
  themeColor: "#2f6f5e",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${barlow.variable} ${barlowCondensed.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col">
        <header className="border-b border-line bg-surface">
          <nav className="mx-auto flex max-w-5xl items-center gap-6 px-4 py-3">
            <Link href="/" className="font-display text-xl font-bold tracking-tight text-pine">
              UniAthleteIQ
            </Link>
            <Link href="/" className="text-ink-soft hover:text-ink">Team</Link>
            <Link href="/about" className="text-ink-soft hover:text-ink">How it works</Link>
          </nav>
        </header>
        <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-8">{children}</main>
        <footer className="border-t border-line px-4 py-4 text-center text-sm text-ink-soft">
          Decision support, not medical advice. Most flagged days are not followed by an injury.
        </footer>
      </body>
    </html>
  );
}
