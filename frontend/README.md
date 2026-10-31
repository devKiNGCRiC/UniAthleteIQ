# UniAthleteIQ web app

Next.js (App Router, TypeScript, Tailwind) frontend for the UniAthleteIQ API.

```powershell
copy .env.example .env.local   # NEXT_PUBLIC_API_URL, default http://localhost:8000
npm install
npm run dev                    # http://localhost:3000
```

The API must be running (`uvicorn api.main:app` from the project root). See the main [README](../README.md).

| Path | Screen |
|---|---|
| `app/page.tsx` | Team view ranked by today's risk |
| `app/athletes/[id]/page.tsx` | Athlete risk, reasons, recent training |
| `app/athletes/[id]/log/page.tsx` | Daily training log (phone-friendly) |
| `app/about/page.tsx` | How the model works, with live accuracy figures |
| `lib/api.ts` | Typed API client (mirrors `api/schemas.py`) |
| `components/RiskTrack.tsx` | The "lane" risk track (log scale, 1× = average day) |
