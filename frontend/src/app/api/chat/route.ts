import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(req: Request) {
  const base = process.env.RAG_API_BASE_URL?.replace(/\/$/, "");
  if (!base) {
    return NextResponse.json(
      { error: "Missing RAG_API_BASE_URL (see frontend/env.example)" },
      { status: 500 },
    );
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const upstream = await fetch(`${base}/api/rag/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!upstream.ok) {
    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: { "Content-Type": "text/plain; charset=utf-8" },
    });
  }

  return new Response(upstream.body, {
    status: 200,
    headers: {
      "Content-Type":
        upstream.headers.get("content-type") ||
        "application/x-ndjson; charset=utf-8",
      "Cache-Control": "no-store",
    },
  });
}
