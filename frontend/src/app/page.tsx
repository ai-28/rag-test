import { RagChat } from "@/components/rag-chat";

export default function Home() {
  return (
    <div className="flex min-h-full flex-1 flex-col bg-zinc-50 dark:bg-zinc-950">
      <RagChat />
    </div>
  );
}
