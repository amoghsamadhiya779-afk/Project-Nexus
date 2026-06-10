"use client";

import React, { useState, useRef, useEffect } from "react";
import { Terminal, Send, Cpu, HelpCircle, Volume2, VolumeX } from "lucide-react";

// --- WEB AUDIO API SCI-FI SYNTH HELPER ---
class AudioSynth {
  private ctx: AudioContext | null = null;
  private humOsc: OscillatorNode | null = null;
  private humGain: GainNode | null = null;
  public isMuted: boolean = true;

  private initCtx() {
    if (!this.ctx) {
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      if (AudioContextClass) {
        this.ctx = new AudioContextClass();
      }
    }
  }

  public playClick() {
    if (this.isMuted) return;
    this.initCtx();
    if (!this.ctx) return;

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = "sine";
    osc.frequency.setValueAtTime(1200, this.ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(150, this.ctx.currentTime + 0.08);

    gain.gain.setValueAtTime(0.04, this.ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime + 0.08);

    osc.connect(gain);
    gain.connect(this.ctx.destination);

    osc.start();
    osc.stop(this.ctx.currentTime + 0.09);
  }

  public playType() {
    if (this.isMuted) return;
    this.initCtx();
    if (!this.ctx) return;

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = "triangle";
    osc.frequency.setValueAtTime(800 + Math.random() * 400, this.ctx.currentTime);

    gain.gain.setValueAtTime(0.012, this.ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime + 0.03);

    osc.connect(gain);
    gain.connect(this.ctx.destination);

    osc.start();
    osc.stop(this.ctx.currentTime + 0.04);
  }

  public playSuccess() {
    if (this.isMuted) return;
    this.initCtx();
    if (!this.ctx) return;

    const time = this.ctx.currentTime;
    // Play two notes
    [600, 900].forEach((freq, i) => {
      if (!this.ctx) return;
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.type = "sine";
      osc.frequency.setValueAtTime(freq, time + i * 0.1);

      gain.gain.setValueAtTime(0.03, time + i * 0.1);
      gain.gain.exponentialRampToValueAtTime(0.0001, time + i * 0.1 + 0.15);

      osc.connect(gain);
      gain.connect(this.ctx.destination);

      osc.start(time + i * 0.1);
      osc.stop(time + i * 0.1 + 0.18);
    });
  }

  public toggleHum(active: boolean) {
    this.initCtx();
    if (!this.ctx) return;

    if (active && !this.isMuted) {
      if (this.humOsc) return;
      
      this.humOsc = this.ctx.createOscillator();
      this.humGain = this.ctx.createGain();

      this.humOsc.type = "sawtooth";
      this.humOsc.frequency.setValueAtTime(55, this.ctx.currentTime); // low frequency hum (A1 note)
      
      // Filter high frequencies
      const filter = this.ctx.createBiquadFilter();
      filter.type = "lowpass";
      filter.frequency.setValueAtTime(110, this.ctx.currentTime);

      this.humGain.gain.setValueAtTime(0.015, this.ctx.currentTime);

      this.humOsc.connect(filter);
      filter.connect(this.humGain);
      this.humGain.connect(this.ctx.destination);

      this.humOsc.start();
    } else {
      if (this.humOsc) {
        try {
          this.humOsc.stop();
        } catch(e){}
        this.humOsc.disconnect();
        this.humGain?.disconnect();
        this.humOsc = null;
        this.humGain = null;
      }
    }
  }
}

const synth = new AudioSynth();

// --- TYPE DEF ---
interface ConsoleProps {
  onTriggerDataFlow: () => void;
  onSubsystemFocus: (subsystem: string) => void;
  onDriftTrigger: (active: boolean) => void;
}

interface LogLine {
  text: string;
  type: "prompt" | "system" | "success" | "warn" | "info" | "text";
}

export default function Console({ onTriggerDataFlow, onSubsystemFocus, onDriftTrigger }: ConsoleProps) {
  const [input, setInput] = useState("");
  const [logs, setLogs] = useState<LogLine[]>([
    { text: "NEXUS OS [Version 1.0.0] - Core Terminal Ingress", type: "system" },
    { text: "Type /help to query supported commands. AI assistant ready.", type: "info" },
  ]);
  const [loading, setLoading] = useState(false);
  const [muted, setMuted] = useState(true);
  const bodyRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Scroll to bottom on log updates
    if (bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [logs]);

  const handleMuteToggle = () => {
    const nextMuted = !muted;
    synth.isMuted = nextMuted;
    setMuted(nextMuted);
    synth.toggleHum(!nextMuted);
    synth.playClick();
  };

  const executeCommand = async (commandStr: string) => {
    setLoading(true);
    synth.playClick();
    
    // Echo command
    setLogs((prev) => [...prev, { text: `~ % ${commandStr}`, type: "prompt" }]);

    const cmd = commandStr.trim();
    
    // API Request
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "";
      const response = await fetch(`${apiUrl}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: cmd })
      });

      if (!response.ok) throw new Error("API Connection broken");
      
      const data = await response.json();
      
      // Hook up UI triggers based on the command executed
      if (data.command_executed === "/recommend") {
        onSubsystemFocus("recommender");
        onTriggerDataFlow();
      } else if (data.command_executed === "/search") {
        onSubsystemFocus("recommender"); // focus recommender/search node
        onTriggerDataFlow();
      } else if (data.command_executed === "/drift") {
        onSubsystemFocus("feature_store");
        onDriftTrigger(true);
      } else if (data.command_executed === "/train") {
        onSubsystemFocus("recommender");
        onDriftTrigger(false); // remove drift alert
      }

      // Stream the characters to create typing effect
      await streamResponse(data.response);
      synth.playSuccess();

    } catch (err) {
      setLogs((prev) => [
        ...prev,
        { text: "SYSTEM CORRUPTION: Failed to establish server contact. Falling back to local simulator cache...", type: "warn" }
      ]);
      // Local fallback simulator logic if server is offline
      await runLocalFallback(cmd);
    } finally {
      setLoading(false);
    }
  };

  const runLocalFallback = async (cmd: string) => {
    const cmdLower = cmd.toLowerCase();
    let reply = "";

    if (cmdLower.startsWith("/recommend")) {
      onSubsystemFocus("recommender");
      onTriggerDataFlow();
      reply = (
        "### [LOCAL SIMULATOR OUTPUT] Recommendation Pipeline\n" +
        "- **Target User**: `usr_guest_local`\n" +
        "- **Online Cache Lookup**: Retrieved 24 aggregates from Redis (1.4ms).\n" +
        "- **Candidate Retrieval**: bi-encoder similarity scan yielded 50 recommendations (3.2ms).\n" +
        "- **MMoE Score**: Calibrated CTR/CVR predictions via Triton (7.8ms).\n" +
        "- **Results**: `itm_electronics_9`, `itm_clothing_24`, `itm_home_15`\n" +
        "- **Latency**: `12.4 ms`"
      );
    } else if (cmdLower.startsWith("/search")) {
      onSubsystemFocus("recommender");
      onTriggerDataFlow();
      reply = (
        "### [LOCAL SIMULATOR OUTPUT] Semantic Search\n" +
        "- **Query**: *\"" + (cmd.split(" ")[1] || "laptops") + "\"*\n" +
        "- **FAISS Index Scan**: Dense vectors retrieved in 3.1ms.\n" +
        "- **LambdaMART Rerank**: Sorted list compiled in 4.9ms.\n" +
        "- **Results**: \n" +
        "  1. `laptop_high_performance` (score: 0.942)\n" +
        "  2. `keyboard_backlit_rgb` (score: 0.811)\n" +
        "- **Latency**: `8.0 ms`"
      );
    } else if (cmdLower.startsWith("/drift")) {
      onSubsystemFocus("feature_store");
      onDriftTrigger(true);
      reply = (
        "### [LOCAL SIMULATOR OUTPUT] Feature Drift Warning\n" +
        "- **Metrics Analyzed**: `user_purchase_rate`, `item_popularity`.\n" +
        "- **Population Stability Index (PSI)**: `0.26` (Threshold exceeded > 0.20)\n" +
        "- **KS statistic**: `0.091` (p-value: 0.0012)\n" +
        "🚨 **Alert**: Feature distribution skew detected. Execute model retraining via `/train`."
      );
    } else if (cmdLower.startsWith("/train")) {
      onSubsystemFocus("recommender");
      onDriftTrigger(false);
      reply = (
        "### [LOCAL SIMULATOR OUTPUT] Model Training Cycle\n" +
        "- **Compute Engine**: Ray on 4x Local GPUs.\n" +
        "- **Epoch 1/3** | Loss: 0.422 | Recall@100: 0.741\n" +
        "- **Epoch 2/3** | Loss: 0.189 | Recall@100: 0.812\n" +
        "- **Epoch 3/3** | Loss: 0.054 | Recall@100: 0.887\n" +
        "- **Export Status**: Registered model `nexus-recsys-v2.1.0` successfully.\n" +
        "🎉 Training complete. Shadows deployed and serving."
      );
    } else if (cmdLower === "/help") {
      reply = (
        "### Nexus OS Command Interface\n" +
        "Supported commands:\n" +
        "- `/recommend <user_id>` : Run recommender pipeline.\n" +
        "- `/search <query>` : Run dense semantic search.\n" +
        "- `/drift` : Run data drift analysis.\n" +
        "- `/train` : Simulate Model Retraining.\n" +
        "- `/help` : Show help menu."
      );
    } else {
      reply = (
        "### Nexus OS Controller Core\n" +
        "Marketplace intelligence system active.\n\n" +
        "Query: *\"" + cmd + "\"*\n\n" +
        "I am an advanced ML system. Type `/help` to see executable system commands or ask architectural questions (e.g. *\"Explain the feature store\"*)."
      );
    }

    await streamResponse(reply);
    synth.playSuccess();
  };

  const streamResponse = async (text: string) => {
    // Add an empty text log line
    setLogs((prev) => [...prev, { text: "", type: "text" }]);
    
    let currentText = "";
    const characters = text.split("");
    
    for (let i = 0; i < characters.length; i++) {
      currentText += characters[i];
      // Update the last log entry
      setLogs((prev) => {
        const next = [...prev];
        next[next.length - 1] = { text: currentText, type: "text" };
        return next;
      });
      // Play typing click sound
      if (i % 2 === 0) synth.playType();
      await new Promise((r) => setTimeout(r, 6)); // Stream speed
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;
    const command = input;
    setInput("");
    executeCommand(command);
  };

  const handlePresetClick = (preset: string) => {
    if (loading) return;
    executeCommand(preset);
  };

  return (
    <div className="flex flex-col h-[400px] w-full glass-panel rounded-xl border border-borderCustom overflow-hidden shadow-2xl relative">
      {/* Console Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-secondaryBg/80 border-b border-borderCustom">
        <div className="flex items-center gap-3">
          <Terminal className="w-4 h-4 text-accent" />
          <span className="hud-text text-xs text-secondaryText">NEXUS_OS // shell_gateway</span>
        </div>
        <div className="flex items-center gap-3">
          <button 
            onClick={handleMuteToggle}
            className="p-1.5 hover:bg-white/5 rounded-md text-secondaryText hover:text-accent transition-colors"
            title={muted ? "Unmute terminal" : "Mute terminal"}
          >
            {muted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
          </button>
          <div className="flex gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-red-500/80"></span>
            <span className="w-2.5 h-2.5 rounded-full bg-yellow-500/80"></span>
            <span className="w-2.5 h-2.5 rounded-full bg-green-500/80"></span>
          </div>
        </div>
      </div>

      {/* Terminal logs body */}
      <div 
        ref={bodyRef}
        className="flex-1 p-4 overflow-y-auto font-mono text-xs leading-relaxed space-y-3 bg-background/50 select-text"
      >
        {logs.map((log, idx) => {
          let color = "text-primaryText";
          if (log.type === "prompt") color = "text-accent font-semibold";
          if (log.type === "system") color = "text-purple-400 font-semibold";
          if (log.type === "info") color = "text-secondaryText";
          if (log.type === "warn") color = "text-yellow-400";
          if (log.type === "success") color = "text-green-400";
          
          // Render markdown subset for headers and list items in replies
          const formatText = (txt: string) => {
            if (txt.startsWith("###")) {
              return <div className="text-sm font-bold text-accent mt-2 mb-1 border-b border-white/5 pb-0.5">{txt.replace("###", "")}</div>;
            }
            if (txt.startsWith("-")) {
              return <div className="pl-2 text-secondaryText">{txt}</div>;
            }
            if (txt.startsWith("`[INFO]`")) {
              return <div className="text-secondaryText">{txt}</div>;
            }
            if (txt.startsWith("`[WARN]`") || txt.startsWith("🚨")) {
              return <div className="text-yellow-400 font-semibold">{txt}</div>;
            }
            if (txt.startsWith("🎉")) {
              return <div className="text-green-400 font-semibold">{txt}</div>;
            }
            return <div>{txt}</div>;
          };

          return (
            <div key={idx} className={`${color} whitespace-pre-wrap`}>
              {log.type === "text" ? formatText(log.text) : log.text}
            </div>
          );
        })}
        {loading && (
          <div className="flex items-center gap-2 text-accent">
            <Cpu className="w-3.5 h-3.5 animate-spin" />
            <span className="animate-pulse">Processing vector query...</span>
          </div>
        )}
      </div>

      {/* Preset Command quickchips */}
      <div className="flex flex-wrap gap-2 px-4 py-2 border-t border-borderCustom bg-secondaryBg/20">
        <button 
          onClick={() => handlePresetClick("/recommend user_75")}
          className="px-2 py-1 text-[10px] font-mono rounded bg-white/5 border border-white/5 hover:border-accent/40 text-secondaryText hover:text-accent transition-all"
        >
          /recommend
        </button>
        <button 
          onClick={() => handlePresetClick("/search electronics")}
          className="px-2 py-1 text-[10px] font-mono rounded bg-white/5 border border-white/5 hover:border-accent/40 text-secondaryText hover:text-accent transition-all"
        >
          /search
        </button>
        <button 
          onClick={() => handlePresetClick("/drift")}
          className="px-2 py-1 text-[10px] font-mono rounded bg-white/5 border border-white/5 hover:border-accent/40 text-secondaryText hover:text-accent transition-all"
        >
          /drift
        </button>
        <button 
          onClick={() => handlePresetClick("/train")}
          className="px-2 py-1 text-[10px] font-mono rounded bg-white/5 border border-white/5 hover:border-accent/40 text-secondaryText hover:text-accent transition-all"
        >
          /train
        </button>
      </div>

      {/* Form Input */}
      <form onSubmit={handleSubmit} className="flex border-t border-borderCustom">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Command console or query system..."
          disabled={loading}
          className="flex-1 px-4 py-3 bg-secondaryBg/40 font-mono text-xs text-primaryText focus:outline-none focus:bg-secondaryBg/60 placeholder-white/25 disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="px-4 bg-accent/15 border-l border-borderCustom hover:bg-accent/25 disabled:bg-transparent text-accent disabled:text-white/20 transition-all flex items-center justify-center"
        >
          <Send className="w-3.5 h-3.5" />
        </button>
      </form>
    </div>
  );
}
