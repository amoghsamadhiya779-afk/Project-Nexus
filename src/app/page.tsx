"use client";

import React, { useState, useEffect, useRef } from "react";
import Scene from "@/components/Scene";
import Console from "@/components/Console";
import LenisScroll from "@/components/LenisScroll";
import { Cpu, Server, Network, Shield, TrendingUp, RefreshCw, AlertTriangle, Play } from "lucide-react";
import gsap from "gsap";

export default function Home() {
  const [activeSubsystem, setActiveSubsystem] = useState("overview");
  const [flowActive, setFlowActive] = useState(false);
  const [driftWarning, setDriftWarning] = useState(false);
  const [telemetry, setTelemetry] = useState({
    redis_connected: true,
    cache_hit_rate: 98.2,
    ray_gpu_utilization: 64.0,
    inference_throughput: 4250,
    flink_status: "RUNNING",
    mlflow_status: "ACTIVE",
    active_alerts: [] as string[],
    system_latency_p99: 11.2,
  });

  const mouse = useRef<[number, number]>([0, 0]);
  const heroRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const titleRef = useRef<HTMLHeadingElement>(null);
  const tagRef = useRef<HTMLDivElement>(null);

  // Capture mouse coordinates for WebGL parallax drift
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      mouse.current = [
        (e.clientX / window.innerWidth) * 2 - 1,
        -(e.clientY / window.innerHeight) * 2 + 1,
      ];
    };
    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  // Live Telemetry poll loop
  useEffect(() => {
    const fetchTelemetry = async () => {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || "";
        const response = await fetch(`${apiUrl}/api/telemetry`);
        if (response.ok) {
          const data = await response.json();
          setTelemetry(data);
          if (data.active_alerts && data.active_alerts.length > 0) {
            setDriftWarning(true);
          }
        }
      } catch (e) {
        // Fallback local dynamic jitter if backend is offline
        setTelemetry((prev) => ({
          ...prev,
          ray_gpu_utilization: Math.round(55 + Math.random() * 12),
          inference_throughput: Math.round(4100 + Math.random() * 200),
          system_latency_p99: parseFloat((9.5 + Math.random() * 2.5).toFixed(2)),
        }));
      }
    };

    fetchTelemetry();
    const interval = setInterval(fetchTelemetry, 3000);
    return () => clearInterval(interval);
  }, []);

  // Premium GSAP Reveal Animation
  useEffect(() => {
    const tl = gsap.timeline();
    
    // Hide overlay at start
    gsap.set(overlayRef.current, { opacity: 0 });
    gsap.set(titleRef.current, { opacity: 0, scale: 0.85, letterSpacing: "-0.05em" });
    gsap.set(tagRef.current, { opacity: 0, y: 15 });

    // 1. Text reveals with smooth letters tracking
    tl.to(titleRef.current, {
      opacity: 1,
      scale: 1,
      letterSpacing: "0.15em",
      duration: 1.8,
      ease: "power4.out",
      delay: 0.5,
    });

    tl.to(tagRef.current, {
      opacity: 1,
      y: 0,
      duration: 1.0,
      ease: "power3.out",
    }, "-=0.8");
    
    // 1.5 Fade out hero text so it doesn't block the UI
    tl.to(heroRef.current, {
      opacity: 0,
      filter: "blur(10px)",
      duration: 1.0,
      ease: "power2.inOut",
      onComplete: () => {
        if (heroRef.current) heroRef.current.style.display = "none";
      }
    }, "+=1.2");

    // 2. Main interface fades into existence
    tl.to(overlayRef.current, {
      opacity: 1,
      duration: 1.4,
      ease: "cubic-bezier(0.22, 1, 0.36, 1)",
    }, "-=0.5");

    // 3. Slide camera slowly into position
    setActiveSubsystem("overview");
  }, []);

  const handleSubsystemClick = (subsystem: string) => {
    setActiveSubsystem(subsystem);
  };

  const handlePersonaSelect = (persona: string) => {
    // Inject preset inputs into the console system trigger
    const consoleTextarea = document.querySelector("input[placeholder*='Command console']") as HTMLInputElement;
    if (consoleTextarea) {
      consoleTextarea.value = `/recommend ${persona}`;
      // Trigger a click on submit button
      const submitBtn = consoleTextarea.nextElementSibling as HTMLButtonElement;
      if (submitBtn) submitBtn.click();
    }
  };

  return (
    <main className="relative min-h-screen w-full bg-background overflow-hidden grid-bg scanlines-overlay">
      <LenisScroll />
      
      {/* 3D WebGL starry canvas layer */}
      <Scene
        activeSubsystem={activeSubsystem}
        flowActive={flowActive}
        onFlowComplete={() => setFlowActive(false)}
        mouse={mouse}
      />

      {/* Cinematic Hero Title Reveal Section */}
      <div 
        ref={heroRef}
        className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none z-10 select-none"
      >
        <h1 
          ref={titleRef}
          className="text-6xl md:text-8xl font-black text-transparent bg-clip-text bg-gradient-to-r from-slate-100 via-blue-100 to-slate-200 uppercase tracking-widest text-center filter drop-shadow-[0_0_35px_rgba(96,165,250,0.18)]"
        >
          NEXUS OS
        </h1>
        <div 
          ref={tagRef}
          className="mt-4 font-mono text-xs uppercase tracking-[0.3em] text-accent/80 flex items-center gap-2"
        >
          <span className="w-1.5 h-1.5 rounded-full bg-accent animate-ping"></span>
          Deep-Space Marketplace Intelligence
        </div>
      </div>

      {/* Main HUD Overlays */}
      <div 
        ref={overlayRef}
        className="relative z-20 min-h-screen w-full max-w-[1600px] mx-auto flex flex-col justify-between p-3 sm:p-6 pointer-events-none"
      >
        {/* Header HUD panel */}
        <header className="w-full flex flex-col md:flex-row gap-4 md:items-center justify-between pointer-events-auto bg-secondaryBg/40 backdrop-blur-md p-4 rounded-xl border border-borderCustom shadow-hudGlow">
          <div className="flex items-center gap-3">
            <div className="w-6 h-6 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-md flex items-center justify-center shadow-btnGlow">
              <Network className="w-3.5 h-3.5 text-white" />
            </div>
            <span className="hud-text text-sm font-bold tracking-[0.15em] text-white">NEXUS // PLATFORM_PLANE</span>
          </div>

          {/* Drift warning banner */}
          {driftWarning && (
            <div className="flex items-center gap-2 px-3 py-1 bg-yellow-500/10 border border-yellow-500/30 rounded-full text-yellow-400 text-[10px] uppercase font-mono animate-pulse-slow">
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>Feature Drift Warning Raised</span>
            </div>
          )}

          <div className="flex flex-wrap items-center gap-4 sm:gap-6 text-[10px] font-mono text-secondaryText">
            <div className="flex items-center gap-2">
              <span>REDIS:</span>
              <span className={telemetry.redis_connected ? "text-green-400 font-semibold" : "text-red-400 font-semibold"}>
                {telemetry.redis_connected ? "CONNECTED" : "OFFLINE"}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span>FLINK:</span>
              <span className="text-green-400 font-semibold">{telemetry.flink_status}</span>
            </div>
            <div className="flex items-center gap-2">
              <span>MLFLOW:</span>
              <span className="text-green-400 font-semibold">{telemetry.mlflow_status}</span>
            </div>
          </div>
        </header>

        {/* Midground Grid: Telemetry (Left) + Console (Center) + Subsystems (Right) */}
        <div className="flex-1 my-4 sm:my-6 grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4 sm:gap-6 items-stretch">
          {/* Left panel: Telemetry stats */}
          <section className="md:col-span-1 lg:col-span-1 flex flex-col gap-4 pointer-events-auto">
            {/* telemetry metrics */}
            <div className="glass-panel p-4 rounded-xl space-y-4">
              <div className="hud-text text-[11px] text-accent border-b border-borderCustom pb-1.5 flex items-center gap-2">
                <Cpu className="w-3.5 h-3.5" />
                Live Telemetry
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-[10px] text-secondaryText font-mono uppercase">p99 Latency</div>
                  <div className="text-lg font-bold tracking-tight text-white mt-1">{telemetry.system_latency_p99} ms</div>
                </div>
                <div>
                  <div className="text-[10px] text-secondaryText font-mono uppercase">Hit Rate</div>
                  <div className="text-lg font-bold tracking-tight text-white mt-1">{telemetry.cache_hit_rate}%</div>
                </div>
                <div>
                  <div className="text-[10px] text-secondaryText font-mono uppercase">GPU load</div>
                  <div className="text-lg font-bold tracking-tight text-white mt-1">{telemetry.ray_gpu_utilization}%</div>
                </div>
                <div>
                  <div className="text-[10px] text-secondaryText font-mono uppercase">Throughput</div>
                  <div className="text-lg font-bold tracking-tight text-white mt-1">{telemetry.inference_throughput} RPS</div>
                </div>
              </div>
            </div>

            {/* active pipelines description (hidden on mobile to prevent scrolling bloat) */}
            <div className="glass-panel p-4 rounded-xl flex-1 hidden md:flex flex-col justify-between">
              <div className="hud-text text-[11px] text-accent border-b border-borderCustom pb-1.5 flex items-center gap-2">
                <Server className="w-3.5 h-3.5" />
                Active Pipelines
              </div>
              <div className="space-y-3 font-mono text-[10px] text-secondaryText my-4">
                <div className="flex items-center justify-between border-b border-white/5 pb-1">
                  <span>Inference Gateway</span>
                  <span className="text-green-400">200 OK</span>
                </div>
                <div className="flex items-center justify-between border-b border-white/5 pb-1">
                  <span>Kafka Message Stream</span>
                  <span className="text-green-400">12k events/s</span>
                </div>
                <div className="flex items-center justify-between border-b border-white/5 pb-1">
                  <span>Great Expectations</span>
                  <span className="text-green-400">Passing (14 rules)</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Ray Distributed Plane</span>
                  <span className="text-accent">Idle (v2.4.0)</span>
                </div>
              </div>
              <div className="text-[9px] font-mono text-secondaryText/60 bg-black/40 p-2.5 rounded border border-white/5 leading-normal">
                `SYSTEM_STAT`: serving production models in us-west-2 cluster. Auto-retraining trigger active.
              </div>
            </div>
          </section>

          {/* Center Column: Console Shell */}
          <section className="md:col-span-3 lg:col-span-2 flex flex-col justify-end pointer-events-auto">
            <Console 
              onTriggerDataFlow={() => setFlowActive(true)} 
              onSubsystemFocus={setActiveSubsystem}
              onDriftTrigger={setDriftWarning}
            />
          </section>

          {/* Right panel: Subsystems nav selector */}
          <section className="md:col-span-2 lg:col-span-1 flex flex-col justify-between gap-4 pointer-events-auto">
            <div className="glass-panel p-4 rounded-xl flex-1 flex flex-col justify-between">
              <div>
                <div className="hud-text text-[11px] text-accent border-b border-borderCustom pb-1.5 flex items-center gap-2">
                  <Network className="w-3.5 h-3.5" />
                  Subsystem Targets
                </div>
                <p className="text-[10px] text-secondaryText font-mono leading-relaxed mt-2.5 hidden sm:block">
                  Click coordinate targets to travel the 3D grid and isolate system components.
                </p>
              </div>

              {/* Subsystem targeting selectors - Responsive Grid on mobile */}
              <div className="grid grid-cols-2 sm:grid-cols-5 md:grid-cols-1 gap-2 my-4 md:my-6 md:space-y-2.5">
                {[
                  { id: "overview", label: "00 // Telemetry", icon: Server },
                  { id: "feature_store", label: "01 // Feature Store", icon: Cpu },
                  { id: "recommender", label: "02 // Recommender", icon: Network },
                  { id: "forecasting", label: "03 // Forecasting", icon: TrendingUp },
                  { id: "fraud", label: "04 // Fraud Network", icon: Shield }
                ].map((sys) => {
                  const Icon = sys.icon;
                  const isActive = activeSubsystem === sys.id;
                  return (
                    <button
                      key={sys.id}
                      onClick={() => handleSubsystemClick(sys.id)}
                      className={`flex items-center gap-2 sm:gap-3 px-3 py-2 sm:py-2.5 rounded-lg border text-left font-mono text-[9px] sm:text-[10px] transition-custom ${
                        isActive
                          ? "bg-accent/15 border-accent/60 text-white shadow-hudGlow font-semibold"
                          : "bg-white/5 border-white/5 hover:border-white/10 text-secondaryText hover:text-white"
                      }`}
                    >
                      <Icon className={`w-3.5 h-3.5 ${isActive ? "text-accent" : "text-secondaryText"}`} />
                      <span>{sys.label}</span>
                    </button>
                  );
                })}
              </div>

              <div className="border-t border-borderCustom pt-3 mt-auto flex items-center justify-between text-[9px] font-mono text-secondaryText">
                <span>Active Target</span>
                <span className="text-accent font-bold uppercase">{activeSubsystem}</span>
              </div>
            </div>
          </section>
        </div>

        {/* Footer HUD panel & Sandboxed Persona select */}
        <footer className="w-full flex flex-col md:flex-row items-center justify-between gap-4 pointer-events-auto bg-secondaryBg/20 border border-borderCustom p-4 rounded-xl backdrop-blur-md">
          <div className="flex flex-col gap-1 md:max-w-md w-full md:w-auto">
            <span className="hud-text text-[10px] text-accent flex items-center gap-1.5 font-semibold">
              <RefreshCw className="w-3 h-3 animate-pulse" />
              RecSys Sandbox
            </span>
            <span className="text-[10px] text-secondaryText font-mono leading-relaxed hidden sm:block">
              Inject mock buyer profiles into the active serving lane to inspect outputs.
            </span>
          </div>

          {/* Persona quick buttons - Responsive Grid on mobile */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 w-full md:flex md:w-auto">
            {[
              { id: "power_buyer", label: "Power Buyer" },
              { id: "tech_enthusiast", label: "Tech Fanatic" },
              { id: "cold_start", label: "Cold Start" }
            ].map((persona) => (
              <button
                key={persona.id}
                onClick={() => handlePersonaSelect(persona.id)}
                className="px-3.5 py-2 font-mono text-[10px] rounded bg-white/5 border border-white/5 hover:border-accent/30 hover:bg-accent/5 text-secondaryText hover:text-white transition-all flex items-center justify-center gap-2 w-full"
              >
                <Play className="w-2.5 h-2.5 text-accent" />
                {persona.label}
              </button>
            ))}
          </div>
        </footer>
      </div>
    </main>
  );
}
