"use client";

import React, { useRef, useEffect, useMemo } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Points, PointMaterial } from "@react-three/drei";
import * as THREE from "three";
import gsap from "gsap";

// --- CUSTOM SHADER FOR NEBULA CLOUD ---
const NebulaShader = {
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform float uTime;
    uniform vec3 uColor1;
    uniform vec3 uColor2;
    varying vec2 vUv;

    // Simplex 2D noise implementation
    vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }
    float snoise(vec2 v){
      const vec4 C = vec4(0.211324865405187, 0.366025403784439,
               -0.577350269189626, 0.024390243902439);
      vec2 i  = floor(v + dot(v, C.yy) );
      vec2 x0 = v -   i + dot(i, C.xx);
      vec2 i1;
      i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
      vec4 x12 = x0.xyxy + C.xxzz;
      x12.xy -= i1;
      i = mod(i, 289.0);
      vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
      + i.x + vec3(0.0, i1.x, 1.0 ));
      vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
        dot(x12.zw,x12.zw)), 0.0);
      m = m*m ;
      m = m*m ;
      vec3 x = 2.0 * fract(p * C.www) - 1.0;
      vec3 h = abs(x) - 0.5;
      vec3 a0 = x - floor(x + 0.5);
      vec3 g = sin(uTime * 0.1) * h + cos(uTime * 0.1) * a0;
      vec3 norm = 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
      vec3 g1;
      g1.x  = a0.x  * norm.x;
      g1.y  = a0.y  * norm.y;
      g1.z  = a0.z  * norm.z;
      vec2  g2 = vec2(g1.x, g1.y);
      float n = 130.0 * dot(m, vec3(dot(x0,g2), dot(x12.xy,g2), dot(x12.zw,g2)));
      return 0.5 + 0.5 * n;
    }

    void main() {
      vec2 uv1 = vUv * 1.5 - vec2(uTime * 0.015);
      vec2 uv2 = vUv * 0.8 + vec2(uTime * 0.008);
      
      float n1 = snoise(uv1);
      float n2 = snoise(uv2);
      
      float noiseMix = n1 * 0.6 + n2 * 0.4;
      
      // Calculate vignette / circular mask
      float dist = distance(vUv, vec2(0.5));
      float mask = smoothstep(0.7, 0.2, dist);
      
      vec3 color = mix(uColor1, uColor2, noiseMix);
      gl_FragColor = vec4(color, noiseMix * 0.18 * mask);
    }
  `
};

// --- DYNAMIC CAMERA CONTROLLER ---
function CameraController({ activeSubsystem, mouse }: { activeSubsystem: string; mouse: React.MutableRefObject<[number, number]> }) {
  const { camera, size } = useThree();
  const targetLook = useRef(new THREE.Vector3(0, 0, 0));
  const isMobile = size.width < 768;

  // Subsystem 3D coordinates configuration with mobile offsets
  const viewCoordinates: Record<string, { pos: [number, number, number]; lookAt: [number, number, number] }> = {
    overview: { pos: [0, 0, isMobile ? 17 : 12], lookAt: [0, 0, 0] },
    feature_store: { pos: [isMobile ? -2.0 : -5, 2, isMobile ? 5 : 4], lookAt: [isMobile ? -2.5 : -6, 2, 0] },
    recommender: { pos: [isMobile ? 2.0 : 5, 2, isMobile ? 5 : 4], lookAt: [isMobile ? 2.5 : 6, 2, 0] },
    forecasting: { pos: [isMobile ? -2.0 : -5, -2, isMobile ? 5 : 4], lookAt: [isMobile ? -2.5 : -6, -2, 0] },
    fraud: { pos: [isMobile ? 2.0 : 5, -2, isMobile ? 5 : 4], lookAt: [isMobile ? 2.5 : 6, -2, 0] },
  };

  useEffect(() => {
    const coord = viewCoordinates[activeSubsystem] || viewCoordinates.overview;
    
    // Animate camera position and target direction using GSAP
    gsap.killTweensOf(camera.position);
    gsap.killTweensOf(targetLook.current);

    gsap.to(camera.position, {
      x: coord.pos[0],
      y: coord.pos[1],
      z: coord.pos[2],
      duration: 2.2,
      ease: "power3.inOut",
    });

    gsap.to(targetLook.current, {
      x: coord.lookAt[0],
      y: coord.lookAt[1],
      z: coord.lookAt[2],
      duration: 2.2,
      ease: "power3.inOut",
    });
  }, [activeSubsystem, camera, isMobile]);

  useFrame(() => {
    // Inject smooth mouse parallax drift (disable or damp on mobile screens to save GPU cycle)
    const driftX = isMobile ? 0 : mouse.current[0] * 0.8;
    const driftY = isMobile ? 0 : mouse.current[1] * 0.4;
    
    const lerpedLook = new THREE.Vector3().copy(targetLook.current);
    lerpedLook.x += driftX;
    lerpedLook.y += driftY;

    camera.lookAt(lerpedLook);
  });

  return null;
}

// --- STARS BACKGROUND SYSTEM ---
function StarField({ mouse }: { mouse: React.MutableRefObject<[number, number]> }) {
  const pointsRef = useRef<THREE.Points>(null);
  const { size } = useThree();
  const isMobile = size.width < 768;

  const starData = useMemo(() => {
    // Halve star count on mobile for 60fps performance
    const count = isMobile ? 1200 : 3000;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 45;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 45;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 45;
    }
    return positions;
  }, [isMobile]);

  useFrame((state) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y = state.clock.getElapsedTime() * 0.005;
      pointsRef.current.rotation.x = state.clock.getElapsedTime() * 0.002;
      
      if (!isMobile) {
        pointsRef.current.position.x += (mouse.current[0] * 0.5 - pointsRef.current.position.x) * 0.05;
        pointsRef.current.position.y += (mouse.current[1] * 0.25 - pointsRef.current.position.y) * 0.05;
      }
    }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[starData, 3]}
        />
      </bufferGeometry>
      <pointMaterial
        color="#ffffff"
        size={isMobile ? 0.045 : 0.06}
        sizeAttenuation={true}
        transparent={true}
        opacity={0.6}
        depthWrite={false}
      />
    </points>
  );
}

// --- SHADER-DRIVEN NEBULA CLOUD ---
function Nebula() {
  const meshRef = useRef<THREE.Mesh>(null);
  
  const uniforms = useMemo(() => ({
    uTime: { value: 0 },
    uColor1: { value: new THREE.Color("#1e3a8a") }, // Deep blue
    uColor2: { value: new THREE.Color("#4c1d95") }  // Indigo purple
  }), []);

  useFrame((state) => {
    if (meshRef.current) {
      const material = meshRef.current.material as THREE.ShaderMaterial;
      material.uniforms.uTime.value = state.clock.getElapsedTime();
    }
  });

  return (
    <mesh ref={meshRef} position={[0, 0, -5]}>
      <planeGeometry args={[35, 25]} />
      <shaderMaterial
        vertexShader={NebulaShader.vertexShader}
        fragmentShader={NebulaShader.fragmentShader}
        uniforms={uniforms}
        transparent={true}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </mesh>
  );
}

// --- FEATURE STORE 3D MODEL (ROTATING BLOCKS GRID) ---
function FeatureStoreModel({ active }: { active: boolean }) {
  const { size } = useThree();
  const isMobile = size.width < 768;
  const groupRef = useRef<THREE.Group>(null);
  const cubeCount = isMobile ? 8 : 12;

  const positions = useMemo(() => {
    const coords: [number, number, number][] = [];
    for (let i = 0; i < cubeCount; i++) {
      const angle = (i / cubeCount) * Math.PI * 2;
      const radius = isMobile ? 0.9 : 1.4;
      coords.push([
        Math.cos(angle) * radius,
        (Math.random() - 0.5) * 0.6,
        Math.sin(angle) * radius
      ]);
    }
    return coords;
  }, [cubeCount, isMobile]);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.getElapsedTime() * 0.25;
      
      const scaleMultiplier = active 
        ? (isMobile ? 0.65 : 1.0) * (1.0 + Math.sin(state.clock.getElapsedTime() * 4) * 0.08) 
        : (isMobile ? 0.65 : 1.0);
      groupRef.current.scale.setScalar(scaleMultiplier);
    }
  });

  const modelPosX = isMobile ? -2.5 : -6;
  const modelPosY = isMobile ? 2.0 : 2;

  return (
    <group ref={groupRef} position={[modelPosX, modelPosY, 0]}>
      <mesh>
        <sphereGeometry args={[0.2, 12, 12]} />
        <meshBasicMaterial color={active ? "#60a5fa" : "#3b82f6"} wireframe />
      </mesh>
      {positions.map((pos, idx) => (
        <mesh key={idx} position={pos}>
          <boxGeometry args={[isMobile ? 0.16 : 0.22, isMobile ? 0.16 : 0.22, isMobile ? 0.16 : 0.22]} />
          <meshBasicMaterial
            color={active ? "#93c5fd" : "#2563eb"}
            wireframe={idx % 2 === 0}
            transparent
            opacity={active ? 0.9 : 0.6}
          />
        </mesh>
      ))}
    </group>
  );
}

// --- RECOMMENDER 3D MODEL (TWO TOWER NODES SCHEMA) ---
function RecommenderModel({ active }: { active: boolean }) {
  const { size } = useThree();
  const isMobile = size.width < 768;
  const groupRef = useRef<THREE.Group>(null);
  
  const userPoints = useMemo(() => Array.from({ length: isMobile ? 5 : 8 }, () => new THREE.Vector3(
    (isMobile ? -0.5 : -0.8) + (Math.random() - 0.5) * 0.3,
    (Math.random() - 0.5) * (isMobile ? 1.0 : 1.5),
    (Math.random() - 0.5) * 0.3
  )), [isMobile]);

  const itemPoints = useMemo(() => Array.from({ length: isMobile ? 5 : 8 }, () => new THREE.Vector3(
    (isMobile ? 0.5 : 0.8) + (Math.random() - 0.5) * 0.3,
    (Math.random() - 0.5) * (isMobile ? 1.0 : 1.5),
    (Math.random() - 0.5) * 0.3
  )), [isMobile]);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.getElapsedTime() * 0.15;
      
      const scaleVal = active 
        ? (isMobile ? 0.65 : 1.0) * (1.0 + Math.sin(state.clock.getElapsedTime() * 5) * 0.05) 
        : (isMobile ? 0.65 : 1.0);
      groupRef.current.scale.setScalar(scaleVal);
    }
  });

  const modelPosX = isMobile ? 2.5 : 6;
  const modelPosY = isMobile ? 2.0 : 2;

  return (
    <group ref={groupRef} position={[modelPosX, modelPosY, 0]}>
      <group>
        {userPoints.map((pt, idx) => (
          <mesh key={idx} position={pt}>
            <sphereGeometry args={[isMobile ? 0.06 : 0.08, 6, 6]} />
            <meshBasicMaterial color="#60a5fa" transparent opacity={active ? 1.0 : 0.6} />
          </mesh>
        ))}
      </group>
      <group>
        {itemPoints.map((pt, idx) => (
          <mesh key={idx} position={pt}>
            <sphereGeometry args={[isMobile ? 0.06 : 0.08, 6, 6]} />
            <meshBasicMaterial color="#a7f3d0" transparent opacity={active ? 1.0 : 0.6} />
          </mesh>
        ))}
      </group>
      {userPoints.map((u, i) => (
        <LineBetween key={i} from={u} to={itemPoints[i % itemPoints.length]} active={active} />
      ))}
    </group>
  );
}

function LineBetween({ from, to, active }: { from: THREE.Vector3; to: THREE.Vector3; active: boolean }) {
  const lineRef = useRef<THREE.Line>(null);
  const points = useMemo(() => [from, to], [from, to]);
  const geometry = useMemo(() => new THREE.BufferGeometry().setFromPoints(points), [points]);

  return (
    <line ref={lineRef} geometry={geometry}>
      <lineBasicMaterial
        color={active ? "#60a5fa" : "#3b82f6"}
        transparent
        opacity={active ? 0.35 : 0.1}
      />
    </line>
  );
}

// --- CAUSAL FORECASTING 3D MODEL (WAVY PARAMETRIC RIBBONS) ---
function ForecastingModel({ active }: { active: boolean }) {
  const { size } = useThree();
  const isMobile = size.width < 768;
  const meshRef = useRef<THREE.Line>(null);
  
  const points = useMemo(() => {
    const arr = [];
    const count = isMobile ? 25 : 40;
    for (let i = 0; i < count; i++) {
      const x = (i / count) * (isMobile ? 2.0 : 3.0) - (isMobile ? 1.0 : 1.5);
      const y = Math.sin((i / count) * Math.PI * 4) * 0.3;
      arr.push(new THREE.Vector3(x, y, 0));
    }
    return arr;
  }, [isMobile]);

  const geometry = useMemo(() => new THREE.BufferGeometry().setFromPoints(points), [points]);

  useFrame((state) => {
    if (meshRef.current) {
      const positions = meshRef.current.geometry.attributes.position.array as Float32Array;
      const time = state.clock.getElapsedTime();
      
      const count = points.length;
      for (let i = 0; i < count; i++) {
        const factor = active ? 6.0 : 2.0;
        const wave = Math.sin((i / count) * Math.PI * 4 + time * factor) * 0.3;
        positions[i * 3 + 1] = wave;
      }
      meshRef.current.geometry.attributes.position.needsUpdate = true;
    }
  });

  const modelPosX = isMobile ? -2.5 : -6;
  const modelPosY = isMobile ? -2.0 : -2;
  const modelScale = isMobile ? 0.65 : 1.0;

  return (
    <group position={[modelPosX, modelPosY, 0]} scale={modelScale}>
      <line ref={meshRef} geometry={geometry}>
        <lineBasicMaterial
          color={active ? "#34d399" : "#059669"}
          linewidth={2}
          transparent
          opacity={active ? 1.0 : 0.5}
        />
      </line>
      <mesh position={[isMobile ? -1.0 : -1.5, 0, 0]}>
        <sphereGeometry args={[0.08, 6, 6]} />
        <meshBasicMaterial color="#34d399" />
      </mesh>
      <mesh position={[isMobile ? 1.0 : 1.5, 0, 0]}>
        <sphereGeometry args={[0.08, 6, 6]} />
        <meshBasicMaterial color="#34d399" />
      </mesh>
    </group>
  );
}

// --- GRAPH FRAUD DETECTION 3D MODEL (NODAL NETWORK) ---
function FraudModel({ active }: { active: boolean }) {
  const { size } = useThree();
  const isMobile = size.width < 768;
  const groupRef = useRef<THREE.Group>(null);

  const nodes = useMemo(() => [
    { pos: [0, 0.6, 0], color: "#ef4444" },
    { pos: [-0.6, 0, 0.3], color: "#3b82f6" },
    { pos: [0.6, 0, -0.3], color: "#3b82f6" },
    { pos: [-0.3, -0.5, -0.4], color: "#3b82f6" },
    { pos: [0.3, -0.5, 0.4], color: "#3b82f6" },
  ], []);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.getElapsedTime() * 0.2;
      
      const redNode = groupRef.current.children[0] as THREE.Mesh;
      if (redNode && active) {
        const scale = 1.0 + Math.sin(state.clock.getElapsedTime() * 6) * 0.2;
        redNode.scale.setScalar(scale);
      }
    }
  });

  const modelPosX = isMobile ? 2.5 : 6;
  const modelPosY = isMobile ? -2.0 : -2;
  const modelScale = isMobile ? 0.65 : 1.0;

  return (
    <group ref={groupRef} position={[modelPosX, modelPosY, 0]} scale={modelScale}>
      {nodes.map((node, idx) => (
        <mesh key={idx} position={node.pos as [number, number, number]}>
          <sphereGeometry args={[idx === 0 ? 0.11 : 0.06, 8, 8]} />
          <meshBasicMaterial
            color={node.color}
            transparent
            opacity={active ? 1.0 : 0.6}
          />
        </mesh>
      ))}
      <LineBetween from={new THREE.Vector3(0, 0.6, 0)} to={new THREE.Vector3(-0.6, 0, 0.3)} active={active} />
      <LineBetween from={new THREE.Vector3(0, 0.6, 0)} to={new THREE.Vector3(0.6, 0, -0.3)} active={active} />
      <LineBetween from={new THREE.Vector3(-0.6, 0, 0.3)} to={new THREE.Vector3(-0.3, -0.5, -0.4)} active={active} />
      <LineBetween from={new THREE.Vector3(0.6, 0, -0.3)} to={new THREE.Vector3(0.3, -0.5, 0.4)} active={active} />
      <LineBetween from={new THREE.Vector3(-0.3, -0.5, -0.4)} to={new THREE.Vector3(0.3, -0.5, 0.4)} active={active} />
    </group>
  );
}

// --- DATA FLOW PACKETS ANIMATION ---
function DataFlowParticles({ flowActive, onFlowComplete }: { flowActive: boolean; onFlowComplete: () => void }) {
  const { size } = useThree();
  const isMobile = size.width < 768;
  const meshRef = useRef<THREE.Mesh>(null);
  
  const pathPoints = useMemo(() => {
    const startX = isMobile ? -2.5 : -6;
    const startY = isMobile ? 2.0 : 2;
    const endZ = isMobile ? 6 : 4;
    const curve = new THREE.QuadraticBezierCurve3(
      new THREE.Vector3(startX, startY, 0),
      new THREE.Vector3(0, 3, 1),
      new THREE.Vector3(0, 0, endZ)
    );
    return curve.getPoints(50);
  }, [isMobile]);

  const progress = useRef(0);

  useFrame((state) => {
    if (flowActive && meshRef.current) {
      progress.current += state.delta * 0.85;
      if (progress.current >= 1.0) {
        progress.current = 0;
        onFlowComplete();
      } else {
        const pointIdx = Math.floor(progress.current * (pathPoints.length - 1));
        const currentPt = pathPoints[pointIdx];
        if (currentPt) {
          meshRef.current.position.copy(currentPt);
          const size = Math.sin(progress.current * Math.PI) * (isMobile ? 0.12 : 0.2) + 0.03;
          meshRef.current.scale.setScalar(size);
        }
      }
    }
  });

  if (!flowActive) return null;

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[1, 12, 12]} />
      <meshBasicMaterial color="#60a5fa" transparent opacity={0.9} />
    </mesh>
  );
}

// --- MAIN WEBGL SCENE WRAPPER ---
interface SceneProps {
  activeSubsystem: string;
  flowActive: boolean;
  onFlowComplete: () => void;
  mouse: React.MutableRefObject<[number, number]>;
}

export default function Scene({ activeSubsystem, flowActive, onFlowComplete, mouse }: SceneProps) {
  return (
    <div className="absolute inset-0 z-0 bg-background pointer-events-none">
      <Canvas
        gl={{ antialias: true, alpha: false }}
        camera={{ position: [0, 0, 12], fov: 60 }}
      >
        <color attach="background" args={["#030712"]} />
        <ambientLight intensity={1.5} />
        
        <StarField mouse={mouse} />
        <Nebula />
        
        <FeatureStoreModel active={activeSubsystem === "feature_store"} />
        <RecommenderModel active={activeSubsystem === "recommender"} />
        <ForecastingModel active={activeSubsystem === "forecasting"} />
        <FraudModel active={activeSubsystem === "fraud"} />

        <CameraController activeSubsystem={activeSubsystem} mouse={mouse} />
        
        <DataFlowParticles flowActive={flowActive} onFlowComplete={onFlowComplete} />
      </Canvas>
    </div>
  );
}
