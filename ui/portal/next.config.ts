import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  transpilePackages: ["three"],
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8080/api/:path*",
      },
      {
        source: "/recommend",
        destination: "http://localhost:8080/recommend",
      },
      {
        source: "/search",
        destination: "http://localhost:8080/search",
      },
    ];
  },
};

export default nextConfig;
