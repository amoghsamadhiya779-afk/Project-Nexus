/** @type {import('next').NextConfig} */
const nextConfig = {
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

module.exports = nextConfig;
