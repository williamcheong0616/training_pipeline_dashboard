/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  async rewrites() {
    // API_URL is a server-side-only var (no NEXT_PUBLIC_ prefix) so it is NOT
    // inlined at build time. Next.js reads it at server startup instead, which
    // means the Docker service hostname "api" resolves correctly at runtime.
    const apiUrl = process.env.API_URL ?? "http://api:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${apiUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
